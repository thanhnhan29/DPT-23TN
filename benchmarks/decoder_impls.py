from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DecoderWeights:
    w_q: torch.Tensor
    w_k: torch.Tensor
    w_v: torch.Tensor
    w_o: torch.Tensor


@dataclass(frozen=True)
class KVCache:
    k: torch.Tensor
    v: torch.Tensor


DecodeFn = Callable[
    [torch.Tensor, torch.Tensor, DecoderWeights, int, bool],
    torch.Tensor,
]


def make_decoder_weights(
    *,
    model_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> DecoderWeights:
    """Create deterministic projection weights with a scaled init."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    scale = 1.0 / math.sqrt(model_dim)
    return DecoderWeights(
        w_q=torch.randn(model_dim, model_dim, device=device, dtype=dtype, generator=generator) * scale,
        w_k=torch.randn(model_dim, model_dim, device=device, dtype=dtype, generator=generator) * scale,
        w_v=torch.randn(model_dim, model_dim, device=device, dtype=dtype, generator=generator) * scale,
        w_o=torch.randn(model_dim, model_dim, device=device, dtype=dtype, generator=generator) * scale,
    )


def project_heads(x: torch.Tensor, weight: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Project [B, S, D] -> [B, H, S, Hd] using a single linear layer."""
    batch_size, seq_len, model_dim = x.shape
    head_dim = model_dim // num_heads
    projected = x @ weight
    return projected.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """Merge heads [B, H, S, Hd] -> [B, S, D]."""
    batch_size, num_heads, seq_len, head_dim = x.shape
    return x.transpose(1, 2).contiguous().view(batch_size, seq_len, num_heads * head_dim)


def _scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
) -> torch.Tensor:
    """Compute scaled dot-product attention.

    Shapes: q [B, H, Tq, Hd], k/v [B, H, Tk, Hd].
    """
    scale = 1.0 / math.sqrt(q.size(-1))
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if causal:
        seq_len = q.size(-2)
        key_len = k.size(-2)
        mask = torch.ones(seq_len, key_len, device=q.device, dtype=torch.bool).triu(1 + key_len - seq_len)
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def causal_self_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Causal self-attention for full sequences."""
    return _scaled_dot_product_attention(q, k, v, causal=True)


def decoder_attention_full_sequence(x: torch.Tensor, weights: DecoderWeights, num_heads: int) -> torch.Tensor:
    """Full causal self-attention over the entire prefix."""
    q = project_heads(x, weights.w_q, num_heads)
    k = project_heads(x, weights.w_k, num_heads)
    v = project_heads(x, weights.w_v, num_heads)
    out = causal_self_attention(q, k, v)
    return merge_heads(out) @ weights.w_o


def build_kv_cache(
    context: torch.Tensor,
    weights: DecoderWeights,
    num_heads: int,
) -> KVCache:
    """Cache K/V for the prefix context so decode steps can reuse them."""
    return KVCache(
        k=project_heads(context, weights.w_k, num_heads).contiguous(),
        v=project_heads(context, weights.w_v, num_heads).contiguous(),
    )


def decoder_step_with_cache(
    token: torch.Tensor,
    cache: KVCache,
    weights: DecoderWeights,
    num_heads: int,
) -> tuple[torch.Tensor, KVCache]:
    """Decode a single token using the cached prefix K/V."""
    q = project_heads(token, weights.w_q, num_heads)
    k_new = project_heads(token, weights.w_k, num_heads)
    v_new = project_heads(token, weights.w_v, num_heads)
    k = torch.cat((cache.k, k_new), dim=-2)
    v = torch.cat((cache.v, v_new), dim=-2)

    # Single-token query: no causal mask needed.
    out = _scaled_dot_product_attention(q, k, v, causal=False)
    return merge_heads(out) @ weights.w_o, KVCache(k=k, v=v)


def decode_no_cache(
    context: torch.Tensor,
    new_tokens: torch.Tensor,
    weights: DecoderWeights,
    num_heads: int,
    return_all: bool = False,
) -> torch.Tensor:
    """Autoregressive decode by recomputing full attention each step."""
    prefix = context
    outputs: list[torch.Tensor] = []

    for idx in range(new_tokens.size(1)):
        prefix = torch.cat((prefix, new_tokens[:, idx : idx + 1]), dim=1)
        out = decoder_attention_full_sequence(prefix, weights, num_heads)[:, -1:]
        outputs.append(out)

    decoded = torch.cat(outputs, dim=1)
    return decoded if return_all else decoded[:, -1:]


def decode_kv_cache(
    context: torch.Tensor,
    new_tokens: torch.Tensor,
    weights: DecoderWeights,
    num_heads: int,
    return_all: bool = False,
) -> torch.Tensor:
    """Same outputs as `decode_no_cache`, but reuses cached K/V for the prefix."""
    cache = build_kv_cache(context, weights, num_heads)
    outputs: list[torch.Tensor] = []

    for idx in range(new_tokens.size(1)):
        out, cache = decoder_step_with_cache(new_tokens[:, idx : idx + 1], cache, weights, num_heads)
        outputs.append(out)

    decoded = torch.cat(outputs, dim=1)
    return decoded if return_all else decoded[:, -1:]


def cache_nbytes(cache: KVCache) -> int:
    """Return cache size in bytes."""
    return cache.k.numel() * cache.k.element_size() + cache.v.numel() * cache.v.element_size()


DECODE_METHODS: dict[str, DecodeFn] = {
    "decode_no_cache": decode_no_cache,
    "decode_kv_cache": decode_kv_cache,
}

