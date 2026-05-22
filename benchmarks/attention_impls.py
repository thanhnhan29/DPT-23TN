from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F

AttentionFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, bool], torch.Tensor]


def naive_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    """Reference implementation that explicitly materializes the NxN scores."""
    scale = 1.0 / math.sqrt(q.size(-1))
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    if causal:
        seq_len = q.size(-2)
        mask = torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    """PyTorch SDPA; on CUDA it can dispatch to memory-efficient/Flash kernels."""
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def flash_sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    """Force PyTorch SDPA to use the Flash Attention backend when available."""
    if q.device.type != "cuda":
        raise RuntimeError("flash_sdpa requires CUDA because Flash Attention is a GPU kernel.")

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError as exc:
        raise RuntimeError("This PyTorch version does not expose sdpa_kernel.") from exc

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


ATTENTION_METHODS: dict[str, AttentionFn] = {
    "naive": naive_attention,
    "sdpa": sdpa_attention,
    "flash_sdpa": flash_sdpa_attention,
}
