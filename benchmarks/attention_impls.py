from __future__ import annotations

import math
from collections.abc import Callable
from importlib import import_module

import torch
import torch.nn.functional as F

AttentionFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, bool], torch.Tensor]


def flash_sdpa_supported(device: torch.device) -> bool:
    if device.type != "cuda":
        return False

    major, minor = torch.cuda.get_device_capability(device)
    return (major, minor) >= (8, 0)


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
    """Force PyTorch SDPA to use the Flash Attention backend; never fallback."""
    if not flash_sdpa_supported(q.device):
        raise RuntimeError("flash_sdpa requires a CUDA device with compute capability sm80+.")

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError as exc:
        raise RuntimeError("This PyTorch version does not expose sdpa_kernel.") from exc

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def flash_attn_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    """Call the external flash-attn package; never fallback to PyTorch SDPA."""
    if q.device.type != "cuda":
        raise RuntimeError("flash_attn requires CUDA.")

    q_bshd = q.transpose(1, 2).contiguous()
    k_bshd = k.transpose(1, 2).contiguous()
    v_bshd = v.transpose(1, 2).contiguous()
    out = _flash_attn_func(q_bshd, k_bshd, v_bshd, causal=causal)
    return out.transpose(1, 2).contiguous()


def _flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
) -> torch.Tensor:
    interface = _flash_attn_interface()
    packed_func = getattr(interface, "flash_attn_func", None)
    if packed_func is not None:
        return packed_func(
            q,
            k,
            v,
            dropout_p=0.0,
            softmax_scale=None,
            causal=causal,
        )

    unpadded_func = (
        getattr(interface, "flash_attn_unpadded_func", None)
        or getattr(interface, "flash_attn_varlen_func", None)
    )
    if unpadded_func is None:
        raise RuntimeError("flash-attn is installed, but no supported attention API was found.")

    batch_size, seq_len, num_heads, head_dim = q.shape
    q_flat = q.reshape(batch_size * seq_len, num_heads, head_dim)
    k_flat = k.reshape(batch_size * seq_len, num_heads, head_dim)
    v_flat = v.reshape(batch_size * seq_len, num_heads, head_dim)
    cu_seqlens = torch.arange(
        0,
        (batch_size + 1) * seq_len,
        seq_len,
        device=q.device,
        dtype=torch.int32,
    )
    return unpadded_func(
        q_flat,
        k_flat,
        v_flat,
        cu_seqlens,
        cu_seqlens,
        seq_len,
        seq_len,
        0.0,
        softmax_scale=None,
        causal=causal,
    ).reshape(batch_size, seq_len, num_heads, head_dim)


def _flash_attn_interface():
    try:
        return import_module("flash_attn.flash_attn_interface")
    except ImportError:
        try:
            return import_module("flash_attn")
        except ImportError as exc:
            raise RuntimeError(
                "flash_attn requires the external flash-attn package. "
                "Install a build compatible with your CUDA/PyTorch."
            ) from exc



ATTENTION_METHODS: dict[str, AttentionFn] = {
    "naive": naive_attention,
    "sdpa": sdpa_attention,
    "flash_sdpa": flash_sdpa_attention,
    "flash_attn": flash_attn_attention,
    "flash_attn_v1": flash_attn_attention,
}
