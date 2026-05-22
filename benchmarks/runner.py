from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch

from benchmarks.attention_impls import ATTENTION_METHODS
from benchmarks.configs import BenchmarkConfig
from benchmarks.decoder_impls import (
    build_kv_cache,
    cache_nbytes,
    decode_no_cache,
    decoder_step_with_cache,
    make_decoder_weights,
)
from benchmarks.metrics import clear_memory, measure_attention, measurement_to_row
from benchmarks.metrics import measure_workload


CSV_FIELDS = (
    "scenario",
    "method",
    "device",
    "dtype",
    "batch_size",
    "num_heads",
    "seq_len",
    "context_len",
    "decode_tokens",
    "head_dim",
    "model_dim",
    "causal",
    "warmup_runs",
    "measure_runs",
    "mean_time_ms",
    "median_time_ms",
    "min_time_ms",
    "max_time_ms",
    "time_per_token_ms",
    "tokens_per_sec",
    "peak_memory_mb",
    "cache_memory_mb",
    "status",
    "error",
)


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def choose_dtype(device: torch.device) -> torch.dtype:
    return torch.float16 if device.type == "cuda" else torch.float32


def make_qkv(
    *,
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    shape = (batch_size, num_heads, seq_len, head_dim)
    q = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    k = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    v = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    return q, k, v


def make_hidden_states(
    *,
    batch_size: int,
    seq_len: int,
    model_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.randn(
        (batch_size, seq_len, model_dim),
        device=device,
        dtype=dtype,
        generator=generator,
    )


def base_row(
    *,
    scenario: str,
    method: str,
    device: torch.device,
    dtype: torch.dtype,
    config: BenchmarkConfig,
    seq_len: int,
    context_len: int | str = "",
    decode_tokens: int | str = "",
) -> dict[str, Any]:
    model_dim = config.num_heads * config.head_dim
    return {
        "scenario": scenario,
        "method": method,
        "device": device.type,
        "dtype": str(dtype).replace("torch.", ""),
        "batch_size": config.batch_size,
        "num_heads": config.num_heads,
        "seq_len": seq_len,
        "context_len": context_len,
        "decode_tokens": decode_tokens,
        "head_dim": config.head_dim,
        "model_dim": model_dim,
        "causal": config.causal,
        "warmup_runs": config.warmup_runs,
        "measure_runs": config.measure_runs,
    }


def run_benchmarks(config: BenchmarkConfig) -> list[dict[str, Any]]:
    device = choose_device()
    dtype = choose_dtype(device)
    rows: list[dict[str, Any]] = []

    if "full_attention" in config.scenarios:
        rows.extend(run_full_attention_benchmarks(config, device, dtype))
    if "kv_decode" in config.scenarios:
        rows.extend(run_kv_decode_benchmarks(config, device, dtype))

    return rows


def run_full_attention_benchmarks(
    config: BenchmarkConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for seq_len in config.seq_lengths:
        q, k, v = make_qkv(
            batch_size=config.batch_size,
            num_heads=config.num_heads,
            seq_len=seq_len,
            head_dim=config.head_dim,
            device=device,
            dtype=dtype,
            seed=config.seed + seq_len,
        )

        for method in config.methods:
            row = base_row(
                scenario="full_attention",
                method=method,
                device=device,
                dtype=dtype,
                config=config,
                seq_len=seq_len,
            )

            try:
                attention_fn = ATTENTION_METHODS[method]
                clear_memory(device)
                measurement = measure_attention(
                    attention_fn,
                    q,
                    k,
                    v,
                    causal=config.causal,
                    warmup_runs=config.warmup_runs,
                    measure_runs=config.measure_runs,
                )
                row.update(measurement_to_row(measurement))
                row.update(_derived_columns(row, cache_memory_mb=0.0))
                row.update({"status": "ok", "error": ""})
            except torch.cuda.OutOfMemoryError as exc:
                clear_memory(device)
                row.update(_empty_measurement_columns())
                row.update({"status": "oom", "error": str(exc).splitlines()[0]})
            except RuntimeError as exc:
                clear_memory(device)
                row.update(_empty_measurement_columns())
                row.update({"status": "error", "error": str(exc).splitlines()[0]})
            except KeyError as exc:
                clear_memory(device)
                row.update(_empty_measurement_columns())
                row.update({"status": "error", "error": f"Unknown attention method: {exc}"})

            rows.append(row)

        del q, k, v
        clear_memory(device)

    return rows


def run_kv_decode_benchmarks(
    config: BenchmarkConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model_dim = config.num_heads * config.head_dim
    weights = make_decoder_weights(
        model_dim=model_dim,
        device=device,
        dtype=dtype,
        seed=config.seed + 10_000,
    )

    for context_len in config.seq_lengths:
        context = make_hidden_states(
            batch_size=config.batch_size,
            seq_len=context_len,
            model_dim=model_dim,
            device=device,
            dtype=dtype,
            seed=config.seed + context_len + 20_000,
        )
        new_tokens = make_hidden_states(
            batch_size=config.batch_size,
            seq_len=config.decode_tokens,
            model_dim=model_dim,
            device=device,
            dtype=dtype,
            seed=config.seed + context_len + 30_000,
        )

        for method in config.decode_methods:
            row = base_row(
                scenario="kv_decode",
                method=method,
                device=device,
                dtype=dtype,
                config=config,
                seq_len=context_len,
                context_len=context_len,
                decode_tokens=config.decode_tokens,
            )

            try:
                clear_memory(device)
                setup_fn, cache_memory_mb = _decode_setup_fn(
                    method=method,
                    context=context,
                    new_tokens=new_tokens,
                    weights=weights,
                    num_heads=config.num_heads,
                )
                measurement = measure_workload(
                    setup_fn=setup_fn,
                    device=device,
                    warmup_runs=config.warmup_runs,
                    measure_runs=config.measure_runs,
                )
                row.update(measurement_to_row(measurement))
                row.update(_derived_columns(row, cache_memory_mb=cache_memory_mb))
                row.update({"status": "ok", "error": ""})
            except torch.cuda.OutOfMemoryError as exc:
                clear_memory(device)
                row.update(_empty_measurement_columns())
                row.update({"status": "oom", "error": str(exc).splitlines()[0]})
            except RuntimeError as exc:
                clear_memory(device)
                row.update(_empty_measurement_columns())
                row.update({"status": "error", "error": str(exc).splitlines()[0]})

            rows.append(row)

        del context, new_tokens
        clear_memory(device)

    del weights
    clear_memory(device)
    return rows


def _decode_setup_fn(
    *,
    method: str,
    context: torch.Tensor,
    new_tokens: torch.Tensor,
    weights: Any,
    num_heads: int,
):
    if method == "decode_no_cache":
        def setup_no_cache():
            return lambda: decode_no_cache(context, new_tokens, weights, num_heads)

        return setup_no_cache, 0.0

    if method == "decode_kv_cache":
        sample_cache = build_kv_cache(context, weights, num_heads)
        cache_memory_mb = cache_nbytes(sample_cache) / (1024**2)
        del sample_cache

        def setup_kv_cache():
            cache = build_kv_cache(context, weights, num_heads)

            def workload():
                local_cache = cache
                output = None
                for idx in range(new_tokens.size(1)):
                    output, local_cache = decoder_step_with_cache(
                        new_tokens[:, idx : idx + 1],
                        local_cache,
                        weights,
                        num_heads,
                    )
                return output

            return workload

        return setup_kv_cache, cache_memory_mb

    raise RuntimeError(f"Unknown decode method: {method}")


def _derived_columns(row: dict[str, Any], *, cache_memory_mb: float) -> dict[str, float | str]:
    median_time_ms = row.get("median_time_ms")
    decode_tokens = row.get("decode_tokens")
    if median_time_ms == "" or decode_tokens in ("", 0):
        return {
            "time_per_token_ms": "",
            "tokens_per_sec": "",
            "cache_memory_mb": cache_memory_mb,
        }

    time_per_token_ms = float(median_time_ms) / int(decode_tokens)
    return {
        "time_per_token_ms": time_per_token_ms,
        "tokens_per_sec": 1000.0 / time_per_token_ms if time_per_token_ms > 0 else "",
        "cache_memory_mb": cache_memory_mb,
    }


def _empty_measurement_columns() -> dict[str, str]:
    return {
        "mean_time_ms": "",
        "median_time_ms": "",
        "min_time_ms": "",
        "max_time_ms": "",
        "time_per_token_ms": "",
        "tokens_per_sec": "",
        "peak_memory_mb": "",
        "cache_memory_mb": "",
    }


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
