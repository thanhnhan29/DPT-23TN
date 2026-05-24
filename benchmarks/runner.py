from __future__ import annotations

import csv
from dataclasses import replace
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
from benchmarks.decoder_impls import decoder_attention_full_sequence
from benchmarks.metrics import clear_memory, measure_attention, measurement_to_row
from benchmarks.metrics import measure_workload

OOM_ERROR = "OOM"
RUNTIME_ERROR = "ERR"
SKIP_ERROR = "SKIP"


CSV_FIELDS = (
    "experiment",
    "scenario",
    "phase",
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
    "sweep_value",
    "baseline_method",
    "speedup_vs_baseline",
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
    experiment: str,
    scenario: str,
    phase: str,
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
        "experiment": experiment,
        "scenario": scenario,
        "phase": phase,
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
        "sweep_value": seq_len,
        "baseline_method": "",
        "speedup_vs_baseline": "",
    }


def run_benchmarks(config: BenchmarkConfig) -> list[dict[str, Any]]:
    device = choose_device()
    dtype = choose_dtype(device)
    rows: list[dict[str, Any]] = []

    if "full_attention" in config.scenarios:
        rows.extend(run_full_attention_benchmarks(config, device, dtype))
    if "kv_decode" in config.scenarios:
        rows.extend(run_kv_decode_benchmarks(config, device, dtype))
    if "batch_size" in config.scenarios:
        rows.extend(run_batch_size_benchmarks(config, device, dtype))
    if "head_dim" in config.scenarios:
        rows.extend(run_head_dim_benchmarks(config, device, dtype))
    if "prefill_decode" in config.scenarios:
        rows.extend(run_prefill_decode_benchmarks(config, device, dtype))

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
                experiment="runtime_memory_scaling",
                scenario="full_attention",
                phase="attention",
                method=method,
                device=device,
                dtype=dtype,
                config=config,
                seq_len=seq_len,
            )

            if _should_skip_quadratic_baseline(method, seq_len, config):
                row.update(_empty_measurement_columns())
                row.update({"status": "skipped", "error": SKIP_ERROR})
                rows.append(row)
                continue
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
                row.update(_exception_status(exc))
            except RuntimeError as exc:
                clear_memory(device)
                row.update(_empty_measurement_columns())
                row.update(_exception_status(exc))
            except KeyError as exc:
                clear_memory(device)
                row.update(_empty_measurement_columns())
                row.update({"status": "error", "error": RUNTIME_ERROR})

            rows.append(row)

        del q, k, v
        clear_memory(device)

    _attach_speedups(rows, baseline_method="naive", group_keys=("seq_len", "batch_size", "head_dim"))
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

    for context_len in config.context_lengths:
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
                experiment="kv_cache",
                scenario="kv_decode",
                phase="decode",
                method=method,
                device=device,
                dtype=dtype,
                config=config,
                seq_len=context_len,
                context_len=context_len,
                decode_tokens=config.decode_tokens,
            )

            if _should_skip_quadratic_decode(method, context_len, config):
                row.update(_empty_measurement_columns())
                row.update({"status": "skipped", "error": SKIP_ERROR})
                rows.append(row)
                continue

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
                row.update(_exception_status(exc))
            except RuntimeError as exc:
                clear_memory(device)
                row.update(_empty_measurement_columns())
                row.update(_exception_status(exc))

            rows.append(row)

        del context, new_tokens
        clear_memory(device)

    del weights
    clear_memory(device)
    return rows


def run_batch_size_benchmarks(
    config: BenchmarkConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for batch_size in config.batch_sizes:
        batch_config = replace(
            config,
            batch_size=batch_size,
            seq_lengths=(config.fixed_seq_len,),
        )
        rows.extend(run_full_attention_benchmarks(batch_config, device, dtype))

    for row in rows:
        row["experiment"] = "batch_size_sensitivity"
        row["scenario"] = "batch_size"
        row["sweep_value"] = row["batch_size"]

    _attach_speedups(rows, baseline_method="naive", group_keys=("seq_len", "batch_size"))
    return rows


def run_head_dim_benchmarks(
    config: BenchmarkConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for head_dim in config.head_dims:
        dim_config = replace(
            config,
            head_dim=head_dim,
            seq_lengths=(config.fixed_seq_len,),
        )
        rows.extend(run_full_attention_benchmarks(dim_config, device, dtype))

    for row in rows:
        row["experiment"] = "head_dim_scaling"
        row["scenario"] = "head_dim"
        row["sweep_value"] = row["head_dim"]

    _attach_speedups(rows, baseline_method="naive", group_keys=("seq_len", "head_dim"))
    return rows


def run_prefill_decode_benchmarks(
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
        seed=config.seed + 40_000,
    )

    for context_len in config.context_lengths:
        context = make_hidden_states(
            batch_size=config.batch_size,
            seq_len=context_len,
            model_dim=model_dim,
            device=device,
            dtype=dtype,
            seed=config.seed + context_len + 50_000,
        )
        new_tokens = make_hidden_states(
            batch_size=config.batch_size,
            seq_len=config.decode_tokens,
            model_dim=model_dim,
            device=device,
            dtype=dtype,
            seed=config.seed + context_len + 60_000,
        )

        if _should_skip_quadratic_prefill_decode("no_cache", context_len, config):
            rows.extend(
                _skipped_prefill_decode_rows(
                    config=config,
                    device=device,
                    dtype=dtype,
                    context_len=context_len,
                    method="no_cache",
                )
            )
        else:
            rows.extend(
                _measure_prefill_decode_case(
                    config=config,
                    device=device,
                    dtype=dtype,
                    context=context,
                    new_tokens=new_tokens,
                    weights=weights,
                    context_len=context_len,
                    method="no_cache",
                    prefill_setup_fn=lambda: lambda: decoder_attention_full_sequence(
                        context, weights, config.num_heads
                    ),
                    decode_setup_fn=lambda: lambda: decode_no_cache(
                        context, new_tokens, weights, config.num_heads
                    ),
                    cache_memory_mb=0.0,
                )
            )

        rows.extend(
            _measure_prefill_decode_case(
                config=config,
                device=device,
                dtype=dtype,
                context=context,
                new_tokens=new_tokens,
                weights=weights,
                context_len=context_len,
                method="kv_cache",
                prefill_setup_fn=lambda: lambda: build_kv_cache(
                    context, weights, config.num_heads
                ),
                decode_setup_fn=_decode_setup_fn(
                    method="decode_kv_cache",
                    context=context,
                    new_tokens=new_tokens,
                    weights=weights,
                    num_heads=config.num_heads,
                )[0],
                cache_memory_mb=_cache_memory_mb(context, weights, config.num_heads),
            )
        )

        del context, new_tokens
        clear_memory(device)

    del weights
    clear_memory(device)
    _attach_speedups(rows, baseline_method="no_cache", group_keys=("context_len", "phase"))
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


def _measure_prefill_decode_case(
    *,
    config: BenchmarkConfig,
    device: torch.device,
    dtype: torch.dtype,
    context: torch.Tensor,
    new_tokens: torch.Tensor,
    weights: Any,
    context_len: int,
    method: str,
    prefill_setup_fn,
    decode_setup_fn,
    cache_memory_mb: float,
) -> list[dict[str, Any]]:
    del context, new_tokens, weights
    rows: list[dict[str, Any]] = []

    for phase, setup_fn, phase_decode_tokens in (
        ("prefill", prefill_setup_fn, context_len),
        ("decode", decode_setup_fn, config.decode_tokens),
    ):
        row = base_row(
            experiment="prefill_decode",
            scenario="prefill_decode",
            phase=phase,
            method=method,
            device=device,
            dtype=dtype,
            config=config,
            seq_len=context_len,
            context_len=context_len,
            decode_tokens=phase_decode_tokens,
        )
        row["sweep_value"] = context_len

        try:
            clear_memory(device)
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
            row.update(_exception_status(exc))
        except RuntimeError as exc:
            clear_memory(device)
            row.update(_empty_measurement_columns())
            row.update(_exception_status(exc))

        rows.append(row)

    return rows


def _skipped_prefill_decode_rows(
    *,
    config: BenchmarkConfig,
    device: torch.device,
    dtype: torch.dtype,
    context_len: int,
    method: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for phase, phase_decode_tokens in (
        ("prefill", context_len),
        ("decode", config.decode_tokens),
    ):
        row = base_row(
            experiment="prefill_decode",
            scenario="prefill_decode",
            phase=phase,
            method=method,
            device=device,
            dtype=dtype,
            config=config,
            seq_len=context_len,
            context_len=context_len,
            decode_tokens=phase_decode_tokens,
        )
        row["sweep_value"] = context_len
        row.update(_empty_measurement_columns())
        row.update({"status": "skipped", "error": SKIP_ERROR})
        rows.append(row)
    return rows


def _cache_memory_mb(context: torch.Tensor, weights: Any, num_heads: int) -> float:
    sample_cache = build_kv_cache(context, weights, num_heads)
    cache_memory_mb = cache_nbytes(sample_cache) / (1024**2)
    del sample_cache
    return cache_memory_mb


def _derived_columns(row: dict[str, Any], *, cache_memory_mb: float) -> dict[str, float | str]:
    median_time_ms = row.get("median_time_ms")
    decode_tokens = row.get("decode_tokens")
    if median_time_ms == "":
        return {
            "time_per_token_ms": "",
            "tokens_per_sec": "",
            "cache_memory_mb": cache_memory_mb,
        }

    token_count = int(decode_tokens) if decode_tokens not in ("", 0) else _token_count(row)
    if token_count <= 0:
        return {
            "time_per_token_ms": "",
            "tokens_per_sec": "",
            "cache_memory_mb": cache_memory_mb,
        }

    time_per_token_ms = float(median_time_ms) / token_count
    return {
        "time_per_token_ms": time_per_token_ms,
        "tokens_per_sec": 1000.0 * token_count / float(median_time_ms)
        if float(median_time_ms) > 0
        else "",
        "cache_memory_mb": cache_memory_mb,
    }


def _token_count(row: dict[str, Any]) -> int:
    seq_len = row.get("seq_len")
    batch_size = row.get("batch_size")
    if seq_len in ("", None) or batch_size in ("", None):
        return 0
    return int(seq_len) * int(batch_size)


def _attach_speedups(
    rows: list[dict[str, Any]],
    *,
    baseline_method: str,
    group_keys: tuple[str, ...],
) -> None:
    baselines: dict[tuple[Any, ...], float] = {}

    for row in rows:
        if row.get("status") != "ok" or row.get("method") != baseline_method:
            continue
        median_time_ms = row.get("median_time_ms")
        if median_time_ms in ("", None):
            continue
        key = tuple(row.get(group_key) for group_key in group_keys)
        baselines[key] = float(median_time_ms)

    for row in rows:
        row["baseline_method"] = baseline_method
        row["speedup_vs_baseline"] = ""
        if row.get("status") != "ok":
            continue
        median_time_ms = row.get("median_time_ms")
        if median_time_ms in ("", None):
            continue
        key = tuple(row.get(group_key) for group_key in group_keys)
        baseline = baselines.get(key)
        if baseline and float(median_time_ms) > 0:
            row["speedup_vs_baseline"] = baseline / float(median_time_ms)


def _should_skip_quadratic_baseline(
    method: str,
    seq_len: int,
    config: BenchmarkConfig,
) -> bool:
    return method == "naive" and seq_len > config.baseline_max_size


def _should_skip_quadratic_decode(
    method: str,
    context_len: int,
    config: BenchmarkConfig,
) -> bool:
    return method == "decode_no_cache" and context_len > config.baseline_max_size


def _should_skip_quadratic_prefill_decode(
    method: str,
    context_len: int,
    config: BenchmarkConfig,
) -> bool:
    return method == "no_cache" and context_len > config.baseline_max_size


def _exception_status(exc: BaseException) -> dict[str, str]:
    message = str(exc).lower()
    if isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in message:
        return {"status": "oom", "error": OOM_ERROR}
    return {"status": "error", "error": RUNTIME_ERROR}


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
        "baseline_method": "",
        "speedup_vs_baseline": "",
    }


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
