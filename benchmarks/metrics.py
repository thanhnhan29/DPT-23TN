from __future__ import annotations

import gc
import statistics
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from benchmarks.attention_impls import AttentionFn


@dataclass(frozen=True)
class Measurement:
    mean_time_ms: float
    median_time_ms: float
    min_time_ms: float
    max_time_ms: float
    peak_memory_mb: float


Workload = Callable[[], Any]
SetupFn = Callable[[], Workload]


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def clear_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024**2)

    _, peak_bytes = tracemalloc.get_traced_memory()
    return peak_bytes / (1024**2)


def measure_attention(
    attention_fn: AttentionFn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    warmup_runs: int,
    measure_runs: int,
) -> Measurement:
    return measure_workload(
        setup_fn=lambda: lambda: attention_fn(q, k, v, causal),
        device=q.device,
        warmup_runs=warmup_runs,
        measure_runs=measure_runs,
    )


def measure_workload(
    *,
    setup_fn: SetupFn,
    device: torch.device,
    warmup_runs: int,
    measure_runs: int,
) -> Measurement:
    with torch.inference_mode():
        for _ in range(warmup_runs):
            workload = setup_fn()
            output = workload()
            del output, workload
        synchronize_if_needed(device)

        clear_memory(device)
        if device.type == "cpu":
            tracemalloc.start()

        times_ms: list[float] = []
        for _ in range(measure_runs):
            workload = setup_fn()
            synchronize_if_needed(device)
            start = time.perf_counter()
            output = workload()
            synchronize_if_needed(device)
            times_ms.append((time.perf_counter() - start) * 1000)

            # Keep the op alive until after synchronization, then release it.
            del output, workload

        peak_memory_mb = _peak_memory_mb(device)
        if device.type == "cpu":
            tracemalloc.stop()

    return Measurement(
        mean_time_ms=statistics.fmean(times_ms),
        median_time_ms=statistics.median(times_ms),
        min_time_ms=min(times_ms),
        max_time_ms=max(times_ms),
        peak_memory_mb=peak_memory_mb,
    )


def measurement_to_row(measurement: Measurement) -> dict[str, Any]:
    return {
        "mean_time_ms": measurement.mean_time_ms,
        "median_time_ms": measurement.median_time_ms,
        "min_time_ms": measurement.min_time_ms,
        "max_time_ms": measurement.max_time_ms,
        "peak_memory_mb": measurement.peak_memory_mb,
    }
