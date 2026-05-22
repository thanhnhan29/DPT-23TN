from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkConfig:
    batch_size: int = 1
    num_heads: int = 8
    head_dim: int = 64
    seq_lengths: tuple[int, ...] = (128, 256, 512, 1024)
    warmup_runs: int = 5
    measure_runs: int = 20
    causal: bool = True
    seed: int = 42
    output_path: Path = Path("results/benchmark_results.csv")
    methods: tuple[str, ...] = field(default_factory=lambda: ("naive", "sdpa"))
    scenarios: tuple[str, ...] = field(default_factory=lambda: ("full_attention", "kv_decode"))
    decode_tokens: int = 1
    decode_methods: tuple[str, ...] = field(
        default_factory=lambda: ("decode_no_cache", "decode_kv_cache")
    )
