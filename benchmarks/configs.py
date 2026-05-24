from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


RUNTIME_SEQ_LENGTHS = (128, 256, 512, 1024, 2048, 4096, 8192, 16384)
KV_CONTEXT_LENGTHS = (1024, 2048, 4096, 8192, 16384)
STRESS_SEQ_LENGTHS = (128, 256, 512, 1024, 2048, 4096, 8192, 10000, 20000, 30000)
STRESS_CONTEXT_LENGTHS = (1024, 2048, 4096, 8192, 10000, 20000, 30000)
BATCH_SIZES = (1, 2, 4, 8, 16)
HEAD_DIMS = (32, 64, 128, 256)
DEFAULT_BASELINE_MAX_SIZE = 8192


@dataclass(frozen=True)
class BenchmarkConfig:
    batch_size: int = 1
    num_heads: int = 8
    head_dim: int = 64
    seq_lengths: tuple[int, ...] = (128, 256, 512, 1024)
    context_lengths: tuple[int, ...] = (128, 256, 512, 1024)
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
    batch_sizes: tuple[int, ...] = BATCH_SIZES
    head_dims: tuple[int, ...] = HEAD_DIMS
    fixed_seq_len: int = 4096
    baseline_max_size: int = DEFAULT_BASELINE_MAX_SIZE
