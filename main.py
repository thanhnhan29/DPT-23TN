from __future__ import annotations

import argparse
import sys

from benchmarks.configs import (
    BATCH_SIZES,
    DEFAULT_BASELINE_MAX_SIZE,
    HEAD_DIMS,
    KV_CONTEXT_LENGTHS,
    RUNTIME_SEQ_LENGTHS,
    STRESS_CONTEXT_LENGTHS,
    STRESS_SEQ_LENGTHS,
    BenchmarkConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark attention time and memory.")
    parser.add_argument(
        "--preset",
        choices=("smoke", "must-have", "cv"),
        default="smoke",
        help="Preconfigured experiment set. Use cv for all project experiments.",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=None,
        help="Cap stress sequence/context lengths, e.g. --max-size 30000.",
    )
    parser.add_argument(
        "--baseline-max-size",
        type=int,
        default=DEFAULT_BASELINE_MAX_SIZE,
        help="Skip quadratic baselines above this sequence/context length.",
    )
    parser.add_argument(
        "--seq-lengths",
        type=int,
        nargs="+",
        default=None,
        help="Sequence lengths to benchmark, e.g. --seq-lengths 128 256 512.",
    )
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=None,
        help="Context lengths for KV-cache and prefill/decode benchmarks.",
    )
    parser.add_argument("--batch-size", type=int, default=BenchmarkConfig.batch_size)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Batch sizes for the batch_size scenario.",
    )
    parser.add_argument("--num-heads", type=int, default=BenchmarkConfig.num_heads)
    parser.add_argument("--head-dim", type=int, default=BenchmarkConfig.head_dim)
    parser.add_argument(
        "--head-dims",
        type=int,
        nargs="+",
        default=None,
        help="Head dimensions for the head_dim scenario.",
    )
    parser.add_argument("--fixed-seq-len", type=int, default=None)
    parser.add_argument("--warmup-runs", type=int, default=BenchmarkConfig.warmup_runs)
    parser.add_argument("--measure-runs", type=int, default=BenchmarkConfig.measure_runs)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Full-attention methods. Available: naive, sdpa, flash_sdpa.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help=(
            "Benchmark scenarios. Available: full_attention, kv_decode, "
            "batch_size, head_dim, prefill_decode."
        ),
    )
    parser.add_argument(
        "--decode-methods",
        nargs="+",
        default=None,
        help="KV decode methods. Available: decode_no_cache, decode_kv_cache.",
    )
    parser.add_argument("--decode-tokens", type=int, default=BenchmarkConfig.decode_tokens)
    parser.add_argument("--output", default=str(BenchmarkConfig.output_path))
    parser.add_argument("--no-causal", action="store_true")
    return parser.parse_args()


def preset_config(args: argparse.Namespace) -> BenchmarkConfig:
    if args.preset == "must-have":
        return BenchmarkConfig(
            seq_lengths=RUNTIME_SEQ_LENGTHS,
            context_lengths=KV_CONTEXT_LENGTHS,
            methods=("naive", "sdpa", "flash_sdpa"),
            scenarios=("full_attention", "kv_decode"),
        )

    if args.preset == "cv":
        return BenchmarkConfig(
            seq_lengths=RUNTIME_SEQ_LENGTHS,
            context_lengths=KV_CONTEXT_LENGTHS,
            methods=("naive", "sdpa", "flash_sdpa"),
            scenarios=(
                "full_attention",
                "kv_decode",
                "batch_size",
                "head_dim",
                "prefill_decode",
            ),
            batch_sizes=BATCH_SIZES,
            head_dims=HEAD_DIMS,
            fixed_seq_len=4096,
        )

    return BenchmarkConfig()


def lengths_up_to(candidates: tuple[int, ...], max_size: int | None) -> tuple[int, ...]:
    if max_size is None:
        return candidates

    lengths = tuple(length for length in candidates if length <= max_size)
    if lengths:
        return lengths
    return (max_size,)


def main() -> None:
    args = parse_args()
    preset = preset_config(args)
    preset_seq_lengths = (
        lengths_up_to(STRESS_SEQ_LENGTHS, args.max_size)
        if args.max_size is not None
        else preset.seq_lengths
    )
    preset_context_lengths = (
        lengths_up_to(STRESS_CONTEXT_LENGTHS, args.max_size)
        if args.max_size is not None
        else preset.context_lengths
    )
    config = BenchmarkConfig(
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        seq_lengths=tuple(args.seq_lengths) if args.seq_lengths else preset_seq_lengths,
        context_lengths=tuple(args.context_lengths)
        if args.context_lengths
        else preset_context_lengths,
        warmup_runs=args.warmup_runs,
        measure_runs=args.measure_runs,
        causal=not args.no_causal,
        output_path=BenchmarkConfig.output_path.__class__(args.output),
        methods=tuple(args.methods) if args.methods else preset.methods,
        scenarios=tuple(args.scenarios) if args.scenarios else preset.scenarios,
        decode_tokens=args.decode_tokens,
        decode_methods=tuple(args.decode_methods)
        if args.decode_methods
        else preset.decode_methods,
        batch_sizes=tuple(args.batch_sizes) if args.batch_sizes else preset.batch_sizes,
        head_dims=tuple(args.head_dims) if args.head_dims else preset.head_dims,
        fixed_seq_len=args.fixed_seq_len if args.fixed_seq_len else preset.fixed_seq_len,
        baseline_max_size=args.baseline_max_size,
    )

    try:
        from benchmarks.runner import run_benchmarks, write_csv
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            print(
                "Missing dependency: torch. Install dependencies with `pip install -r requirements.txt`.",
                file=sys.stderr,
            )
            raise SystemExit(1) from exc
        raise

    rows = run_benchmarks(config)
    write_csv(rows, config.output_path)

    print(f"Wrote {len(rows)} benchmark rows to {config.output_path}")
    for row in rows:
        if row["status"] == "ok":
            print(
                f"{row['scenario']:>14} | {row['method']:>15} | N={row['seq_len']:<5} | "
                f"{float(row['median_time_ms']):>8.3f} ms | "
                f"{float(row['peak_memory_mb']):>8.2f} MB"
            )
        else:
            print(
                f"{row['scenario']:>14} | {row['method']:>15} | "
                f"N={row['seq_len']:<5} | {row['status']} | {row['error']}"
            )


if __name__ == "__main__":
    main()
