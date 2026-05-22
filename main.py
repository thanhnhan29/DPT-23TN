from __future__ import annotations

import argparse
import sys

from benchmarks.configs import BenchmarkConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark attention time and memory.")
    parser.add_argument(
        "--seq-lengths",
        type=int,
        nargs="+",
        default=None,
        help="Sequence lengths to benchmark, e.g. --seq-lengths 128 256 512.",
    )
    parser.add_argument("--batch-size", type=int, default=BenchmarkConfig.batch_size)
    parser.add_argument("--num-heads", type=int, default=BenchmarkConfig.num_heads)
    parser.add_argument("--head-dim", type=int, default=BenchmarkConfig.head_dim)
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
        help="Benchmark scenarios. Available: full_attention, kv_decode.",
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


def main() -> None:
    args = parse_args()
    config = BenchmarkConfig(
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        seq_lengths=tuple(args.seq_lengths) if args.seq_lengths else BenchmarkConfig.seq_lengths,
        warmup_runs=args.warmup_runs,
        measure_runs=args.measure_runs,
        causal=not args.no_causal,
        output_path=BenchmarkConfig.output_path.__class__(args.output),
        methods=tuple(args.methods) if args.methods else BenchmarkConfig().methods,
        scenarios=tuple(args.scenarios) if args.scenarios else BenchmarkConfig().scenarios,
        decode_tokens=args.decode_tokens,
        decode_methods=tuple(args.decode_methods)
        if args.decode_methods
        else BenchmarkConfig().decode_methods,
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
