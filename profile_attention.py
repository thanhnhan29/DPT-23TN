from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from benchmarks.attention_impls import ATTENTION_METHODS
from benchmarks.runner import choose_device, choose_dtype, make_qkv

OOM_ERROR = "OOM"
RUNTIME_ERROR = "ERR"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile attention kernels with PyTorch profiler.")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=("naive", "sdpa", "flash_attn"),
    )
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--no-causal", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device()
    dtype = choose_dtype(device)
    q, k, v = make_qkv(
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        device=device,
        dtype=dtype,
        seed=1234,
    )

    rows: list[dict[str, str | float | int]] = []
    for method in args.methods:
        try:
            attention_fn = ATTENTION_METHODS[method]
            with torch.inference_mode():
                attention_fn(q, k, v, not args.no_causal)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)

            activities = [torch.profiler.ProfilerActivity.CPU]
            if device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)

            with torch.profiler.profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            ) as prof:
                with torch.inference_mode():
                    attention_fn(q, k, v, not args.no_causal)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)

            trace_path = output_dir / f"profiler_trace_{method}_N{args.seq_len}.json"
            prof.export_chrome_trace(str(trace_path))

            key_averages = prof.key_averages()
            self_time_us = sum(
                _self_device_time_us(item, device.type)
                for item in key_averages
            )
            rows.append(
                {
                    "method": method,
                    "device": device.type,
                    "seq_len": args.seq_len,
                    "batch_size": args.batch_size,
                    "num_heads": args.num_heads,
                    "head_dim": args.head_dim,
                    "self_time_ms": self_time_us / 1000.0,
                    "trace_path": str(trace_path),
                    "status": "ok",
                    "error": "",
                }
            )
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            status, error = _exception_status(exc)
            rows.append(
                {
                    "method": method,
                    "device": device.type,
                    "seq_len": args.seq_len,
                    "batch_size": args.batch_size,
                    "num_heads": args.num_heads,
                    "head_dim": args.head_dim,
                    "self_time_ms": "",
                    "trace_path": "",
                    "status": status,
                    "error": error,
                }
            )

    summary_path = output_dir / "profiler_summary.csv"
    with summary_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote profiler summary to {summary_path}")


def _exception_status(exc: BaseException) -> tuple[str, str]:
    message = str(exc).lower()
    if isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in message:
        return "oom", OOM_ERROR
    return "error", RUNTIME_ERROR


def _self_device_time_us(item: object, device_type: str) -> float:
    if device_type == "cuda":
        for attr_name in (
            "self_cuda_time_total",
            "self_device_time_total",
            "self_gpu_time_total",
        ):
            value = getattr(item, attr_name, None)
            if value is not None:
                return float(value)

    return float(getattr(item, "self_cpu_time_total", 0.0))


if __name__ == "__main__":
    main()
