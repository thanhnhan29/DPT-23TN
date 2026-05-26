from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate benchmark figures from CSV results.")
    parser.add_argument("--input", default="results/benchmark_results.csv")
    parser.add_argument("--output-dir", default="figures")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))

    for row in rows:
        for key in (
            "seq_len",
            "context_len",
            "batch_size",
            "num_heads",
            "head_dim",
            "model_dim",
            "mean_time_ms",
            "median_time_ms",
            "min_time_ms",
            "max_time_ms",
            "peak_memory_mb",
            "time_per_token_ms",
            "tokens_per_sec",
            "cache_memory_mb",
            "speedup_vs_baseline",
            "self_time_ms",
        ):
            row[key] = _number_or_blank(row.get(key, ""))
    return rows


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / ".matplotlib-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    rows = read_rows(input_path)

    import matplotlib.pyplot as plt

    _line_plot(
        rows,
        output_dir / "fig1_runtime_vs_sequence_length.png",
        title="Runtime vs Sequence Length",
        experiment="runtime_memory_scaling",
        x_key="seq_len",
        y_key="median_time_ms",
        y_label="Median time (ms)",
        loglog=True,
    )
    _line_plot(
        rows,
        output_dir / "fig2_peak_memory_vs_sequence_length.png",
        title="Peak Memory vs Sequence Length",
        experiment="runtime_memory_scaling",
        x_key="seq_len",
        y_key="peak_memory_mb",
        y_label="Peak memory (MB)",
        loglog=True,
        mark_oom=True,
    )
    _line_plot(
        rows,
        output_dir / "fig3_speedup_ratio.png",
        title="Speedup vs Naive",
        experiment="runtime_memory_scaling",
        x_key="seq_len",
        y_key="speedup_vs_baseline",
        y_label="Speedup ratio",
        loglog=False,
    )
    _line_plot(
        rows,
        output_dir / "fig9_tokens_per_sec_vs_sequence_length.png",
        title="Throughput vs Sequence Length",
        experiment="runtime_memory_scaling",
        x_key="seq_len",
        y_key="tokens_per_sec",
        y_label="Tokens per second",
        loglog=True,
    )
    _line_plot(
        rows,
        output_dir / "fig10_batch_size_speedup_lines.png",
        title="Batch Size Speedup (Lines)",
        experiment="batch_size_sensitivity",
        x_key="batch_size",
        y_key="speedup_vs_baseline",
        y_label="Speedup vs naive",
        loglog=False,
        mark_oom=False,
    )
    _line_plot(
        rows,
        output_dir / "fig11_batch_size_throughput.png",
        title="Batch Size Throughput",
        experiment="batch_size_sensitivity",
        x_key="batch_size",
        y_key="tokens_per_sec",
        y_label="Tokens per second",
        loglog=False,
        mark_oom=False,
    )
    _facet_line_plot(
        rows,
        output_dir / "fig12_runtime_vs_sequence_length_by_head_dim.png",
        title="Runtime vs Sequence Length by Head Dim",
        experiment="runtime_memory_scaling",
        x_key="seq_len",
        y_key="median_time_ms",
        y_label="Median time (ms)",
        facet_key="head_dim",
        facet_label="Head dim",
        loglog=True,
    )
    _facet_line_plot(
        rows,
        output_dir / "fig13_peak_memory_vs_sequence_length_by_head_dim.png",
        title="Peak Memory vs Sequence Length by Head Dim",
        experiment="runtime_memory_scaling",
        x_key="seq_len",
        y_key="peak_memory_mb",
        y_label="Peak memory (MB)",
        facet_key="head_dim",
        facet_label="Head dim",
        loglog=True,
        mark_oom=True,
    )
    _line_plot(
        rows,
        output_dir / "fig4_kv_cache_per_token_latency.png",
        title="KV-cache Per-token Latency",
        experiment="kv_cache",
        x_key="context_len",
        y_key="time_per_token_ms",
        y_label="Per-token latency (ms)",
        loglog=True,
    )
    _line_plot(
        rows,
        output_dir / "fig5_kv_cache_memory_tradeoff.png",
        title="KV-cache Memory Trade-off",
        experiment="kv_cache",
        x_key="context_len",
        y_key="cache_memory_mb",
        y_label="Cache memory (MB)",
        loglog=False,
    )
    _batch_heatmap(rows, output_dir / "fig6_batch_size_heatmap.png", plt)
    _memory_hierarchy_diagram(output_dir / "fig7_gpu_memory_hierarchy.png", plt)
    _profiler_plot(output_dir / "fig8_gpu_profiling_timeline.png", plt)


def _number_or_blank(value: Any) -> float | int | str:
    if value in ("", None):
        return ""
    try:
        number = float(value)
    except ValueError:
        return value
    return int(number) if number.is_integer() else number


def _ok_rows(rows: list[dict[str, Any]], experiment: str) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("experiment") == experiment
        and row.get("status") == "ok"
    ]


def _line_plot(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    title: str,
    experiment: str,
    x_key: str,
    y_key: str,
    y_label: str,
    loglog: bool,
    mark_oom: bool = True,
) -> None:
    import matplotlib.pyplot as plt

    plot_rows = _ok_rows(rows, experiment)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in plot_rows:
        if row.get(x_key) != "" and row.get(y_key) != "":
            grouped[str(row["method"])].append(row)

    if not grouped:
        return

    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    for method, method_rows in sorted(grouped.items()):
        method_rows.sort(key=lambda item: item[x_key])
        xs = [row[x_key] for row in method_rows]
        ys = [row[y_key] for row in method_rows]
        ax.plot(xs, ys, marker="o", linewidth=2, label=method)

    if mark_oom:
        _mark_status_rows(rows, ax, experiment=experiment, x_key=x_key)

    if loglog:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_label)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _sorted_unique_values(rows: list[dict[str, Any]], key: str) -> list[Any]:
    values: list[Any] = []
    for row in rows:
        value = row.get(key)
        if value in ("", None):
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        values.append(value)

    def sort_key(value: Any) -> tuple[int, Any]:
        if isinstance(value, (int, float)):
            return (0, float(value))
        return (1, str(value))

    return sorted({value for value in values}, key=sort_key)


def _facet_line_plot(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    title: str,
    experiment: str,
    x_key: str,
    y_key: str,
    y_label: str,
    facet_key: str,
    facet_label: str,
    loglog: bool,
    mark_oom: bool = True,
) -> None:
    import matplotlib.pyplot as plt

    plot_rows = _ok_rows(rows, experiment)
    facet_values = _sorted_unique_values(plot_rows, facet_key)
    if len(facet_values) < 2:
        return

    columns = min(3, len(facet_values))
    rows_count = math.ceil(len(facet_values) / columns)
    fig, axes = plt.subplots(
        rows_count,
        columns,
        figsize=(columns * 5.0, rows_count * 4.2),
        dpi=160,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for idx, facet_value in enumerate(facet_values):
        ax = axes[idx // columns][idx % columns]
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in plot_rows:
            if row.get(facet_key) != facet_value:
                continue
            if row.get(x_key) != "" and row.get(y_key) != "":
                grouped[str(row["method"])].append(row)

        if not grouped:
            ax.axis("off")
            continue

        for method, method_rows in sorted(grouped.items()):
            method_rows.sort(key=lambda item: item[x_key])
            xs = [row[x_key] for row in method_rows]
            ys = [row[y_key] for row in method_rows]
            ax.plot(xs, ys, marker="o", linewidth=2, label=method)

        if mark_oom:
            _mark_status_rows(
                rows,
                ax,
                experiment=experiment,
                x_key=x_key,
                filters={facet_key: facet_value},
            )

        if loglog:
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
        ax.set_title(f"{facet_label}: {facet_value}")
        ax.set_xlabel(x_key)
        ax.set_ylabel(y_label)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()

    for idx in range(len(facet_values), rows_count * columns):
        axes[idx // columns][idx % columns].axis("off")

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path)
    plt.close(fig)


def _mark_status_rows(
    rows: list[dict[str, Any]],
    ax: Any,
    *,
    experiment: str,
    x_key: str,
    filters: dict[str, Any] | None = None,
) -> None:
    styles = {
        "oom": ("x", "crimson", "OOM"),
        "skipped": ("|", "darkorange", "SKIP"),
    }
    labeled: set[str] = set()
    for row in rows:
        status = row.get("status")
        if row.get("experiment") != experiment or status not in styles or row.get(x_key) == "":
            continue
        if filters:
            if any(row.get(key) != value for key, value in filters.items()):
                continue
        marker, color, label = styles[str(status)]
        ax.scatter(
            row[x_key],
            _status_y_position(ax),
            marker=marker,
            color=color,
            s=90,
            label=label if label not in labeled else None,
        )
        labeled.add(label)


def _status_y_position(ax: Any) -> float:
    low, high = ax.get_ylim()
    if math.isfinite(high) and high > 0:
        return high
    return max(low, 1.0)


def _batch_heatmap(rows: list[dict[str, Any]], output_path: Path, plt: Any) -> None:
    heat_rows = [
        row
        for row in _ok_rows(rows, "batch_size_sensitivity")
        if row.get("speedup_vs_baseline") != ""
    ]
    if not heat_rows:
        return

    methods = sorted({str(row["method"]) for row in heat_rows})
    batch_sizes = sorted({int(row["batch_size"]) for row in heat_rows})
    matrix = []
    for method in methods:
        matrix.append(
            [
                _find_value(heat_rows, method=method, batch_size=batch_size)
                for batch_size in batch_sizes
            ]
        )

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=160)
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_title("Batch Size Speedup Heatmap")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Method")
    ax.set_xticks(range(len(batch_sizes)), batch_sizes)
    ax.set_yticks(range(len(methods)), methods)
    fig.colorbar(image, ax=ax, label="Speedup vs naive")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _find_value(rows: list[dict[str, Any]], *, method: str, batch_size: int) -> float:
    for row in rows:
        if row["method"] == method and int(row["batch_size"]) == batch_size:
            return float(row["speedup_vs_baseline"])
    return float("nan")


def _memory_hierarchy_diagram(output_path: Path, plt: Any) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    ax.axis("off")
    layers = [
        ("Registers", "fastest / smallest"),
        ("Shared memory / SRAM", "FlashAttention tiles Q/K/V here"),
        ("L2 cache", "reuse across thread blocks"),
        ("HBM / DRAM", "naive attention materializes NxN scores here"),
    ]
    colors = ["#234f68", "#2f7f6f", "#c08a2c", "#9b3f3f"]
    for idx, (name, note) in enumerate(layers):
        y = 0.82 - idx * 0.2
        width = 0.35 + idx * 0.14
        x = 0.5 - width / 2
        rect = plt.Rectangle((x, y), width, 0.12, color=colors[idx], alpha=0.9)
        ax.add_patch(rect)
        ax.text(0.5, y + 0.075, name, ha="center", va="center", color="white", weight="bold")
        ax.text(0.5, y + 0.025, note, ha="center", va="center", color="white", fontsize=8)
    ax.text(0.5, 0.06, "Lower levels are larger and slower; IO-aware kernels reduce HBM traffic.", ha="center")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _profiler_plot(output_path: Path, plt: Any) -> None:
    summary_path = Path("results/profiler_summary.csv")
    if not summary_path.exists():
        return
    rows = read_rows(summary_path)
    rows = [row for row in rows if row.get("self_time_ms") != ""]
    if not rows:
        return

    labels = [str(row["method"]) for row in rows]
    values = [float(row["self_time_ms"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    ax.bar(labels, values, color="#326b91")
    ax.set_title("Profiler Kernel Time Summary")
    ax.set_ylabel("Self time (ms)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
