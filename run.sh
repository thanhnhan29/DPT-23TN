#!/usr/bin/env bash
set -euo pipefail

MAX_SIZE=30000
BASELINE_MAX_SIZE=8192
WARMUP_RUNS=5
MEASURE_RUNS=20
OUTPUT="results/benchmark_results.csv"
FIGURES_DIR="figures"
PROFILE_OUTPUT_DIR="results"
PYTHON_BIN="${PYTHON:-python}"

usage() {
  cat <<'EOF'
Usage: ./run.sh [options]

Options:
  --max-size N            Max sequence/context length for stress runs. Default: 30000
  --baseline-max-size N   Skip quadratic baselines above N. Default: 8192
  --warmup-runs N         Warmup runs per benchmark. Default: 5
  --measure-runs N        Measured runs per benchmark. Default: 20
  --output PATH           Benchmark CSV path. Default: results/benchmark_results.csv
  --figures-dir PATH      Figures output directory. Default: figures
  --profile-output-dir P  Profiler output directory. Default: results
  -h, --help              Show this help message.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-size)
      MAX_SIZE="$2"
      shift 2
      ;;
    --baseline-max-size)
      BASELINE_MAX_SIZE="$2"
      shift 2
      ;;
    --warmup-runs)
      WARMUP_RUNS="$2"
      shift 2
      ;;
    --measure-runs)
      MEASURE_RUNS="$2"
      shift 2
      ;;
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    --figures-dir)
      FIGURES_DIR="$2"
      shift 2
      ;;
    --profile-output-dir)
      PROFILE_OUTPUT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if (( MAX_SIZE < 1 )); then
  echo "--max-size must be a positive integer" >&2
  exit 2
fi

FIXED_SEQ_LEN=$(( MAX_SIZE < 4096 ? MAX_SIZE : 4096 ))
PROFILE_SEQ_LEN=$FIXED_SEQ_LEN

echo "Running stress benchmark up to ${MAX_SIZE} tokens"
echo "Skipping quadratic baselines above ${BASELINE_MAX_SIZE} tokens"

"${PYTHON_BIN}" main.py \
  --preset cv \
  --max-size "${MAX_SIZE}" \
  --baseline-max-size "${BASELINE_MAX_SIZE}" \
  --fixed-seq-len "${FIXED_SEQ_LEN}" \
  --warmup-runs "${WARMUP_RUNS}" \
  --measure-runs "${MEASURE_RUNS}" \
  --output "${OUTPUT}"

"${PYTHON_BIN}" profile_attention.py \
  --seq-len "${PROFILE_SEQ_LEN}" \
  --methods naive sdpa flash_sdpa \
  --output-dir "${PROFILE_OUTPUT_DIR}"

"${PYTHON_BIN}" plot_results.py \
  --input "${OUTPUT}" \
  --output-dir "${FIGURES_DIR}"

echo "Done."
echo "Benchmark CSV: ${OUTPUT}"
echo "Profiler output: ${PROFILE_OUTPUT_DIR}"
echo "Figures: ${FIGURES_DIR}"
