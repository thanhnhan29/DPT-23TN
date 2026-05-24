# Attention Benchmark Suite

Code nền dùng chung để đo runtime, peak memory, KV-cache decode, batch-size sensitivity,
head-dimension scaling, prefill/decode split và PyTorch profiler cho các biến thể attention.

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy benchmark

Chạy toàn bộ experiment stress + profiling + figures bằng một lệnh:

```bash
chmod +x run.sh
./run.sh --max-size 80000
```

Chạy bản nhẹ để kiểm tra pipeline:

```bash
./run.sh --max-size 2048 --warmup-runs 1 --measure-runs 2
```

Trong stress run, baseline quadratic (`naive`, `decode_no_cache`, `no_cache`) cũng được thử ở mọi mốc. Nếu GPU hết bộ nhớ, CSV sẽ ghi `status=oom,error=OOM`.

```bash
python main.py
```

Mặc định là preset `smoke`: nhỏ, chạy được để kiểm tra pipeline.

Chạy nhanh để smoke test:

```bash
python main.py --seq-lengths 128 256 --warmup-runs 2 --measure-runs 5
```

Chạy bộ MUST HAVE:

```bash
python main.py --preset must-have
```

Chạy bộ CV-tier đầy đủ:

```bash
python main.py --preset cv
```

Chạy CV-tier có stress lengths tới một ngưỡng cụ thể:

```bash
python main.py --preset cv --max-size 80000
```

Chỉ chạy full self-attention:

```bash
python main.py --scenarios full_attention --methods naive sdpa
```

Chạy full self-attention với thư viện `flash-attn` thật:

```bash
python main.py --scenarios full_attention --methods naive sdpa flash_attn
```

`flash_attn` gọi trực tiếp package ngoài `flash-attn` và không fallback sang SDPA. Nếu package/CUDA/PyTorch không tương thích, row sẽ là `status=error,error=ERR`.

Chỉ chạy KV-cache inference decode:

```bash
python main.py --scenarios kv_decode --decode-methods decode_no_cache decode_kv_cache
```

Chạy batch-size sensitivity:

```bash
python main.py --scenarios batch_size --fixed-seq-len 4096 --batch-sizes 1 2 4 8 16
```

Chạy head-dimension scaling:

```bash
python main.py --scenarios head_dim --fixed-seq-len 4096 --head-dims 32 64 128 256
```

Chạy prefill/decode split:

```bash
python main.py --scenarios prefill_decode --context-lengths 1024 2048 4096 8192 16384
```

Kết quả được ghi vào:

```text
results/benchmark_results.csv
```

## Vẽ figures

Sau khi có CSV:

```bash
python plot_results.py --input results/benchmark_results.csv --output-dir figures
```

Script này sinh các hình chính:

- `fig1_runtime_vs_sequence_length.png`
- `fig2_peak_memory_vs_sequence_length.png`
- `fig3_speedup_ratio.png`
- `fig4_kv_cache_per_token_latency.png`
- `fig5_kv_cache_memory_tradeoff.png`
- `fig6_batch_size_heatmap.png`
- `fig7_gpu_memory_hierarchy.png`

Nếu đã chạy profiler và có `results/profiler_summary.csv`, script sẽ sinh thêm:

- `fig8_gpu_profiling_timeline.png`

## GPU profiling

```bash
python profile_attention.py --seq-len 4096 --methods naive sdpa flash_attn
```

Kết quả:

- `results/profiler_summary.csv`
- `results/profiler_trace_<method>_N<seq_len>.json`

File trace JSON mở được bằng Chrome tracing hoặc các tool đọc Chrome trace.

## Methods hiện có

### Full attention

- `naive`: standard attention, materialize ma trận attention `N x N`.
- `sdpa`: PyTorch `scaled_dot_product_attention`; trên CUDA có thể dispatch sang memory-efficient hoặc Flash kernels tùy phần cứng/PyTorch.
- `flash_attn`: gọi trực tiếp thư viện ngoài `flash-attn`, không fallback sang SDPA thường; đây là method FlashAttention mặc định trong preset.
- `flash_sdpa`: method optional để ép PyTorch dùng Flash SDPA backend thật; nếu không có CUDA/backend phù hợp thì row sẽ báo `error`.

### KV-cache decode

- `decode_no_cache`: mô phỏng autoregressive decoding không cache; mỗi token mới recompute full causal self-attention trên toàn prefix.
- `decode_kv_cache`: build cache K/V cho context, rồi mỗi token mới chỉ tính Q/K/V của token mới và attention vào cache.

Trong scenario `kv_decode`, thời gian đo là **decode phase**. Với `decode_kv_cache`, cache của context được chuẩn bị ngoài vùng đo thời gian decode, nhưng `cache_memory_mb` vẫn ghi dung lượng K/V cache để thể hiện trade-off memory-bound.

## CSV schema

```text
scenario,method,device,dtype,batch_size,num_heads,seq_len,context_len,decode_tokens,
head_dim,model_dim,causal,warmup_runs,measure_runs,mean_time_ms,median_time_ms,
min_time_ms,max_time_ms,time_per_token_ms,tokens_per_sec,peak_memory_mb,
cache_memory_mb,status,error
```

Schema hiện tại có thêm các cột phục vụ phân nhóm và plot:

```text
experiment,phase,sweep_value,baseline_method,speedup_vs_baseline
```

Các giá trị lỗi được chuẩn hóa để CSV gọn:

- `status=oom,error=OOM`
- `status=error,error=ERR`

Ý nghĩa `seq_len`:

- `full_attention`: độ dài sequence đầu vào.
- `kv_decode`: độ dài context trước khi sinh token mới.

## Ghi chú đo memory

- Trên CUDA, code dùng `torch.cuda.max_memory_allocated()`, phù hợp để báo cáo peak memory.
- Trên CPU, code dùng `tracemalloc`, chủ yếu phục vụ smoke test; số này không phản ánh đầy đủ native memory mà PyTorch cấp phát.
