# Attention Benchmark Base

Code nền dùng chung để đo execution time và peak memory cho các biến thể attention.

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy benchmark

```bash
python main.py
```

Chạy nhanh để smoke test:

```bash
python main.py --seq-lengths 128 256 --warmup-runs 2 --measure-runs 5
```

Chỉ chạy full self-attention:

```bash
python main.py --scenarios full_attention --methods naive sdpa
```

Chạy full self-attention và ép Flash Attention backend:

```bash
python main.py --scenarios full_attention --methods naive sdpa flash_sdpa
```

Chỉ chạy KV-cache inference decode:

```bash
python main.py --scenarios kv_decode --decode-methods decode_no_cache decode_kv_cache
```

Kết quả được ghi vào:

```text
results/benchmark_results.csv
```

## Methods hiện có

### Full attention

- `naive`: standard attention, materialize ma trận attention `N x N`.
- `sdpa`: PyTorch `scaled_dot_product_attention`; trên CUDA có thể dispatch sang memory-efficient hoặc Flash kernels tùy phần cứng/PyTorch.
- `flash_sdpa`: ép PyTorch dùng Flash Attention backend; nếu không có CUDA/backend phù hợp thì row sẽ báo `error`.

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

Ý nghĩa `seq_len`:

- `full_attention`: độ dài sequence đầu vào.
- `kv_decode`: độ dài context trước khi sinh token mới.

## Ghi chú đo memory

- Trên CUDA, code dùng `torch.cuda.max_memory_allocated()`, phù hợp để báo cáo peak memory.
- Trên CPU, code dùng `tracemalloc`, chủ yếu phục vụ smoke test; số này không phản ánh đầy đủ native memory mà PyTorch cấp phát.
