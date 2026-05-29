| Configuration | preferred_batch_size | max_queue_delay_microseconds | Avg Latency (ms) | P95 (ms) | P99 (ms) | Throughput (RPS) | Success |
|---|---|---:|---:|---:|---:|---:|---:|
| No batching | - | - | 743.19 | 895.79 | 925.35 | 65.43 | 1000/1000 |
| Small batch | [2, 4] | 50000 | 387.30 | 494.82 | 538.59 | 126.94 | 1000/1000 |
| Medium batch | [4, 8, 16] | 100000 | 378.23 | 453.48 | 502.01 | 129.62 | 1000/1000 |
| Large batch | [8, 16, 32] | 200000 | 377.22 | 479.11 | 525.50 | 130.34 | 1000/1000 |
