# Results — vLLM Serving Topology on 4× RTX 3090

Model: **Qwen3.6-35B-A3B-AWQ** · Workload: 512-in / 256-out · vLLM 0.21.0 · see METHODOLOGY.md

## Feasibility

- **4× TP=1 (one copy per card): INFEASIBLE** — OOM at init (~23 GB weights vs 24 GB card; 283 MiB free). TP≥2 mandatory.

- **1× TP=4** and **2× TP=2**: both feasible; compared below.


## KV cache (measured)
```
TP=4  : 962,344 tokens  (117x concurrency @ 8k ctx)  — weights split 4 ways -> largest KV
TP=2×2: per-instance 360,448 and 417,047 tokens (44x / 51x @ 8k) x2 instances  — weights split 2 ways
```

## Aggregate decode throughput (output tokens/sec) — higher is better

| Concurrency | 1× TP=4 | 2× TP=2 | Winner | Δ |
|---:|---:|---:|:--|--:|
| 1 | 111 | 116 | **TP=2×2** | 4% |
| 2 | 196 | 235 | **TP=2×2** | 20% |
| 4 | 292 | 383 | **TP=2×2** | 31% |
| 8 | 375 | 583 | **TP=2×2** | 56% |
| 16 | 436 | 821 | **TP=2×2** | 88% |
| 32 | 527 | 1026 | **TP=2×2** | 95% |
| 64 | 580 | 1210 | **TP=2×2** | 109% |
| 128 | 610 | 1437 | **TP=2×2** | 136% |

## Latency (p50) at a mid load

| | TTFT ms | e2e ms | TPOT ms |
|:--|--:|--:|--:|
| 1× TP=4 (c=16) | 3919.6 | 9313.1 | 21.14 |
| 2× TP=2 (c=16) | 1420.8 | 4956.5 | 13.81 |

## Peak throughput

- **1× TP=4:** 609.7 tok/s (at concurrency 128)
- **2× TP=2:** 1436.7 tok/s (at concurrency 128)
- **Peak winner: 2× TP=2** by 136%.
