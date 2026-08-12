# vLLM Serving Topology on 4× RTX 3090: Replicas vs Tensor Parallelism

**Question:** to serve one model on 4 GPUs for maximum concurrent throughput, is it better to
**replicate** the model (a copy per card) or **split** it with tensor parallelism? And how many ways?

**Model under test:** `Qwen3.6-35B-A3B-AWQ` (35B-param MoE, ~3B active, AWQ 4-bit) · **4× RTX 3090 (24 GB, no NVLink, PCIe)** · vLLM 0.21.0.

## TL;DR

1. **You cannot put one copy per card.** `4× TP=1` **OOMs** — the AWQ weights are **~23 GB**, nearly the whole 24 GB card, leaving no room for KV cache. **TP ≥ 2 is mandatory** for this model on 3090s.
2. **Among the feasible layouts, `2× TP=2` beats `1× TP=4` at every concurrency, and the gap grows with load** — from +4% at 1 request to **+136% at 128** (1,437 vs 610 tok/s). The 4-way all-reduce over **PCIe (no NVLink)** is the tax.
3. **`2× TP=2` also has ~2× lower latency** (TTFT 1.4 s vs 3.9 s, e2e 5.0 s vs 9.3 s at 16 concurrent). It wins on throughput *and* latency.

**Bottom line:** on no-NVLink 3090s, **use the fewest tensor-parallel GPUs that make the model fit, and replicate the rest.** Here that's exactly `2× TP=2` — the current production layout is optimal.

### Aggregate decode throughput (output tok/s)
| Concurrency | 1× TP=4 | 2× TP=2 | 2×TP2 advantage |
|---:|---:|---:|:--|
| 1 | 111 | 116 | +4% |
| 8 | 375 | 583 | +56% |
| 32 | 527 | 1026 | +95% |
| 64 | 580 | 1210 | +109% |
| 128 | 610 | **1437** | **+136%** |

Full numbers, latency, and KV-cache measurements in **[RESULTS.md](RESULTS.md)**; full method/caveats in **[METHODOLOGY.md](METHODOLOGY.md)**.

## Why
Throughput here is **not** KV-limited — both feasible modes have far more KV cache (44–117× the workload) than the ≤128 concurrency we test. The difference is **interconnect**: tensor parallelism does an all-reduce **every layer**, and with no NVLink that crosses PCIe. `TP=4` pays it across 4 GPUs and **saturates ~600 tok/s**; two independent `TP=2` replicas each pay a smaller 2-GPU penalty and run in parallel, scaling past **1,400 tok/s**.

## Layout
```
METHODOLOGY.md          # hardware, software, configs, workload, metrics, threats-to-validity
RESULTS.md              # generated comparison tables (throughput / latency / KV)
bench.py                # stdlib streaming load generator (round-robins across replicas)
run_bench.sh            # orchestrator: launch each topology → sweep → teardown → restore
analyze.py              # results/*.jsonl -> RESULTS.md
test_single_card_fit.sh # reproduces the 4×TP=1 OOM (model too big for one card)
results/                # raw per-concurrency JSON + measured KV-cache sizes
```

## Reproduce
```bash
# needs: docker w/ NVIDIA runtime, the model at /data/models, python3 (stdlib only)
sg docker -c "bash run_bench.sh"     # ~20 min; benchmarks 1×TP4 and 2×TP2, restores your pair
python3 analyze.py                    # regenerate RESULTS.md
```
Controlled constants (identical across topologies): `max-model-len 8192`, `gpu-mem-util 0.92`,
`max-num-seqs 256`, `max-num-batched-tokens 16384`, AWQ, expert-parallel where TP>1, prefix-cache
off, unique prompt per request. Workload: 512-in / 256-out, `ignore_eos`, temp 0, concurrency sweep 1→128.
