# vLLM Serving-Topology Benchmark — Methodology

**Question:** For serving one model on 4× RTX 3090, what maximizes throughput under concurrency — replicating the model per GPU (data parallel), or splitting it with tensor parallelism? Specifically we compare **4× TP=1**, **2× TP=2**, and **1× TP=4** for `Qwen3.6-35B-A3B-AWQ`.

## Hardware
- 4× NVIDIA GeForce RTX 3090 (24 GB, Ampere sm_86), **no NVLink** — inter-GPU traffic crosses **PCIe**.
- AMD Ryzen Threadripper 2920X (12C/24T), 128 GB RAM.
- Driver 590.48.01, CUDA 13.

## Software
- **vLLM 0.21.0** via the official `vllm/vllm-openai:v0.21.0` container (entrypoint `["vllm","serve"]`).
- One config launched at a time; all 4 GPUs dedicated to it; benchmarked; torn down; next.

## Model
- **QuantTrio Qwen3.6-35B-A3B-AWQ** — 35B-parameter Mixture-of-Experts (~3B active/token), **AWQ 4-bit** weights.
- **Measured weight footprint ≈ 23 GB on a single card** (see TP=1 result) → the model does **not** fit on one 24 GB 3090 with any room for KV cache.

## The three topologies
| Mode | Layout | GPUs/instance | Instances | Feasible? |
|---|---|---|---|---|
| **4× TP=1** | one full copy per GPU (data parallel) | 1 | 4 | **NO — OOM** |
| **2× TP=2** | two replicas, each split across 2 GPUs | 2 | 2 | yes |
| **1× TP=4** | one instance split across all 4 GPUs | 4 | 1 | yes |

**4× TP=1 is infeasible on this hardware:** launching a single TP=1 instance OOMs at model init — `GPU 0 … 283 MiB free … OutOfMemoryError` — because the ~23 GB of AWQ weights nearly fill the 24 GB card, leaving no room for activations/KV cache. This is a hard result, not a tuning failure (lowering `max-model-len` doesn't help; the *weights* are the constraint). **TP ≥ 2 is mandatory for this model on 3090s.**

## Controlled variables (held identical across the feasible modes)
Only the tensor-parallel size / replica count changes. Everything else is fixed:
- `--dtype auto`, `--gpu-memory-utilization 0.92`
- `--max-model-len 8192` (same context budget for all; well above the 768-token workload)
- `--max-num-seqs 256`, `--max-num-batched-tokens 16384`
- `--enable-expert-parallel` for TP>1 (intrinsic to multi-GPU MoE; N/A for a single GPU)
- prefix caching **off** + **unique prompt per request** → measures raw throughput, not cache hits

Measured KV-cache budget (the quantity topology actually changes):
- **1× TP=4:** GPU KV cache = **962,344 tokens** (117× concurrency @ 8k ctx) — weights split 4 ways → huge KV.
- **2× TP=2:** _(captured at run time in `results/tp2x2.kv`)_ — weights split 2 ways → ~half the KV per instance, ×2 instances.

## Workload
- **Input 512 tokens, output 256 tokens**, `temperature=0`, `ignore_eos=true` (every request emits exactly 256 tokens → clean, comparable token accounting).
- Prompts are unique per request (seeded filler) to defeat prefix-cache/identical-batch effects.

## Load generation (`bench.py`, stdlib-only, streaming)
- Holds **N concurrent requests in flight for 25 s** (after 6 s warmup) at each concurrency level.
- For multi-endpoint modes (2× TP=2), requests are **round-robined across the replicas** — i.e. a proper load-balanced data-parallel aggregate.
- Concurrency sweep: **N ∈ {1, 2, 4, 8, 16, 32, 64, 128}**.
- Streaming SSE captures **TTFT**; final usage gives exact output-token counts.

## Metrics
- **`out_toks_per_s`** — aggregate decode throughput = Σ(output tokens) / measurement window. **Headline.**
- `req_per_s`, **TTFT** (p50/p99), **e2e latency** (p50/p99), **TPOT** (per-output-token time, p50).

## Threats to validity / caveats
- **Single run per point**, one machine — no variance bars; treat small differences as noise, large/consistent trends as real.
- **KV budget differs by topology by design** — that *is* the mechanism under test, not a confound.
- **Benchmark `max-model-len` (8192) < production (131072)** — chosen so all modes are comparable (and so TP=1 could even be attempted). Longer contexts would further favor whichever topology has more KV headroom.
- **No-NVLink PCIe** interconnect penalizes tensor parallelism more than an NVLink system would; results are specific to this class of hardware.
- **First-shape Triton JIT** causes a one-time TTFT spike (visible in p99); warmup mitigates but does not fully eliminate it.
- MoE **expert-parallel** is enabled only where TP>1 (it cannot exist on a single GPU); this is inherent to the topology, not an independent knob.
