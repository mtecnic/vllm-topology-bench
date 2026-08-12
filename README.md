<h1 align="center">🧩 Replicas &gt; Tensor Parallelism</h1>
<p align="center"><b>How should you split one model across 4× RTX 3090?</b><br>
A measured answer for <code>Qwen3.6-35B-A3B-AWQ</code> on no-NVLink consumer GPUs — with a reproducible harness.</p>

<p align="center">
<img alt="winner" src="https://img.shields.io/badge/2%C3%97TP%3D2-wins%20by%20up%20to%20136%25-brightgreen">
<img alt="hardware" src="https://img.shields.io/badge/4%C3%97RTX%203090-no%20NVLink%20(PCIe)-76b900">
<img alt="vllm" src="https://img.shields.io/badge/vLLM-0.21.0-1f6feb">
<img alt="model" src="https://img.shields.io/badge/model-Qwen3.6--35B--A3B--AWQ-8957e5">
<img alt="repro" src="https://img.shields.io/badge/harness-reproducible-brightgreen">
<img alt="stars" src="https://img.shields.io/github/stars/mtecnic/vllm-topology-bench?style=social">
</p>

---

> ### 🎣 The counterintuitive result
> Everyone reaches for **`tensor-parallel-size 4`** to "use all the GPUs." On no-NVLink 3090s that's the **slowest** feasible option. Splitting the model *less* — **two `TP=2` replicas** instead of one `TP=4` — nearly **doubles throughput and halves latency.** And the "obvious" idea of **one copy per card doesn't even fit.**

---

## 📊 The one chart

**Aggregate decode throughput (output tokens/sec) — the gap widens the busier you get.**

| Concurrency | 1× TP=4 | 2× TP=2 | |
|---:|---:|---:|:--|
| 1   | `▍` 111 | `▍` 116 | +4% |
| 8   | `██▌` 375 | `████` 583 | +56% |
| 32  | `███▌` 527 | `███████` 1026 | +95% |
| 64  | `████` 580 | `████████▌` 1210 | +109% |
| **128** | `████` **610** | `██████████████████████████` **1,437** | **🏆 +136%** |

`1× TP=4` **saturates ~600 tok/s**; `2× TP=2` keeps climbing past **1,400**. Same 4 GPUs, same model, same workload.

## 🎯 Findings

1. **🚫 "One copy per card" is impossible.** `4× TP=1` **OOMs** — the AWQ weights are **~23 GB**, nearly the whole 24 GB card, with no room for KV cache. **TP ≥ 2 is mandatory** for this model on 3090s. *(Lowering `max-model-len` doesn't help — the weights are the wall, not the KV.)*
2. **🥇 `2× TP=2` beats `1× TP=4` at every concurrency** — from +4% (1 req) to **+136% (128 reqs)**. Peak **1,437 vs 610 tok/s**.
3. **⚡ It also wins latency** — TTFT **1.4 s vs 3.9 s**, end-to-end **5.0 s vs 9.3 s** (16 concurrent). Throughput *and* responsiveness.

**➡️ The rule:** on no-NVLink GPUs, **use the fewest tensor-parallel GPUs that make the model fit, then replicate the rest.** Here that's exactly `2× TP=2`.

## 🔬 Why (it's the interconnect, not the KV cache)

Throughput here is **not** KV-limited — both feasible modes have **44–117× more KV cache** than the ≤128 concurrency tested. The lever is **interconnect**: tensor parallelism does an **all-reduce every single layer**, and with **no NVLink that crosses PCIe**. `TP=4` pays that tax across 4 GPUs → saturates. Two independent `TP=2` replicas each pay a smaller 2-GPU penalty and run in parallel → scale. *NVLink systems (A100/H100) would shift this — these results are specific to PCIe-coupled consumer cards.*

## 🧪 Reproduce it

```bash
# needs: docker + NVIDIA runtime, the model at /data/models, python3 (stdlib only)
sg docker -c "bash run_bench.sh"   # ~20 min: benchmarks 1×TP4 and 2×TP2, then restores your servers
python3 analyze.py                  # regenerate RESULTS.md
```

**Controlled** (identical across topologies — only TP/replica count varies): `max-model-len 8192`, `gpu-mem-util 0.92`, `max-num-seqs 256`, `max-num-batched-tokens 16384`, AWQ, expert-parallel where TP>1, prefix-cache **off**, **unique prompt per request**. Workload: **512-in / 256-out**, `ignore_eos`, temp 0, concurrency sweep **1→128**, round-robined across replicas (true load-balanced aggregate).

## 🗂 What's inside

| File | |
|---|---|
| **[METHODOLOGY.md](METHODOLOGY.md)** | hardware, software, configs, workload, metrics, **threats-to-validity** |
| **[RESULTS.md](RESULTS.md)** | full throughput / latency / KV-cache tables (generated) |
| `bench.py` | stdlib streaming load generator (TTFT via SSE, round-robins replicas) |
| `run_bench.sh` | orchestrator: launch topology → sweep → teardown → restore |
| `test_single_card_fit.sh` | reproduces the `4×TP=1` OOM |
| `results/*.jsonl` | raw per-concurrency data + measured KV-cache sizes |

## 💡 If you serve LLMs on consumer GPUs, take this away

- **`--tensor-parallel-size 4` is a trap without NVLink.** Prefer replicas.
- **Check if your model fits on N-1 (or fewer) GPUs** before adding TP — every extra TP GPU costs you an all-reduce/layer.
- **Throughput is usually interconnect- or KV-bound, not FLOP-bound** — measure `finish_reason`, tokens/sec, and your KV-cache size, not just "GPUs busy."

## ⚠️ Honest caveats (full list in the paper)
Single run per point (no variance bars — trust the large, consistent trends). Benchmarked at `max-model-len 8192` (< production 131072) so all topologies were comparable and `4×TP=1` could even be attempted — long contexts would *further* favor the higher-KV layout. Results are specific to **no-NVLink PCIe 3090s + this AWQ MoE**; NVLink or a smaller model changes the calculus.

---

<p align="center"><sub><b>Keywords:</b> vLLM · tensor parallelism · data parallelism · LLM serving · GPU inference throughput · RTX 3090 · KV cache · NVLink / PCIe · Mixture-of-Experts · AWQ quantization · Qwen · concurrency · tokens per second · self-hosted LLM · inference optimization</sub></p>
