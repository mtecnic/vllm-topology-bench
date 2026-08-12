#!/usr/bin/env python3
"""Read results/*.jsonl and emit RESULTS.md with a topology comparison."""
import json, os, glob
BASE=os.path.dirname(os.path.abspath(__file__))

def load(mode):
    f=os.path.join(BASE,"results",f"{mode}.jsonl")
    if not os.path.exists(f): return {}
    out={}
    for line in open(f):
        line=line.strip()
        if not line: continue
        try: d=json.loads(line)
        except Exception: continue
        out[d["concurrency"]]=d
    return out

def kv(mode):
    f=os.path.join(BASE,"results",f"{mode}.kv")
    return open(f).read().strip() if os.path.exists(f) else "(n/a)"

tp4=load("tp4"); tp2=load("tp2x2")
concs=sorted(set(tp4)|set(tp2))
L=[]
L.append("# Results — vLLM Serving Topology on 4× RTX 3090\n")
L.append("Model: **Qwen3.6-35B-A3B-AWQ** · Workload: 512-in / 256-out · vLLM 0.21.0 · see METHODOLOGY.md\n")
L.append("## Feasibility\n")
L.append("- **4× TP=1 (one copy per card): INFEASIBLE** — OOM at init (~23 GB weights vs 24 GB card; 283 MiB free). TP≥2 mandatory.\n")
L.append("- **1× TP=4** and **2× TP=2**: both feasible; compared below.\n")
L.append("\n## KV cache (measured)\n```")
L.append("TP=4  : "+kv("tp4")); L.append("TP=2×2: "+kv("tp2x2")); L.append("```\n")

L.append("## Aggregate decode throughput (output tokens/sec) — higher is better\n")
L.append("| Concurrency | 1× TP=4 | 2× TP=2 | Winner | Δ |")
L.append("|---:|---:|---:|:--|--:|")
for c in concs:
    a=tp4.get(c,{}).get("out_toks_per_s"); b=tp2.get(c,{}).get("out_toks_per_s")
    if a is None or b is None:
        L.append(f"| {c} | {a if a is not None else '—'} | {b if b is not None else '—'} | — | — |"); continue
    win="**TP=2×2**" if b>a else ("**TP=4**" if a>b else "tie")
    hi,lo=max(a,b),min(a,b); delta=f"{(hi/lo-1)*100:.0f}%" if lo>0 else "—"
    L.append(f"| {c} | {a:.0f} | {b:.0f} | {win} | {delta} |")

def peak(d):
    if not d: return (None,None)
    c=max(d, key=lambda k: d[k].get("out_toks_per_s",0)); return (c,d[c]["out_toks_per_s"])
p4=peak(tp4); p2=peak(tp2)
L.append("\n## Latency (p50) at a mid load\n")
L.append("| | TTFT ms | e2e ms | TPOT ms |")
L.append("|:--|--:|--:|--:|")
for mode,d in [("1× TP=4",tp4),("2× TP=2",tp2)]:
    # pick concurrency=16 if present else the median available
    ks=sorted(d);
    if not ks: continue
    c=16 if 16 in d else ks[len(ks)//2]; r=d[c]
    L.append(f"| {mode} (c={c}) | {r.get('ttft_ms_p50')} | {r.get('e2e_ms_p50')} | {r.get('tpot_ms_p50')} |")

L.append("\n## Peak throughput\n")
L.append(f"- **1× TP=4:** {p4[1]} tok/s (at concurrency {p4[0]})")
L.append(f"- **2× TP=2:** {p2[1]} tok/s (at concurrency {p2[0]})")
if p4[1] and p2[1]:
    if p2[1]>p4[1]: L.append(f"- **Peak winner: 2× TP=2** by {(p2[1]/p4[1]-1)*100:.0f}%.")
    elif p4[1]>p2[1]: L.append(f"- **Peak winner: 1× TP=4** by {(p4[1]/p2[1]-1)*100:.0f}%.")
open(os.path.join(BASE,"RESULTS.md"),"w").write("\n".join(L)+"\n")
print("\n".join(L))
