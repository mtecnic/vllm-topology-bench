#!/usr/bin/env python3
"""Streaming load generator for vLLM serving-topology benchmark.

Holds `concurrency` requests in flight for `duration` seconds against one or more
OpenAI-compatible endpoints (round-robin across them = load-balanced data parallel),
using a fixed synthetic workload. Measures steady-state aggregate throughput and
latency. Streaming SSE is used to capture TTFT (time-to-first-token). Stdlib only.

Metrics reported (JSON):
  out_toks_per_s : aggregate DECODE throughput = sum(output tokens)/measurement window  (headline)
  req_per_s      : completed requests / window
  ttft_ms        : time to first token  (p50/p99)
  e2e_ms         : end-to-end latency    (p50/p99)
  tpot_ms        : per-output-token time = (e2e - ttft)/output_tokens (p50)
"""
import argparse, json, time, threading, urllib.request, statistics as st, itertools, random

def build_prompt(uid, approx_tokens):
    # unique-per-request so prefix caching / identical-batch effects don't skew results;
    # ~approx_tokens words (~1 token/word) of filler.
    words=["alpha","bravo","delta","echo","gamma","harbor","ionize","jungle","kelvin","lumen",
           "matrix","nectar","optics","photon","quartz","raster","sigma","tensor","umbra","vertex"]
    r=random.Random(uid)
    body=" ".join(r.choice(words) for _ in range(max(1,approx_tokens-8)))
    return f"[req {uid}] Continue this technical log verbatim style. {body} =>"

def one_request(ep, model, prompt, out_len, timeout):
    payload=json.dumps({"model":model,"prompt":prompt,"max_tokens":out_len,"temperature":0.0,
                        "ignore_eos":True,"stream":True,
                        "stream_options":{"include_usage":True}}).encode()
    req=urllib.request.Request(ep+"/v1/completions",data=payload,
                               headers={"Content-Type":"application/json"})
    t0=time.time(); ttft=None; comp_toks=0
    with urllib.request.urlopen(req,timeout=timeout) as r:
        for raw in r:
            line=raw.decode("utf-8","replace").strip()
            if not line.startswith("data:"): continue
            data=line[5:].strip()
            if data=="[DONE]": break
            try: obj=json.loads(data)
            except Exception: continue
            ch=obj.get("choices") or []
            if ch and ch[0].get("text"):
                if ttft is None: ttft=time.time()-t0
            u=obj.get("usage")
            if u and u.get("completion_tokens"): comp_toks=u["completion_tokens"]
    e2e=time.time()-t0
    return {"ttft":ttft if ttft is not None else e2e,"e2e":e2e,"out_toks":comp_toks,"start":t0,"end":t0+e2e}

def run(endpoints, model, concurrency, duration, in_len, out_len, timeout):
    stop_at=[0.0]; recs=[]; lock=threading.Lock(); ctr=itertools.count()
    def worker(wid):
        ep=endpoints[wid % len(endpoints)]
        while time.time()<stop_at[0]:
            uid=next(ctr)
            try:
                rec=one_request(ep,model,build_prompt(uid,in_len),out_len,timeout)
                with lock: recs.append(rec)
            except Exception as e:
                with lock: recs.append({"error":str(e)[:120]})
    stop_at[0]=time.time()+duration
    ths=[threading.Thread(target=worker,args=(i,),daemon=True) for i in range(concurrency)]
    t_start=time.time()
    for t in ths: t.start()
    for t in ths: t.join()
    ok=[r for r in recs if "error" not in r]
    errs=[r for r in recs if "error" in r]
    if not ok:
        return {"concurrency":concurrency,"completed":0,"errors":len(errs),"note":"all failed",
                "sample_error":(errs[0]["error"] if errs else "")}
    window=max(r["end"] for r in ok)-min(r["start"] for r in ok)
    tot_out=sum(r["out_toks"] for r in ok)
    pct=lambda xs,p: round(st.quantiles(xs,n=100)[p-1]*1000,1) if len(xs)>1 else round(xs[0]*1000,1)
    ttfts=[r["ttft"] for r in ok]; e2es=[r["e2e"] for r in ok]
    tpots=[(r["e2e"]-r["ttft"])/r["out_toks"] for r in ok if r["out_toks"]>1]
    return {"concurrency":concurrency,"completed":len(ok),"errors":len(errs),
            "window_s":round(window,2),"total_out_tokens":tot_out,
            "out_toks_per_s":round(tot_out/window,1),"req_per_s":round(len(ok)/window,2),
            "ttft_ms_p50":pct(ttfts,50),"ttft_ms_p99":pct(ttfts,99),
            "e2e_ms_p50":pct(e2es,50),"e2e_ms_p99":pct(e2es,99),
            "tpot_ms_p50":(round(st.median(tpots)*1000,2) if tpots else None),
            "mean_out_toks":round(tot_out/len(ok),1)}

if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--endpoints",required=True,help="comma-separated base URLs")
    a.add_argument("--model",required=True)
    a.add_argument("--concurrency",type=int,required=True)
    a.add_argument("--duration",type=float,default=25)
    a.add_argument("--in-len",type=int,default=512)
    a.add_argument("--out-len",type=int,default=256)
    a.add_argument("--timeout",type=float,default=600)
    a.add_argument("--warmup",type=float,default=6)
    args=a.parse_args()
    eps=[e.strip().rstrip("/") for e in args.endpoints.split(",") if e.strip()]
    if args.warmup>0: run(eps,args.model,min(4,args.concurrency),args.warmup,args.in_len,args.out_len,args.timeout)
    res=run(eps,args.model,args.concurrency,args.duration,args.in_len,args.out_len,args.timeout)
    res["endpoints"]=len(eps); res["in_len"]=args.in_len; res["out_len"]=args.out_len
    print(json.dumps(res))
