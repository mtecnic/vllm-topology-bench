#!/bin/bash
# Orchestrate the serving-topology benchmark. Run via:  sg docker -c "bash run_bench.sh"
# Modes: 1x TP=4 (all 4 GPUs), 2x TP=2 (data-parallel pair). (4x TP=1 is infeasible: OOM.)
set -u
BASE=/data/projects/vllm-topology-bench
IMG=vllm/vllm-openai:v0.21.0
MODEL=/app/models/QuantTrio-Qwen3.6-35B-A3B-AWQ
MNT="-v /data/models:/app/models -v /home/waive5/.cache/huggingface:/root/.cache/huggingface"
# CONTROLLED constants held identical across topologies (only TP/replica count varies):
COMMON="--model $MODEL --dtype auto --max-model-len 8192 --gpu-memory-utilization 0.92 --max-num-seqs 256 --max-num-batched-tokens 16384 --trust-remote-code"
CONC="1 2 4 8 16 32 64 128"
DUR=25
IN=512; OUT=256
log(){ echo "[$(date +%H:%M:%S)] $*"; }
teardown(){ docker rm -f $(docker ps -aq --filter name=bench-) >/dev/null 2>&1; sleep 2; }
wait_ready(){ for i in $(seq 1 130); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:$1/health 2>/dev/null)" = 200 ] && return 0
    docker ps --format '{{.Names}}' | grep -q bench- || return 1; sleep 3; done; return 1; }
kvinfo(){ docker logs "$1" 2>&1 | grep -iE "GPU KV cache size|Maximum concurrency|GPU blocks" | tail -3; }
sweep(){ local mode=$1 eps=$2; : > "$BASE/results/$mode.jsonl"
  for c in $CONC; do
    log "  [$mode] concurrency=$c ..."
    python3 "$BASE/bench.py" --endpoints "$eps" --model "$MODEL" --concurrency "$c" \
      --duration $DUR --in-len $IN --out-len $OUT >> "$BASE/results/$mode.jsonl" 2>>"$BASE/logs/bench_err.log"
  done; }

log "=== benchmark start ==="
teardown
docker stop vllm-pair-8000 vllm-pair-8001 >/dev/null 2>&1

# ---------- MODE C: 1x TP=4 ----------
log "MODE tp4: launching 1 container on GPUs 0,1,2,3"
docker run -d --name bench-tp4-9000 --gpus '"device=0,1,2,3"' $MNT -p 9000:9000 $IMG \
  $COMMON --tensor-parallel-size 4 --enable-expert-parallel --port 9000 >/dev/null 2>&1
if wait_ready 9000; then
  log "tp4 READY"; kvinfo bench-tp4-9000 | tee "$BASE/results/tp4.kv"
  sweep tp4 "http://localhost:9000"
else log "tp4 FAILED TO START"; docker logs bench-tp4-9000 2>&1 | tail -12 > "$BASE/logs/tp4_fail.log"; fi
teardown

# ---------- MODE A: 2x TP=2 ----------
log "MODE tp2x2: launching 2 containers (GPUs 0,1 and 2,3)"
docker run -d --name bench-tp2-9000 --gpus '"device=0,1"' $MNT -p 9000:9000 $IMG \
  $COMMON --tensor-parallel-size 2 --enable-expert-parallel --port 9000 >/dev/null 2>&1
docker run -d --name bench-tp2-9001 --gpus '"device=2,3"' $MNT -p 9001:9001 $IMG \
  $COMMON --tensor-parallel-size 2 --enable-expert-parallel --port 9001 >/dev/null 2>&1
if wait_ready 9000 && wait_ready 9001; then
  log "tp2x2 READY"; kvinfo bench-tp2-9000 | tee "$BASE/results/tp2x2.kv"
  sweep tp2x2 "http://localhost:9000,http://localhost:9001"
else log "tp2x2 FAILED TO START"; docker logs bench-tp2-9000 2>&1 | tail -12 > "$BASE/logs/tp2_fail.log"; fi
teardown

# ---------- restore user's production pair ----------
log "restoring vllm-pair-8000 / vllm-pair-8001"
docker start vllm-pair-8000 vllm-pair-8001 >/dev/null 2>&1
log "=== benchmark DONE ==="
