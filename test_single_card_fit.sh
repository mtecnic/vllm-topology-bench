set -u
docker rm -f bench-smoke >/dev/null 2>&1
docker run -d --name bench-smoke --gpus '"device=0"' \
  -v /data/models:/app/models -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 9000:9000 vllm/vllm-openai:v0.21.0 \
  --model /app/models/QuantTrio-Qwen3.6-35B-A3B-AWQ --dtype auto \
  --max-model-len 8192 --gpu-memory-utilization 0.92 --max-num-seqs 256 \
  --max-num-batched-tokens 16384 --trust-remote-code --port 9000 >/dev/null 2>&1
echo "loading (up to ~5min)..."; ready=0
for i in $(seq 1 100); do
  code=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:9000/health 2>/dev/null)
  [ "$code" = "200" ] && { echo "READY ~$((i*3))s"; ready=1; break; }
  docker ps --format '{{.Names}}' | grep -q bench-smoke || { echo "DIED"; break; }
  sleep 3
done
echo "=== KV cache / concurrency ==="; docker logs bench-smoke 2>&1 | grep -iE "KV cache|GPU blocks|Maximum concurrency|Free memory|memory profiling" | tail -6
[ "$ready" = 1 ] && { echo "=== test generation ==="; curl -s http://localhost:9000/v1/completions -H 'Content-Type: application/json' -d '{"model":"/app/models/QuantTrio-Qwen3.6-35B-A3B-AWQ","prompt":"Hello","max_tokens":5}' | head -c 200; echo; } || { echo "=== err ==="; docker logs bench-smoke 2>&1 | grep -iE "error|oom|out of memory|cuda|no available memory|fail" | tail -10; }
docker rm -f bench-smoke >/dev/null 2>&1
