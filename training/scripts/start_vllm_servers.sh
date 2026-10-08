#!/bin/bash
# 一键启动8个vLLM实例，分别跑在8张GPU上
set -e

MODEL_PATH="/path/to/spider_checkpoints/spider_20260421_0922/global_step_384/actor/huggingface"
BASE_PORT=9000
NUM_GPUS=8

echo "=== Starting ${NUM_GPUS} vLLM instances ==="
echo "Model: ${MODEL_PATH}"
echo "Ports: ${BASE_PORT}-$((BASE_PORT + NUM_GPUS - 1))"
echo ""

# 先清理已有的vLLM进程
echo "Killing existing vLLM processes..."
pkill -f "vllm serve ${MODEL_PATH}" 2>/dev/null || true
sleep 2

for i in $(seq 0 $((NUM_GPUS - 1))); do
    PORT=$((BASE_PORT + i))
    echo "[GPU ${i}] Starting vLLM on port ${PORT}..."
    
    CUDA_VISIBLE_DEVICES=${i} vllm serve "${MODEL_PATH}" \
        --host 0.0.0.0 \
        --port ${PORT} \
        --dtype auto \
        --served-model-name qwen2.5-coder \
        --max-num-seqs 8 \
        --gpu-memory-utilization 0.85 \
        --max-model-len 16384 \
        --enforce-eager \
        > /tmp/vllm_gpu${i}.log 2>&1 &
    
    echo "  PID: $!"
    sleep 2  # 错开启动时间，避免同时竞争
done

echo ""
echo "=== All ${NUM_GPUS} instances started ==="
echo "Logs: /tmp/vllm_gpu{0..7}.log"
echo ""
echo "Waiting for health checks..."

# 健康检查
for i in $(seq 0 $((NUM_GPUS - 1))); do
    PORT=$((BASE_PORT + i))
    for attempt in $(seq 1 60); do
        if curl -sf http://localhost:${PORT}/health > /dev/null 2>&1; then
            echo "  [GPU ${i}] Port ${PORT} OK"
            break
        fi
        if [ $attempt -eq 60 ]; then
            echo "  [GPU ${i}] Port ${PORT} FAILED (check /tmp/vllm_gpu${i}.log)"
        fi
        sleep 2
    done
done

echo ""
echo "Done. Test with: curl http://localhost:9000/v1/models"
