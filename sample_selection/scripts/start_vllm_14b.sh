#!/bin/bash
# 一键启动 8 个 vLLM 实例(每张卡 1 个), 部署 14B 模型, 端口 9000-9007。
# 供 BIRD train 方差采样使用: 16 条 rollout 均匀分发到这 8 个端点。
set -e

# 14B 模型路径: 默认使用 prepare_local_data.sh 复制到本地的目录(读盘更快)
# 如需直接用 ModelScope 原始目录, 传入 MODEL_PATH=/path/to/models/Qwen/Qwen2___5-Coder-14B-Instruct
MODEL_PATH="${MODEL_PATH:-/root/local_models/Qwen2.5-Coder-14B-Instruct}"
BASE_PORT="${BASE_PORT:-9000}"
NUM_GPUS="${NUM_GPUS:-8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen2.5-coder}"
# 使用带完整依赖(vllm/langgraph/agentlightning)的虚拟环境
VENV_BIN="${VENV_BIN:-/path/to/venv/bin}"

export PATH="${VENV_BIN}:${PATH}"

echo "=== 启动 ${NUM_GPUS} 个 vLLM 实例 ==="
echo "模型: ${MODEL_PATH}"
echo "端口: ${BASE_PORT}-$((BASE_PORT + NUM_GPUS - 1))"
echo "served-model-name: ${SERVED_MODEL_NAME}"
echo ""

if [ ! -d "${MODEL_PATH}" ]; then
    echo "[ERROR] 模型路径不存在: ${MODEL_PATH}"
    exit 1
fi

# 清理已存在的 vLLM 进程
echo "清理已有 vLLM 进程..."
pkill -f "vllm serve ${MODEL_PATH}" 2>/dev/null || true
sleep 2

mkdir -p /tmp/vllm_logs

for i in $(seq 0 $((NUM_GPUS - 1))); do
    PORT=$((BASE_PORT + i))
    echo "[GPU ${i}] 在端口 ${PORT} 启动 vLLM..."

    CUDA_VISIBLE_DEVICES=${i} vllm serve "${MODEL_PATH}" \
        --host 0.0.0.0 \
        --port ${PORT} \
        --dtype auto \
        --served-model-name "${SERVED_MODEL_NAME}" \
        --max-num-seqs 16 \
        --gpu-memory-utilization 0.90 \
        --max-model-len 16384 \
        --enforce-eager \
        > /tmp/vllm_logs/vllm_gpu${i}.log 2>&1 &

    echo "  PID: $!"
    sleep 2  # 错开启动, 避免同时竞争显存/端口
done

echo ""
echo "=== 全部 ${NUM_GPUS} 个实例已启动, 等待健康检查 ==="
echo "日志: /tmp/vllm_logs/vllm_gpu{0..$((NUM_GPUS - 1))}.log"
echo ""

ALL_OK=1
for i in $(seq 0 $((NUM_GPUS - 1))); do
    PORT=$((BASE_PORT + i))
    OK=0
    for attempt in $(seq 1 180); do
        if curl -sf http://localhost:${PORT}/health > /dev/null 2>&1; then
            echo "  [GPU ${i}] 端口 ${PORT} 就绪"
            OK=1
            break
        fi
        sleep 5
    done
    if [ $OK -eq 0 ]; then
        echo "  [GPU ${i}] 端口 ${PORT} 启动失败 (查看 /tmp/vllm_logs/vllm_gpu${i}.log)"
        ALL_OK=0
    fi
done

echo ""
if [ $ALL_OK -eq 1 ]; then
    echo "✅ 全部 ${NUM_GPUS} 个 vLLM 就绪。测试: curl http://localhost:${BASE_PORT}/v1/models"
    exit 0
else
    echo "❌ 部分 vLLM 未就绪, 请检查日志。"
    exit 1
fi
