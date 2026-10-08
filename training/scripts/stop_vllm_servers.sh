#!/bin/bash
# 停止所有vLLM实例
set -e

MODEL_PATH="/path/to/spider_checkpoints/spider_20260421_0922/global_step_384/actor/huggingface"

echo "=== Stopping vLLM instances ==="
pkill -f "vllm serve ${MODEL_PATH}" 2>/dev/null || true
sleep 2

echo "Remaining vLLM processes:"
ps aux | grep "vllm serve" | grep -v grep || echo "  None (clean)"
echo "Done."
