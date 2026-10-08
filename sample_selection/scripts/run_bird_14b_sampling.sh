#!/bin/bash
# ============================================================
# DLC 一键运行: BIRD train 14B 采样 (每样本 16 条轨迹, 各自 0/1 奖励)
#   Step 0. 把 14B 模型 与 BIRD 数据库 复制/解压到本地磁盘
#   Step 1. 启动 8 张卡的 vLLM (14B, 端口 9000-9007)
#   Step 2. 每个样本采样 16 条轨迹, 均匀分发到 8 个端点 (max-turns=8, 温度1.1)
#           奖励用 binary_reward_bird, 结果记录每个样本的 16 条奖励
# 断点续跑: 直接重复执行本脚本即可 (采样脚本会跳过已完成样本)。
# ============================================================
set -e

# ---------------- 环境 / venv ----------------
VENV_BIN="${VENV_BIN:-/path/to/venv/bin}"
PY="${VENV_BIN}/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="${VENV_BIN}:${PATH}"

# ---------------- 源路径 (OSS / ModelScope) ----------------
export MODEL_SRC="${MODEL_SRC:-/path/to/models/Qwen/Qwen2.5-Coder-14B-Instruct}"
export BIRD_ZIP="${BIRD_ZIP:-/path/to/nl2sql_dataset/bird/train/train_databases.zip}"

# ---------------- 本地目标路径 ----------------
export LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-/root/local_models/Qwen2.5-Coder-14B-Instruct}"
export LOCAL_DB_ROOT="${LOCAL_DB_ROOT:-/root/local_bird_db}"
LOCAL_DB_DIR="${LOCAL_DB_ROOT}/train_databases"

# ---------------- 采样参数 ----------------
export MODEL_PATH="${LOCAL_MODEL_DIR}"     # vLLM 加载本地模型
export NUM_GPUS="${NUM_GPUS:-8}"
N_TRIALS="${N_TRIALS:-16}"
PARQUET="${PARQUET:-/path/to/nl2sql_dataset/bird/bird_clean_data/train.parquet}"
OUTPUT="${OUTPUT:-/path/to/sampling_outputs/14b_bird_train_maxturns8_golden_sampling.json}"

echo "############################################################"
echo "# BIRD train 14B 采样 (每样本 16 条轨迹 + 各自奖励)"
echo "#   模型源:   ${MODEL_SRC}"
echo "#   本地模型: ${LOCAL_MODEL_DIR}"
echo "#   数据zip:  ${BIRD_ZIP}"
echo "#   本地数据: ${LOCAL_DB_DIR}"
echo "#   卡数:     ${NUM_GPUS}   采样次数/样本: ${N_TRIALS}"
echo "#   题目:     ${PARQUET}"
echo "#   输出:     ${OUTPUT}"
echo "############################################################"

# ---------------- Step 0: 复制/解压到本地 ----------------
echo ""
echo ">>> [Step 0/2] 准备本地模型与数据..."
bash "${SCRIPT_DIR}/prepare_local_data.sh"

# ---------------- Step 1: 启动 vLLM ----------------
echo ""
echo ">>> [Step 1/2] 启动 ${NUM_GPUS} 个 vLLM 实例 (14B, 本地模型)..."
bash "${SCRIPT_DIR}/start_vllm_14b.sh"

# ---------------- Step 2: 采样 ----------------
echo ""
echo ">>> [Step 2/2] 开始采样 (每样本 ${N_TRIALS} 条轨迹, max-turns=8, 温度1.1)..."
cd "${SCRIPT_DIR}"
"${PY}" sample_bird_14b.py \
    --parquet "${PARQUET}" \
    --db-dir "${LOCAL_DB_DIR}" \
    --output "${OUTPUT}" \
    --n-trials "${N_TRIALS}" \
    --num-gpus "${NUM_GPUS}"

echo ""
echo "✅ 全部完成! 每个样本的 16 条奖励已记录在 reward_list 字段:"
echo "   ${OUTPUT}"
echo ""
echo "如需进一步过滤 16 条全0/全1(方差为0) 的样本, 可执行:"
echo "   ${PY} filter_variance_zero.py --input ${OUTPUT} --output ${OUTPUT%.json}_filtered.json"
echo ""
echo "如需释放显存, 停止 vLLM: pkill -f 'vllm serve ${MODEL_PATH}'"
