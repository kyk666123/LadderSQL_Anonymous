#!/bin/bash
# ============================================================
# redundant-schema 采样驱动 (被 3 个 DLC 入口脚本调用)
#   Step 0. 把模型 与 BIRD 库 复制/解压到本地磁盘 (prepare_local_data.sh)
#   Step 1. 启动 8 张卡的 vLLM (start_vllm_redundant.sh)
#   Step 2. 每样本采 N_TRIALS 条轨迹, 用 redundant-schema agent + binary_reward_bird
# 全部差异(模型路径/served名/思考开关/上下文长度/输出) 由调用方经环境变量传入。
# 断点续跑: 直接重复执行即可 (采样脚本会跳过已完成样本)。
# ============================================================
set -e

VENV_BIN="${VENV_BIN:-/path/to/venv/bin}"
PY="${VENV_BIN}/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="${VENV_BIN}:${PATH}"

# ---------------- 源 / 本地路径 ----------------
export MODEL_SRC="${MODEL_SRC:-/path/to/models/Qwen/Qwen2.5-Coder-14B-Instruct}"
export BIRD_ZIP="${BIRD_ZIP:-/path/to/nl2sql_dataset/bird/train/train_databases.zip}"
export LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-/root/local_models/Qwen2.5-Coder-14B-Instruct}"
export LOCAL_DB_ROOT="${LOCAL_DB_ROOT:-/root/local_bird_db}"
export DB_SUBDIR="${DB_SUBDIR:-train_databases}"   # dev 采样传 dev_databases
LOCAL_DB_DIR="${LOCAL_DB_ROOT}/${DB_SUBDIR}"

# ---------------- vLLM / 采样参数 (由入口脚本设定) ----------------
export MODEL_PATH="${LOCAL_MODEL_DIR}"
export NUM_GPUS="${NUM_GPUS:-8}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen2.5-coder}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
export EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
export ENABLE_THINKING="${ENABLE_THINKING:-}"     # true/false/(空)
export THINKING_BUDGET="${THINKING_BUDGET:-}"     # 思考 token 预算(仅 thinking=true 生效)
export MAX_TOKENS="${MAX_TOKENS:-2048}"
export TEMPERATURE="${TEMPERATURE:-1.1}"
export LLM_TIMEOUT="${LLM_TIMEOUT:-45}"
export SCHEMA_CACHE="${SCHEMA_CACHE:-/path/to/schema_link_results/bird_train/schema_cache_train_rendered.json}"

N_TRIALS="${N_TRIALS:-16}"
# 采样脚本(默认方差采样; 带 SQL 落盘的变体传 SAMPLE_SCRIPT=sample_bird_redundant_with_sql.py)
SAMPLE_SCRIPT="${SAMPLE_SCRIPT:-sample_bird_redundant.py}"
PER_GPU="${PER_GPU:-10}"
TRAJ_TIMEOUT="${TRAJ_TIMEOUT:-120}"
PARQUET="${PARQUET:-/path/to/nl2sql_dataset/bird/bird_clean_data/train.parquet}"
OUTPUT="${OUTPUT:-/path/to/sampling_outputs/redundant_bird_train_sampling.json}"

echo "############################################################"
echo "# BIRD train redundant-schema 采样"
echo "#   模型源:   ${MODEL_SRC}"
echo "#   本地模型: ${LOCAL_MODEL_DIR}   served: ${SERVED_MODEL_NAME}"
echo "#   thinking: ${ENABLE_THINKING:-<off/none>}   budget: ${THINKING_BUDGET:-<none>}   max_tokens: ${MAX_TOKENS}   temp: ${TEMPERATURE}"
echo "#   vLLM:     max_model_len=${MAX_MODEL_LEN} max_num_seqs=${MAX_NUM_SEQS} extra='${EXTRA_VLLM_ARGS}'"
echo "#   schema:   ${SCHEMA_CACHE}"
echo "#   卡数:     ${NUM_GPUS}   采样/样本: ${N_TRIALS}   并发/卡: ${PER_GPU}   轨迹超时: ${TRAJ_TIMEOUT}s"
echo "#   题目:     ${PARQUET}"
echo "#   输出:     ${OUTPUT}"
echo "############################################################"

# ---------------- Step 0: 本地化模型/数据 ----------------
echo ""; echo ">>> [Step 0/2] 准备本地模型与数据..."
bash "${SCRIPT_DIR}/prepare_local_data.sh"

# ---------------- Step 1: 启动 vLLM ----------------
echo ""; echo ">>> [Step 1/2] 启动 ${NUM_GPUS} 个 vLLM 实例..."
bash "${SCRIPT_DIR}/start_vllm_redundant.sh"

# ---------------- Step 2: 采样 ----------------
echo ""; echo ">>> [Step 2/2] 开始采样 (redundant-schema agent)..."
cd "${SCRIPT_DIR}"
echo "#   采样脚本: ${SAMPLE_SCRIPT}   采样/样本: ${N_TRIALS}"
"${PY}" "${SAMPLE_SCRIPT}" \
    --parquet "${PARQUET}" \
    --db-dir "${LOCAL_DB_DIR}" \
    --output "${OUTPUT}" \
    --n-trials "${N_TRIALS}" \
    --num-gpus "${NUM_GPUS}" \
    --per-gpu "${PER_GPU}" \
    --traj-timeout "${TRAJ_TIMEOUT}"

echo ""
echo "✅ 采样完成, 每样本 ${N_TRIALS} 条奖励记录在 reward_list: ${OUTPUT}"
echo "过滤 16 条全0/全1(方差为0) 的样本:"
echo "   ${PY} filter_variance_zero.py --input ${OUTPUT} --output ${OUTPUT%.json}_filtered.json"
echo "释放显存: pkill -f 'vllm serve ${MODEL_PATH}'"
