#!/bin/bash
set -e

# ============================================================
# Bird 14B 训练 —— 冗余(高召回)schema 版
#   与 train_bird_14b_dlc.sh 的区别(仅两处业务差异 + 配套 schema 来源):
#     1. agent   : --agent-variant redundant (ReAct_agent_redundant_schema, 冗余schema甄别prompt)
#     2. 训练数据 : 14b 冗余采样过滤集 train_bird_14b_redundant_filtered.parquet (2694)
#     3. schema   : --schema-truncate cache + 预渲染冗余schema缓存(train∪old_dev 合并)
#                   val 用 old dev 500 (old_dev_500_redundant_val.parquet)
#     4. 上下文长度 : 与采样对齐 max_model_len=24576 / response=3072 / prompt=21504
#                   (早停预算 REACT_PROMPT_TOKEN_BUDGET=21504=24576-3072, 与采样一致)
#   其余训练超参(轮数/batch/lr/sample/温度)与 train_bird_14b_dlc.sh 一致。
# ============================================================

TOTAL_START=$(date +%s)
RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_TAG="${RUN_TAG:-}"
RUN_DIR="/path/to/training_outputs/ReAct_agent_bird_redundant/${RUN_TIMESTAMP}${RUN_TAG}"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints"

echo "=========================================="
echo " Bird 14B 训练启动脚本 [冗余schema版]"
echo " 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo " Checkpoint 目录: ${CHECKPOINT_DIR}"
echo "=========================================="

# ============================================================
# 阶段1: 复制项目代码 (OSS → 本地)
# ============================================================
echo ""
echo "[1/6] 复制项目代码 (OSS → /path/to/LadderSQL)..."
T=$(date +%s)
cp -r /path/to/LadderSQL /path/to/LadderSQL
echo "  ✓ 耗时: $(($(date +%s) - T))秒"

# ============================================================
# 阶段2: 复制并解压 venv
# ============================================================
echo ""
echo "[2/6] 复制 venv 压缩包 (OSS → /root, 5.7G)..."
T=$(date +%s)
cp /path/to/venv_backup.tar.gz /root/venv_backup.tar.gz
echo "  ✓ 耗时: $(($(date +%s) - T))秒"

echo ""
echo "[2b] 解压 uv Python 解释器..."
T=$(date +%s)
mkdir -p /root/.local/share/uv/python/
tar -xf /path/to/uv_python_3.12.12.tar -C /root/.local/share/uv/python/
echo "  ✓ 耗时: $(($(date +%s) - T))秒"

echo ""
echo "[2c] 本地解压 venv (~13G, 约2分钟) → /path/to/LadderSQL/.venv..."
T=$(date +%s)
tar -xzf /root/venv_backup.tar.gz -C /path/to/LadderSQL/
echo "  ✓ 耗时: $(($(date +%s) - T))秒"

# ============================================================
# 阶段3: 复制 14B 模型 (OSS → 本地)
# ============================================================
echo ""
echo "[3/6] 复制 14B 模型 (OSS → /root/models)..."
T=$(date +%s)
mkdir -p /root/models
cp -r /path/to/models/Qwen/Qwen2.5-Coder-14B-Instruct \
      /root/models/Qwen2.5-Coder-14B-Instruct
echo "  ✓ 耗时: $(($(date +%s) - T))秒"

# ============================================================
# 阶段4: 复制 Bird 数据库 (OSS → 本地，路径与代码硬编码一致)
#   train_databases: 训练题目所在库; dev_databases: 取自 old dev (dev_20240627), 与 val 集对齐
# ============================================================
echo ""
echo "[4/6] 准备 Bird 数据库 (OSS → /root/local_bird_db)..."
T=$(date +%s)
mkdir -p /root/local_bird_db
unzip -q -o /path/to/nl2sql_dataset/bird/train/train_databases.zip -d /root/local_bird_db
cp -r /path/to/nl2sql_dataset/bird/dev_20240627/dev_databases  /root/local_bird_db/dev_databases
echo "  ✓ 耗时: $(($(date +%s) - T))秒"

# ============================================================
# 阶段5: 复制训练/验证数据 parquet + 冗余 schema 缓存 (OSS → 本地)
# ============================================================
echo ""
echo "[5/6] 复制训练/验证数据 + 冗余schema缓存 (OSS → /root/local_train_data)..."
T=$(date +%s)
mkdir -p /root/local_train_data/bird
# 训练集: 14b 冗余采样过滤集 (2694 mixed)
cp /path/to/nl2sql_dataset/bird/bird_clean_data/train_bird_14b_redundant_filtered.parquet \
   /root/local_train_data/bird/train.parquet
# 验证集: old dev 随机500
cp /path/to/nl2sql_dataset/bird/bird_clean_data/old_dev_500_redundant_val.parquet \
   /root/local_train_data/bird/dev_500.parquet
# 预渲染冗余 schema 缓存 (train ∪ old_dev, key=db_id|||question), agent 用 truncate=cache 消费
cp /path/to/schema_link_results/bird_train/schema_cache_train_plus_olddev_rendered.json \
   /root/local_train_data/bird/schema_cache_redundant.json
echo "  ✓ 耗时: $(($(date +%s) - T))秒"

# ============================================================
# 准备完成，打印总耗时
# ============================================================
PREPARE_TIME=$(($(date +%s) - TOTAL_START))
echo ""
echo "=========================================="
echo " 本地数据准备完成！耗时: ${PREPARE_TIME}秒"
echo " 开始训练..."
echo "=========================================="

# ============================================================
# 阶段6: 启动训练 (全部使用本地路径，checkpoint 写回 OSS)
# ============================================================
cd /path/to/LadderSQL/agent
export PATH=/path/to/LadderSQL/.venv/bin:$PATH

# AliCloud LLM API（schema linking / summarizer 模式使用；cache 模式其实用不到，保留不影响）
export ALICLOUD_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1/"
export ALICLOUD_API_KEY="${ALICLOUD_API_KEY:?set ALICLOUD_API_KEY to your LLM API key}"

# 本地测试禁用 wandb（DLC 运行时设置 WANDB_API_KEY 后可删除此行）
# export WANDB_MODE=disabled
export WANDB_API_KEY="${WANDB_API_KEY:-}"   # leave empty and set WANDB_MODE=disabled to skip wandb

mkdir -p "${CHECKPOINT_DIR}"
echo "Checkpoints will be saved to: ${CHECKPOINT_DIR}"

# 修复 agentlightning editable 路径（venv backup 中 .pth 指向旧路径）
echo "[fix] agentlightning path..."
ln -sfn /path/to/LadderSQL/agentlightning_framework /path/to/LadderSQL/agentlightning
echo "/path/to/LadderSQL" > /path/to/LadderSQL/.venv/lib/python3.12/site-packages/_agentlightning.pth
echo "  ✓ agentlightning path fixed"

# 早停 prompt token 预算与采样对齐: = max_model_len - max_response_length = 24576 - 3072
# (ReAct_Agent 默认 12288 是旧训练配置; 冗余 schema 更大, 必须上调以免误早停)
export REACT_PROMPT_TOKEN_BUDGET=21504

/path/to/LadderSQL/.venv/bin/python -m original_agent.train_ReAct_agent \
    --agent-variant redundant \
    --schema-truncate cache \
    --schema-cache-path /root/local_train_data/bird/schema_cache_redundant.json \
    --dataset bird \
    --max-turns 8 \
    --local-model-path /root/models/Qwen2.5-Coder-14B-Instruct \
    --train-data /root/local_train_data/bird/train.parquet \
    --val-data /root/local_train_data/bird/dev_500.parquet \
    --epochs 2 \
    --train-batch-size 16 \
    --lr 1e-6 \
    --sample 16 \
    --temperature 1.1 \
    --entropy-coeff 0.0015 \
    --n-gpus 8 \
    --save-freq 64 \
    --test-freq 64 \
    --max-prompt-length 21504 \
    --max-response-length 3072 \
    --max-model-len 24576 \
    --prompt-truncate right \
    --n-runners 16 \
    --port 4747 \
    --project-name ReAct_agent_bird_redundant \
    --default-local-dir "${CHECKPOINT_DIR}"
    # --debug
    # --seq-masking
    # --resume-from-checkpoint ""
    # --val-before-train
    # --shuffle

# 训练完成后将本地轨迹日志复制到OSS（OSS FUSE不支持append模式，只能事后cp）
echo ""
echo "复制轨迹日志到OSS..."
mkdir -p ${RUN_DIR}/trajectories
cp -r /path/to/LadderSQL/agent/react_trajectories/. \
    ${RUN_DIR}/trajectories/ \
    && echo "  ✓ 轨迹日志已保存" \
    || echo "  WARNING: 轨迹日志复制失败（不影响训练结果）"
