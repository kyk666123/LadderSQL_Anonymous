#!/bin/bash
set -e

TOTAL_START=$(date +%s)
echo "=========================================="
echo " Spider 7B 全量数据（composite奖励）训练启动脚本"
echo " 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo " 训练数据: train.parquet (7000条全量)"
echo " train_batch_size=16 (与过滤版一致，step数约翻倍)"
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
# 阶段3: 复制 7B 模型 (OSS → 本地)
# ============================================================
echo ""
echo "[3/6] 复制 7B 模型 (OSS → /root/models)..."
T=$(date +%s)
mkdir -p /root/models
cp -r /path/to/models/Qwen/Qwen2.5-Coder-7B-Instruct \
      /root/models/Qwen2.5-Coder-7B-Instruct
echo "  ✓ 耗时: $(($(date +%s) - T))秒"

# ============================================================
# 阶段4: 复制 Spider 数据库 (OSS → 本地)
# ============================================================
echo ""
echo "[4/6] 复制 Spider 数据库 (OSS → /root/local_spider)..."
T=$(date +%s)
mkdir -p /root/local_spider
cp -r /path/to/nl2sql_dataset/spider/database      /root/local_spider/database
cp -r /path/to/nl2sql_dataset/spider/test_database /root/local_spider/test_database
echo "  ✓ 耗时: $(($(date +%s) - T))秒"

# ============================================================
# 阶段5: 复制训练/验证数据 parquet (OSS → 本地)
# 使用全量 train.parquet (7000条，非过滤版)
# ============================================================
echo ""
echo "[5/6] 复制训练数据 (OSS → /root/local_train_data)..."
T=$(date +%s)
mkdir -p /root/local_train_data/spider
cp /path/to/nl2sql_dataset/spider/train.parquet \
   /root/local_train_data/spider/train.parquet
cp /path/to/nl2sql_dataset/spider/test_dev_500.parquet \
   /root/local_train_data/spider/dev_500.parquet
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

# 运行时间戳：checkpoint 与轨迹统一归到带时间戳的运行目录，避免多次运行相互覆盖
RUN_TS=$(date '+%Y%m%d_%H%M%S')
OUTPUT_DIR=/path/to/training_outputs/ReAct_agent_spider_7b_full_${RUN_TS}
echo "本次运行输出目录: ${OUTPUT_DIR}"

# AliCloud LLM API（schema linking / summarizer 模式使用）
export ALICLOUD_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1/"
export ALICLOUD_API_KEY="${ALICLOUD_API_KEY:?set ALICLOUD_API_KEY to your LLM API key}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"   # leave empty and set WANDB_MODE=disabled to skip wandb

# Spider 数据库路径通过环境变量注入
export SPIDER_DATA_DIR=/root/local_spider

mkdir -p ${OUTPUT_DIR}/checkpoints

# 修复 agentlightning editable 路径（venv backup 中 .pth 指向旧路径）
echo "[fix] agentlightning path..."
ln -sfn /path/to/LadderSQL/agentlightning_framework /path/to/LadderSQL/agentlightning
echo "/path/to/LadderSQL" > /path/to/LadderSQL/.venv/lib/python3.12/site-packages/_agentlightning.pth
echo "  ✓ agentlightning path fixed"

/path/to/LadderSQL/.venv/bin/python -m original_agent.train_ReAct_agent \
    --schema-truncate golden \
    --dataset spider \
    --max-turns 10 \
    --local-model-path /root/models/Qwen2.5-Coder-7B-Instruct \
    --train-data /root/local_train_data/spider/train.parquet \
    --val-data /root/local_train_data/spider/dev_500.parquet \
    --epochs 2 \
    --train-batch-size 16 \
    --lr 1e-6 \
    --sample 16 \
    --temperature 1.1 \
    --entropy-coeff 0.0015 \
    --n-gpus 8 \
    --save-freq 64 \
    --test-freq 64 \
    --max-prompt-length 14336 \
    --max-response-length 2048 \
    --max-model-len 16384 \
    --prompt-truncate right \
    --n-runners 16 \
    --port 4747 \
    --project-name ReAct_agent_spider \
    --default-local-dir ${OUTPUT_DIR}/checkpoints \
    # --debug
    # --seq-masking
    # --resume-from-checkpoint ""
    # --val-before-train
    # --shuffle

# 训练完成后将本地轨迹日志复制到OSS（OSS FUSE不支持append模式，只能事后cp）
echo ""
echo "复制轨迹日志到OSS..."
mkdir -p ${OUTPUT_DIR}/trajectories
cp -r /path/to/LadderSQL/agent/react_trajectories/. \
    ${OUTPUT_DIR}/trajectories/ \
    && echo "  ✓ 轨迹日志已保存" \
    || echo "  WARNING: 轨迹日志复制失败（不影响训练结果）"
