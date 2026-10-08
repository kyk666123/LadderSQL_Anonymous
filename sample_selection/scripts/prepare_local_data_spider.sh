#!/bin/bash
# ============================================================
# DLC 准备步骤(Spider 版): 把 14B 模型 与 Spider test 数据库 复制到本地磁盘
#   - 模型: checkpoint huggingface 目录 -> 本地 (28G, 直接 cp)
#   - 数据: Spider test_database 是**目录**(非 zip), 直接 cp -a 复制到本地
#           (与 BIRD 的 unzip 流程不同; Spider DB 结构 {db_id}/{db_id}.sqlite)
# 幂等: 已存在且完整则跳过, 可重复执行。
# ============================================================
set -e

# ---- 源路径 ----
MODEL_SRC="${MODEL_SRC:-/path/to/training_outputs/spider_20260629_1530/20260629_153045/checkpoints/global_step_320/actor/huggingface}"
# Spider test 库(目录, 非 zip)
SPIDER_DB_SRC="${SPIDER_DB_SRC:-/path/to/nl2sql_dataset/spider/test_database}"

# ---- 本地目标路径 ----
LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-/root/local_models/spider_ckpt_step320}"
LOCAL_DB_ROOT="${LOCAL_DB_ROOT:-/root/local_spider_db}"
DB_SUBDIR="${DB_SUBDIR:-test_database}"
DB_MIN_COUNT="${DB_MIN_COUNT:-200}"            # Spider test 约 206 个库
MODEL_MIN_SHARDS="${MODEL_MIN_SHARDS:-6}"      # 判定模型完整的 safetensors 分片数(14B=7 用6; 7B=4 用4)
LOCAL_DB_DIR="${LOCAL_DB_ROOT}/${DB_SUBDIR}"

echo "=== [Prepare/Spider] 准备本地模型与数据 ==="

# ---------------- 1. 复制 14B 模型到本地 ----------------
# SKIP_MODEL_COPY=1 时跳过模型复制(用于走 API 的采样, 只需本地 Spider 库算 reward)。
if [ "${SKIP_MODEL_COPY:-0}" = "1" ]; then
    echo ""
    echo ">>> 模型: 跳过复制 (SKIP_MODEL_COPY=1, 走远程 API 采样)"
    NEED_COPY_MODEL=0
fi
echo ""
echo ">>> 模型: ${MODEL_SRC}"
echo "         -> ${LOCAL_MODEL_DIR}"
NEED_COPY_MODEL="${NEED_COPY_MODEL:-1}"
if [ -f "${LOCAL_MODEL_DIR}/config.json" ]; then
    N=$(ls "${LOCAL_MODEL_DIR}"/*.safetensors 2>/dev/null | wc -l)
    if [ "$N" -ge "$MODEL_MIN_SHARDS" ]; then
        echo "    已存在完整模型 (${N} 个 safetensors), 跳过。"
        NEED_COPY_MODEL=0
    fi
fi
if [ "$NEED_COPY_MODEL" -eq 1 ]; then
    if [ ! -d "${MODEL_SRC}" ]; then
        echo "[ERROR] 模型源不存在: ${MODEL_SRC}"; exit 1
    fi
    mkdir -p "${LOCAL_MODEL_DIR}"
    echo "    复制中 (约 28G, 请耐心等待)..."
    cp -a "${MODEL_SRC}/." "${LOCAL_MODEL_DIR}/"
    echo "    模型复制完成。"
fi

# ---------------- 2. 复制 Spider test 数据库目录到本地 ----------------
echo ""
echo ">>> 数据: ${SPIDER_DB_SRC}"
echo "         -> ${LOCAL_DB_DIR}"
NEED_COPY_DB=1
if [ -d "${LOCAL_DB_DIR}" ]; then
    N=$(ls -d "${LOCAL_DB_DIR}"/*/ 2>/dev/null | grep -v MACOSX | wc -l)
    if [ "$N" -ge "$DB_MIN_COUNT" ]; then
        echo "    已存在数据库目录 (${N} 个库), 跳过。"
        NEED_COPY_DB=0
    fi
fi
if [ "$NEED_COPY_DB" -eq 1 ]; then
    if [ ! -d "${SPIDER_DB_SRC}" ]; then
        echo "[ERROR] Spider test 数据库目录不存在: ${SPIDER_DB_SRC}"; exit 1
    fi
    mkdir -p "${LOCAL_DB_DIR}"
    echo "    复制中 (目录直接 cp, 约 865M)..."
    cp -a "${SPIDER_DB_SRC}/." "${LOCAL_DB_DIR}/"
    rm -rf "${LOCAL_DB_DIR}/__MACOSX" 2>/dev/null || true
    echo "    数据复制完成。"
fi

echo ""
echo "✅ [Prepare/Spider] 完成"
echo "   本地模型: ${LOCAL_MODEL_DIR}"
echo "   本地数据: ${LOCAL_DB_DIR}"
