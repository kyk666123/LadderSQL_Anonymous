#!/bin/bash
# ============================================================
# DLC 准备步骤: 把 14B 模型 与 BIRD train 数据库 复制/解压到本地磁盘
#   - 模型: ModelScope 下载目录 -> 本地 (28G, 直接 cp)
#   - 数据: train_databases.zip -> 本地解压 (8.8G, 必须 unzip 解压, 不能直接 cp 目录)
# 幂等: 已存在且完整则跳过, 可重复执行。
# ============================================================
set -e

# ---- 源路径 (OSS / ModelScope) ----
MODEL_SRC="${MODEL_SRC:-/path/to/models/Qwen/Qwen2.5-Coder-14B-Instruct}"
BIRD_ZIP="${BIRD_ZIP:-/path/to/nl2sql_dataset/bird/train/train_databases.zip}"

# ---- 本地目标路径 ----
LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-/root/local_models/Qwen2.5-Coder-14B-Instruct}"
LOCAL_DB_ROOT="${LOCAL_DB_ROOT:-/root/local_bird_db}"
DB_SUBDIR="${DB_SUBDIR:-train_databases}"      # dev 采样传 dev_databases
DB_MIN_COUNT="${DB_MIN_COUNT:-60}"             # 已存在库数达到此值则跳过解压(dev 传 10)
MODEL_MIN_SHARDS="${MODEL_MIN_SHARDS:-6}"      # 判定模型完整的 safetensors 分片数(14B=7 用6; 7B=4 用4)
LOCAL_DB_DIR="${LOCAL_DB_ROOT}/${DB_SUBDIR}"

echo "=== [Prepare] 准备本地模型与数据 ==="

# ---------------- 1. 复制 14B 模型到本地 ----------------
# SKIP_MODEL_COPY=1 时跳过模型复制(用于走 API 的采样, 只需本地 BIRD 库算 reward)。
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

# ---------------- 2. 解压 BIRD train 数据库到本地 ----------------
echo ""
echo ">>> 数据: ${BIRD_ZIP}"
echo "         -> ${LOCAL_DB_DIR}"
NEED_UNZIP=1
if [ -d "${LOCAL_DB_DIR}" ]; then
    N=$(ls -d "${LOCAL_DB_DIR}"/*/ 2>/dev/null | grep -v MACOSX | wc -l)
    if [ "$N" -ge "$DB_MIN_COUNT" ]; then
        echo "    已存在解压后的数据库 (${N} 个库), 跳过。"
        NEED_UNZIP=0
    fi
fi
if [ "$NEED_UNZIP" -eq 1 ]; then
    if [ ! -f "${BIRD_ZIP}" ]; then
        echo "[ERROR] 数据 zip 不存在: ${BIRD_ZIP}"; exit 1
    fi
    mkdir -p "${LOCAL_DB_ROOT}"
    echo "    解压中 (必须解压而非直接复制)..."
    # zip 顶层即为 ${DB_SUBDIR}/, 解压到 LOCAL_DB_ROOT 下
    unzip -q -o "${BIRD_ZIP}" -d "${LOCAL_DB_ROOT}"
    rm -rf "${LOCAL_DB_ROOT}/__MACOSX" 2>/dev/null || true
    echo "    数据解压完成。"
fi

echo ""
echo "✅ [Prepare] 完成"
echo "   本地模型: ${LOCAL_MODEL_DIR}"
echo "   本地数据: ${LOCAL_DB_DIR}"
