#!/usr/bin/env python3
"""BIRD train 数据集 —— 14B 模型方差采样。

对 BIRD train 的每个样本采样 N_TRIALS(默认16) 次，16 条 rollout 均匀分发到
8 个 vLLM 端点(每卡1个, 端口 9000-9007), 记录每次的 0/1 reward。

采样 agent 流程使用 original_agent/ReAct_Agent.py 里的基础 `Agent`
(Think -> SQL -> Observation 循环), BIRD 专用 prompt (含 evidence)。

采样完成后可用 filter_variance_zero.py 过滤掉 16 条全 0 或全 1 的样本
(即方差为 0), 只保留 mixed(有难有易) 的样本。

用法:
    python sample_bird_14b.py
    # 或指定参数
    python sample_bird_14b.py --parquet ... --output ... --n-trials 16 --num-gpus 8

断点续跑: 重复执行会跳过已完成样本, 并自动重跑标记为 timed_out 的样本。
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
import logging
import math
import signal
import threading
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
from typing import Any, Dict, List, Optional

import pandas as pd

# ---- 让 original_agent / reward_func 可被导入(指向 agent/ 目录) ----
_AGENT_DIR = Path(__file__).resolve().parent.parent / "training"  # provides original_agent + reward_func
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))

from original_agent.ReAct_Agent import Agent  # type: ignore
from original_agent.prompt import REACT_SQL_BIRD_PROMPT  # type: ignore
from reward_func.reward import binary_reward_bird  # type: ignore

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("sample_bird_14b")
logger.setLevel(logging.INFO)

# ============================ 默认配置 ============================
DEFAULT_PARQUET = "/path/to/nl2sql_dataset/bird/bird_clean_data/train.parquet"
DEFAULT_DB_DIR = "/root/local_bird_db/train_databases"   # {db_id}/{db_id}.sqlite
DEFAULT_OUTPUT = "/path/to/sampling_outputs/14b_bird_train_maxturns8_golden_sampling.json"

N_TRIALS = 16               # 每个样本采样次数
NUM_GPUS = 8                # vLLM 实例数(每卡一个)
BASE_PORT = 9000            # vLLM 起始端口 -> 9000..9007
SERVED_MODEL_NAME = "qwen2.5-coder"  # 与启动 vLLM 的 --served-model-name 保持一致
MAX_TURNS = 8              # 必须与训练 train_bird_14b_dlc.sh 的 --max-turns 对齐
EXECUTION_TRUNCATE = 2048
TEMPERATURE = 1.1
LLM_TIMEOUT = 45            # 单次 LLM 请求超时(秒), 防止 vLLM 连接挂起
TRAJ_TIMEOUT = 120          # 单条 rollout 硬超时(秒): SQL每条最15s × max-turns 8 = 120s 上限
PARALLEL_PER_GPU = 10       # 每个 vLLM 端点并发的 rollout 数 -> 总并发 = num_gpus * 该值(L20 46G KV上限约10-16)
SAVE_EVERY = 16             # 每完成多少样本落盘一次(流式, 断点续跑粒度)
RECURSION_LIMIT = 100

# ---- 给 Agent 内部 LLM 调用注入请求级超时 + 降低重试, 防止 vLLM 连接挂起拖垮整体 ----
import original_agent.ReAct_Agent as _react_mod  # type: ignore  # noqa: E402
_ORIG_INIT_CHAT_MODEL = _react_mod.init_chat_model
def _init_chat_model_patched(*a, **k):
    k.setdefault("timeout", LLM_TIMEOUT)
    k["max_retries"] = 1  # 原为 3, 连接挂起时会把等待放大 3 倍
    return _ORIG_INIT_CHAT_MODEL(*a, **k)
_react_mod.init_chat_model = _init_chat_model_patched

# ---- golden pruned schema: 与训练 train_bird_14b_dlc.sh 的 --schema-truncate golden 对齐 ----
# 训练时(LitAgent, truncate=="golden")从该文件按 (db_id, question) 取 pruned_schema 注入 prompt,
# 采样必须用同一份 schema, 否则数据分布与训练不一致。
GOLD_SCHEMA_FILE = (
    Path(__file__).resolve().parent.parent.parent
    / "schema_preprocessing" / "gold_schema_rule_based" / "bird_train_schema.json"
)


def _make_key(db_id: str, question: str) -> str:
    # 与 ReAct_Agent.py golden 分支完全一致: 归一化空白后拼接
    return f"{db_id}||{' '.join(question.strip().split())}"


def _load_pruned_schema_map(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(
            f"golden schema 文件不存在: {path} "
            f"(采样必须与训练 --schema-truncate golden 对齐)"
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    m: Dict[str, str] = {}
    for item in data:
        m[_make_key(item["db_id"], item["question"])] = item["pruned_schema"]
    logger.info(f"[schema] 已加载 {len(m)} 条 golden pruned schema: {path}")
    return m


# 模块级加载(fork 前), 子进程继承, 无需重复读取
_PRUNED_SCHEMA_MAP: Dict[str, str] = _load_pruned_schema_map(GOLD_SCHEMA_FILE)


def build_endpoints(num_gpus: int) -> List[str]:
    return [f"http://localhost:{BASE_PORT + i}/v1" for i in range(num_gpus)]


def get_db_path(db_dir: str, db_id: str) -> str:
    return os.path.join(db_dir, db_id, f"{db_id}.sqlite")


def get_golden_schema(db_id: str, question: str) -> str:
    """取训练同款 golden pruned schema(按 db_id+question), 不截断。miss 时与训练一致返回占位串。"""
    key = _make_key(db_id, question)
    schema = _PRUNED_SCHEMA_MAP.get(key)
    if schema is None:
        logger.warning(f"[schema] golden pruned schema 未命中: {key[:100]}")
        return "No pruned schema available."
    return schema


def run_one_trial(task: Dict[str, Any], endpoint: str, db_dir: str) -> Optional[float]:
    """单条 rollout: 跑 agent 得到 SQL, 再算 0/1 reward。失败返回 None。"""
    db_id = task["db_id"]
    question = task["question"]
    ground_truth = task["query"]
    evidence = task.get("evidence", "") or ""

    db_file = get_db_path(db_dir, db_id)
    if not os.path.exists(db_file):
        logger.error(f"[trial] 数据库不存在: {db_file}")
        return None

    # 只读连接, 避免并发写锁(NL2SQL 全为 SELECT, 多读安全)
    db_uri = f"sqlite:///{db_file}"
    # schema 与训练对齐: 使用 golden pruned schema(而非完整 DDL)
    db_schema = get_golden_schema(db_id, question)

    try:
        agent = Agent(
            db_path=db_uri,
            db_schema=db_schema,
            max_turns=MAX_TURNS,
            endpoint=endpoint,
            verl_replacement={"model": SERVED_MODEL_NAME, "temperature": TEMPERATURE},
            execution_truncate=EXECUTION_TRUNCATE,
            prompt_template=REACT_SQL_BIRD_PROMPT,
            evidence=evidence,
        ).graph()
        result = agent.invoke({"question": question}, {"recursion_limit": RECURSION_LIMIT})
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[trial] db_id={db_id} agent 执行失败: {e}")
        return None

    pred_sql = result.get("query", "") if isinstance(result, dict) else ""
    try:
        reward = binary_reward_bird(pred_sql, ground_truth, db_file, raise_on_error=False)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[trial] db_id={db_id} reward 计算失败: {e}")
        return None
    return float(reward)


def _run_trial_task(packed):
    """子进程执行单条 rollout: 用 SIGALRM 施加硬超时(进程可被杀), 保证绝不卡死。"""
    task, endpoint, db_dir, traj_timeout = packed

    def _on_alarm(signum, frame):
        raise TimeoutError(f"trajectory exceeded {traj_timeout}s")

    old = signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(int(traj_timeout))
    try:
        return run_one_trial(task, endpoint, db_dir)
    except BaseException as e:  # noqa: BLE001 (含 TimeoutError)
        logger.warning(f"[trial] db={task.get('db_id')} 超时/异常, 记为 None: {e}")
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


# ============================ 结果持久化 ============================

def load_existing(output_file: str) -> List[Dict]:
    if not os.path.exists(output_file):
        return []
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"结果文件损坏({e}), 从头开始。")
        return []
    if data and any("task_idx" not in r for r in data):
        logger.warning("检测到旧格式(无 task_idx)结果, 不兼容流式续跑, 将从头开始(旧文件请另存)。")
        return []
    print(f"[INFO] 已加载 {len(data)} 条历史结果: {output_file}")
    return data


def save_results(output_file: str, results: List[Dict]) -> None:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    tmp = output_file + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    os.replace(tmp, output_file)


def find_timed_out(results: List[Dict]) -> List[int]:
    return [i for i, r in enumerate(results) if r.get("timed_out", False)]


def summarize(results: List[Dict]) -> None:
    all_one = all_zero = mixed = timed = invalid = 0
    for s in results:
        if s.get("timed_out"):
            timed += 1
            continue
        rl = s.get("reward_list", [])
        if not rl or any(r is None for r in rl):
            invalid += 1
            continue
        if all(r == 1.0 for r in rl):
            all_one += 1
        elif all(r == 0.0 for r in rl):
            all_zero += 1
        else:
            mixed += 1
    total = len(results)
    print("\n" + "=" * 70)
    print(f"采样统计 (total={total})")
    print(f"  mixed(保留, 方差>0): {mixed}")
    print(f"  all_1 (剔除, 太简单): {all_one}")
    print(f"  all_0 (剔除, 太难)  : {all_zero}")
    print(f"  timed_out           : {timed}")
    print(f"  invalid(含None)     : {invalid}")
    print("=" * 70)


# ============================ 主流程 ============================

def process(args) -> None:
    import multiprocessing as mp
    from collections import deque, defaultdict

    tasks = pd.read_parquet(args.parquet).to_dict(orient="records")
    total = len(tasks)
    endpoints = build_endpoints(args.num_gpus)
    max_inflight = max(1, args.num_gpus * args.per_gpu)
    print(f"[INFO] 样本总数: {total}")
    print(f"[INFO] 每样本采样: {args.n_trials} 次 -> {len(endpoints)} 个端点; 并发 rollout: {max_inflight} (流式)")
    print(f"[INFO] max_turns={MAX_TURNS} 单轨迹硬超时={args.traj_timeout}s 单次LLM超时={LLM_TIMEOUT}s")
    print(f"[INFO] 结果文件: {args.output}")

    results = load_existing(args.output)
    done_idx = {r["task_idx"] for r in results if "task_idx" in r}
    pending = [i for i in range(total) if i not in done_idx]
    if not pending:
        print(f"[INFO] 全部 {total} 个样本已完成。")
        summarize(results)
        return
    print(f"[INFO] 待采样 {len(pending)} / {total} (已完成 {len(done_idx)})")

    ctx = mp.get_context("fork")
    executor = ProcessPoolExecutor(max_workers=max_inflight, mp_context=ctx)

    task_queue = deque((si, ti) for si in pending for ti in range(args.n_trials))
    futures = {}                       # fut -> (si, ti)
    per_sample = defaultdict(lambda: [None] * args.n_trials)
    remaining = {si: args.n_trials for si in pending}
    counter = 0

    def submit_next():
        nonlocal counter
        si, ti = task_queue.popleft()
        ep = endpoints[counter % len(endpoints)]
        counter += 1
        fut = executor.submit(_run_trial_task, (tasks[si], ep, args.db_dir, args.traj_timeout))
        futures[fut] = (si, ti)

    def finalize(si):
        rl = per_sample.pop(si)
        task = tasks[si]
        timed = any(r is None for r in rl)
        results.append({
            "task_idx": si,
            "db_id": task["db_id"],
            "question": task["question"],
            "query": task["query"],
            "evidence": task.get("evidence", ""),
            "reward_list": rl,
            "timed_out": timed,
        })
        valid = [r for r in rl if r is not None]
        avg = (sum(valid) / len(valid)) if valid else 0.0
        return timed, avg

    # 初始填满 in-flight 窗口
    while task_queue and len(futures) < max_inflight:
        submit_next()

    completed = 0
    since_save = 0
    t_start = time.time()
    last_progress = time.time()
    try:
        while futures:
            done, _ = wait(list(futures.keys()), timeout=30, return_when=FIRST_COMPLETED)
            if not done:
                # 兑底: 长时间无任何完成 -> 疑似卡死, 放弃在飞任务并重建池
                if time.time() - last_progress > args.traj_timeout + 120:
                    logger.error(f"[stuck] {len(futures)} 个 worker 长时间无响应, 标记为超时并重建进程池。")
                    for f, (si, ti) in list(futures.items()):
                        per_sample[si][ti] = None
                        remaining[si] -= 1
                        if remaining[si] == 0:
                            finalize(si); completed += 1; since_save += 1
                    executor.shutdown(wait=False, cancel_futures=True)
                    futures.clear()
                    executor = ProcessPoolExecutor(max_workers=max_inflight, mp_context=ctx)
                    while task_queue and len(futures) < max_inflight:
                        submit_next()
                    last_progress = time.time()
                continue
            last_progress = time.time()
            for fut in done:
                si, ti = futures.pop(fut)
                try:
                    per_sample[si][ti] = fut.result()
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"[trial] future 异常: {e}")
                    per_sample[si][ti] = None
                remaining[si] -= 1
                if task_queue:
                    submit_next()
                if remaining[si] == 0:
                    timed, avg = finalize(si)
                    completed += 1
                    since_save += 1
                    rate = completed / max(1e-9, time.time() - t_start) * 60
                    print(f"[{len(results)}/{total}] idx={si} db={tasks[si]['db_id']} "
                          f"avg={avg:.3f} timed_out={timed} ~{rate:.1f}/min")
            if since_save >= args.save_every:
                results.sort(key=lambda r: r.get("task_idx", 0))
                save_results(args.output, results)
                since_save = 0
    finally:
        results.sort(key=lambda r: r.get("task_idx", 0))
        save_results(args.output, results)
        executor.shutdown(wait=False, cancel_futures=True)

    summarize(results)
    print(f"\n✅ 采样完成, 结果已保存: {args.output}")
    print(f"👉 下一步过滤(方差为0 + 超时):")
    print(f"   python filter_variance_zero.py --input {args.output} "
          f"--output {args.output.replace('.json', '_filtered.json')}")


def main() -> None:
    p = argparse.ArgumentParser(description="BIRD train 14B 方差采样")
    p.add_argument("--parquet", default=DEFAULT_PARQUET)
    p.add_argument("--db-dir", default=DEFAULT_DB_DIR)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--n-trials", type=int, default=N_TRIALS)
    p.add_argument("--num-gpus", type=int, default=NUM_GPUS)
    p.add_argument("--per-gpu", type=int, default=PARALLEL_PER_GPU,
                   help="每个 vLLM 端点并发的 rollout 数, 总并发=num_gpus*per_gpu")
    p.add_argument("--traj-timeout", type=int, default=TRAJ_TIMEOUT,
                   help="单条 rollout(ReAct 轨迹) 硬超时秒数")
    p.add_argument("--save-every", type=int, default=SAVE_EVERY,
                   help="每完成多少样本落盘一次(流式)")
    args = p.parse_args()
    process(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ 已中断, 结果已安全保存, 可重跑续跑。")
