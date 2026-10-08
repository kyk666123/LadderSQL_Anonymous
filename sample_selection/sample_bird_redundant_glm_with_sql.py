#!/usr/bin/env python3
"""BIRD **redundant-schema agent** 采样 —— 用 GLM-5.2(OpenAI 兼容 API)代替本地 vLLM 权重。

本脚本是 sample_bird_redundant_with_sql.py 的「GLM API」变体, 专为
「用 glm-5.2 在 BIRD old dev 上贪心采样, 保存实际生成 SQL + 0/1 奖励 + 完整轨迹」而生。

与 sample_bird_redundant_with_sql.py 的**核心区别**:
  1. LLM 不再指向本地 vLLM(localhost:9000..), 而是通过 langchain init_chat_model
     指向 GLM 的 OpenAI 兼容端点(ALICLOUD_BASE_URL / ALICLOUD_API_KEY, model=glm-5.2),
     与 ReAct_Agent.py 中 schema_filter/summarizer 的调用方式完全一致;
  2. ★ 鲁棒重试机制(应对 GLM 限流 429/5xx/超时), 确保「最后所有样本都采样成功」:
       - LLM 层: init_chat_model(max_retries=GLM_MAX_RETRIES) 由 openai 客户端做指数退避重试;
       - 轨迹层: 单条 rollout 失败(超时/异常/None)后**自动重新入队**, 带指数退避,
                 直到成功或达到 MAX_TRIAL_ATTEMPTS 上限(默认极大, 实际=不放弃);
       - 无进度看门狗: 卡死时重建进程池并把在途任务**重新入队**(而非直接判负)。
  3. 并发不再按 GPU 数, 而是由 --concurrency / CONCURRENCY 控制(默认较小以避免触发限流)。

prompt / schema / ReAct 循环 / binary_reward / 早停预算 / 流式续跑 / 进程池硬超时
其余部分与 sample_bird_redundant_with_sql.py 保持一致, 保证结果可比。

环境变量:
  ALICLOUD_BASE_URL / ALICLOUD_API_KEY   GLM OpenAI 兼容端点与密钥(必需)
  SERVED_MODEL_NAME(=GLM 模型名, 默认 glm-5.2) / TEMPERATURE / MAX_TOKENS / MAX_MODEL_LEN
  LLM_TIMEOUT(单次请求超时秒) / GLM_MAX_RETRIES(单次请求内部重试次数)
  SCHEMA_CACHE / SAVE_TRAJECTORY / MAX_TURNS_AGENT / TRAJ_TIMEOUT
  CONCURRENCY(并发 rollout 数) / MAX_TRIAL_ATTEMPTS / RETRY_BACKOFF_BASE / RETRY_BACKOFF_CAP

用法:
    python sample_bird_redundant_glm_with_sql.py --parquet ... --db-dir ... --output ... \
        --n-trials 1 --concurrency 8 --traj-timeout 600
断点续跑: 重复执行会跳过已完成样本(且不会残留 None: 失败样本会被持续重试直到成功)。
"""
from __future__ import annotations

import os
import sys
import json
import time
import random
import argparse
import logging
import signal
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from typing import Any, Dict, List, Optional

import pandas as pd

# ---- 让 original_agent / reward_func 可被导入(指向 agent/ 目录) ----
_AGENT_DIR = Path(__file__).resolve().parent.parent / "training"  # provides original_agent + reward_func
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))

# 行数版观测截断: REACT_EXEC_MAX_ROWS>0 时用 RowTruncAgent(按整行截断), 否则用基础 Agent
if int(os.environ.get("REACT_EXEC_MAX_ROWS", "0")) > 0:
    from original_agent.ReAct_Agent_rowtrunc import RowTruncAgent as Agent  # type: ignore
else:
    from original_agent.ReAct_Agent import Agent  # type: ignore
from original_agent.prompt_redundant_schema import REACT_SQL_BIRD_REDUNDANT_PROMPT  # type: ignore
from reward_func.reward import binary_reward_bird  # type: ignore

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("sample_bird_redundant_glm_with_sql")
logger.setLevel(logging.INFO)

# ============================ 默认配置 ============================
DEFAULT_PARQUET = "/path/to/nl2sql_dataset/bird/dev_20240627/dev.parquet"
DEFAULT_DB_DIR = "/root/local_bird_db/dev_databases"   # {db_id}/{db_id}.sqlite
DEFAULT_OUTPUT = "/path/to/sampling_outputs/glm52_bird_old_dev_maxturns5_greedy_traj_with_sql.json"
DEFAULT_SCHEMA_CACHE = (
    "/path/to/schema_link_results/bird_old_dev/"
    "schema_cache_old_dev_rendered.json"
)

N_TRIALS = 1                # 每个样本采样次数(贪心默认 1)
BASE_PORT = 9000            # 兼容占位(GLM API 不用端口)
MAX_TURNS = int(os.environ.get("MAX_TURNS_AGENT", "5"))
EXECUTION_TRUNCATE = int(os.environ.get("EXECUTION_TRUNCATE", "2048"))
RECURSION_LIMIT = 100
SAVE_EVERY = 16

# ---- 运行时配置(脚本层设定) ----
SERVED_MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", "glm-5.2")   # ★ GLM 模型名
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0"))             # ★ 贪心默认 0
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "3072"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "24576"))
# ReAct 提前终止的 prompt token 预算, 与 max_model_len/max_tokens 对齐
os.environ.setdefault("REACT_PROMPT_TOKEN_BUDGET", str(max(1024, MAX_MODEL_LEN - MAX_TOKENS)))
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "120"))             # ★ API 单请求超时(比 vLLM 更宽松)
TRAJ_TIMEOUT = int(os.environ.get("TRAJ_TIMEOUT", "600"))           # ★ 单轨迹硬超时(含多轮+内部重试)
_SAVE_TRAJ = os.environ.get("SAVE_TRAJECTORY", "0").lower() in ("1", "true", "yes")

# ---- GLM(OpenAI 兼容)端点与鲁棒重试参数 ----
GLM_BASE_URL = os.environ.get("ALICLOUD_BASE_URL", "").rstrip("/") + ("/" if os.environ.get("ALICLOUD_BASE_URL") else "")
GLM_API_KEY = os.environ.get("ALICLOUD_API_KEY", "")
GLM_MAX_RETRIES = int(os.environ.get("GLM_MAX_RETRIES", "6"))       # openai 客户端内部重试(指数退避, 覆盖 429/5xx/超时)
# ★ GLM 默认开思考; 需显式关闭。GLM_ENABLE_THINKING=1 时才开(默认关)。
GLM_ENABLE_THINKING = os.environ.get("GLM_ENABLE_THINKING", "0").lower() in ("1", "true", "yes")
MAX_TRIAL_ATTEMPTS = int(os.environ.get("MAX_TRIAL_ATTEMPTS", "1000"))  # 轨迹层最大重试次数(默认极大 = 不放弃)
RETRY_BACKOFF_BASE = float(os.environ.get("RETRY_BACKOFF_BASE", "3.0"))  # 轨迹层退避基数(秒)
RETRY_BACKOFF_CAP = float(os.environ.get("RETRY_BACKOFF_CAP", "60.0"))   # 轨迹层退避上限(秒)
# ★ 内存控制: fork worker 因 CPython 引用计数会逐渐破坏 COW, 每个 worker 最终趋于
#    独立占用 ~1GB+。多模型并行时需控制单模型并发(CONCURRENCY), 使
#    总 worker 数 × ~1.2GB 不超机器内存。(fork 不支持 max_tasks_per_child)

if not GLM_BASE_URL or not GLM_API_KEY:
    raise RuntimeError(
        "GLM API 未配置: 需要环境变量 ALICLOUD_BASE_URL 与 ALICLOUD_API_KEY "
        "(见 ReAct_Agent.py schema_filter 的调用方式)。"
    )

# ---- 给 Agent 内部 LLM 调用注入: 指向 GLM API + 请求级超时 + 强重试 + max_tokens ----
import original_agent.ReAct_Agent as _react_mod  # type: ignore  # noqa: E402
_ORIG_INIT_CHAT_MODEL = _react_mod.init_chat_model


def _init_chat_model_patched(*a, **k):
    """把 Agent 的 init_chat_model 调用改指到 GLM 的 OpenAI 兼容端点, 并强化重试/超时。"""
    # ★ 覆盖 base_url / api_key 到 GLM(无论 Agent 传入的 endpoint/dummy key 为何)
    k["openai_api_base"] = GLM_BASE_URL
    k["openai_api_key"] = GLM_API_KEY
    # ★ 强重试: openai 客户端对 429/5xx/超时做指数退避重试
    k["max_retries"] = GLM_MAX_RETRIES
    k.setdefault("timeout", LLM_TIMEOUT)
    k["max_tokens"] = MAX_TOKENS
    # ★ GLM 原生是 reasoning 模型(默认开思考), 显式关闭思考(与 bon_structured 判官一致)
    if GLM_ENABLE_THINKING:
        k["extra_body"] = {"enable_thinking": True}
    else:
        k["extra_body"] = {"enable_thinking": False}
    return _ORIG_INIT_CHAT_MODEL(*a, **k)


_react_mod.init_chat_model = _init_chat_model_patched


# ============================ redundant schema 缓存 ============================
def _make_key(db_id: str, question: str) -> str:
    return f"{db_id}|||{question}"


def _load_schema_cache(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(
            f"redundant schema 缓存不存在: {path} (key=db_id|||question)"
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"[schema] 已加载 {len(data)} 条 redundant schema 缓存: {path}")
    return data


_SCHEMA_CACHE_PATH = Path(os.environ.get("SCHEMA_CACHE", DEFAULT_SCHEMA_CACHE))
_SCHEMA_MAP: Dict[str, str] = _load_schema_cache(_SCHEMA_CACHE_PATH)
_SCHEMA_MAP_NORM: Dict[str, str] = {
    f"{k.split('|||', 1)[0]}|||{' '.join(k.split('|||', 1)[1].split())}": v
    for k, v in _SCHEMA_MAP.items() if "|||" in k
}


def build_endpoints(num: int) -> List[str]:
    """GLM API 只有一个端点; 返回重复列表以兼容轮询逻辑。"""
    return [GLM_BASE_URL for _ in range(max(1, num))]


def get_db_path(db_dir: str, db_id: str) -> str:
    return os.path.join(db_dir, db_id, f"{db_id}.sqlite")


def get_redundant_schema(db_id: str, question: str) -> str:
    s = _SCHEMA_MAP.get(_make_key(db_id, question))
    if s is None:
        s = _SCHEMA_MAP_NORM.get(f"{db_id}|||{' '.join(question.strip().split())}")
    if s is None:
        logger.warning(f"[schema] redundant schema 未命中: {db_id} | {question[:80]}")
        return "No candidate schema available."
    return s


def _serialize_trajectory(messages) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in messages or []:
        role = getattr(m, "type", None) or m.__class__.__name__
        content = getattr(m, "content", m)
        if not isinstance(content, str):
            content = str(content)
        out.append({"role": role, "content": content})
    return out


def run_one_trial(task: Dict[str, Any], endpoint: str, db_dir: str) -> Optional[Dict[str, Any]]:
    """单条 rollout: 跑 redundant-schema agent 得到 SQL, 再算 0/1 reward。
    返回 dict(含 pred_sql); DB 不存在返回 "PERMANENT"(不可重试); 其余失败返回 None(可重试)。"""
    db_id = task["db_id"]
    question = task["question"]
    ground_truth = task["query"]
    evidence = task.get("evidence", "") or ""

    db_file = get_db_path(db_dir, db_id)
    if not os.path.exists(db_file):
        logger.error(f"[trial] 数据库不存在(不可重试): {db_file}")
        return "PERMANENT"  # type: ignore[return-value]

    db_uri = f"sqlite:///{db_file}"
    db_schema = get_redundant_schema(db_id, question)

    try:
        agent_obj = Agent(
            db_path=db_uri,
            db_schema=db_schema,
            max_turns=MAX_TURNS,
            endpoint=endpoint,   # 会被 init_chat_model patch 覆盖为 GLM_BASE_URL
            verl_replacement={"model": SERVED_MODEL_NAME, "temperature": TEMPERATURE},
            execution_truncate=EXECUTION_TRUNCATE,
            prompt_template=REACT_SQL_BIRD_REDUNDANT_PROMPT,
            evidence=evidence,
        )
        result = agent_obj.graph().invoke({"question": question}, {"recursion_limit": RECURSION_LIMIT})
    except Exception as e:  # noqa: BLE001 (限流/网络/超时 -> 可重试)
        logger.warning(f"[trial] db_id={db_id} agent 执行失败(可重试): {type(e).__name__}: {e}")
        return None

    pred_sql = result.get("query", "") if isinstance(result, dict) else ""
    try:
        reward = binary_reward_bird(pred_sql, ground_truth, db_file, raise_on_error=False)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[trial] db_id={db_id} reward 计算失败(可重试): {e}")
        return None

    hist = getattr(agent_obj, "_turn_history", []) or []
    llm_time = sum(float(h.get("llm_time") or 0.0) for h in hist)
    gen_tokens = sum(int(h.get("gen_tokens") or 0) for h in hist)
    rsn_chars = sum(int(h.get("reasoning_chars") or 0) for h in hist)
    _rtoks = [h.get("reasoning_tokens") for h in hist if h.get("reasoning_tokens") is not None]
    rsn_tokens = int(sum(_rtoks)) if _rtoks else None
    trajectory = (_serialize_trajectory(result.get("messages", []))
                  if (_SAVE_TRAJ and isinstance(result, dict)) else [])
    return {
        "pred_sql": pred_sql,
        "trajectory": trajectory,
        "reward": float(reward),
        "turns": len(hist),
        "llm_time": round(llm_time, 3),
        "gen_tokens": gen_tokens,
        "reasoning_tokens": rsn_tokens,
        "reasoning_chars": rsn_chars,
    }


def _run_trial_task(packed):
    """子进程执行单条 rollout: 先按重试次数退避, 再用 SIGALRM 施加硬超时。"""
    task, endpoint, db_dir, traj_timeout, attempt = packed

    # ★ 轨迹层指数退避: 重试(attempt>0)时先睡一会, 缓解 GLM 限流
    if attempt > 0:
        delay = min(RETRY_BACKOFF_CAP, RETRY_BACKOFF_BASE * (2 ** (attempt - 1)))
        delay += random.uniform(0, RETRY_BACKOFF_BASE)  # 抖动, 避免同步风暴
        time.sleep(delay)

    def _on_alarm(signum, frame):
        raise TimeoutError(f"trajectory exceeded {traj_timeout}s")

    old = signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(int(traj_timeout))
    try:
        return run_one_trial(task, endpoint, db_dir)
    except BaseException as e:  # noqa: BLE001 (含 TimeoutError -> 可重试)
        logger.warning(f"[trial] db={task.get('db_id')} 超时/异常(可重试): {e}")
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
        logger.warning("检测到旧格式(无 task_idx)结果, 将从头开始(旧文件请另存)。")
        return []
    print(f"[INFO] 已加载 {len(data)} 条历史结果: {output_file}")
    return data


def save_results(output_file: str, results: List[Dict]) -> None:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    tmp = output_file + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    os.replace(tmp, output_file)


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
    print(f"  mixed(方差>0): {mixed}")
    print(f"  all_1 (全对) : {all_one}")
    print(f"  all_0 (全错) : {all_zero}")
    print(f"  timed_out    : {timed}")
    print(f"  invalid(含None): {invalid}")
    print("=" * 70)


# ============================ 主流程 ============================
def process(args) -> None:
    import multiprocessing as mp
    from collections import deque, defaultdict

    tasks = pd.read_parquet(args.parquet).to_dict(orient="records")
    # 兼容金标 SQL 列名: old dev 用 "query", clean dev 用 "SQL"。
    for _t in tasks:
        if "query" not in _t and "SQL" in _t:
            _t["query"] = _t["SQL"]
    total = len(tasks)
    max_inflight = max(1, args.concurrency)
    endpoints = build_endpoints(max_inflight)

    print(f"[INFO] ★ 后端 = GLM API | model={SERVED_MODEL_NAME} base={GLM_BASE_URL} "
          f"temp={TEMPERATURE} max_tokens={MAX_TOKENS} req_timeout={LLM_TIMEOUT}s "
          f"llm_max_retries={GLM_MAX_RETRIES} thinking={'ON' if GLM_ENABLE_THINKING else 'OFF'}")
    print(f"[INFO] 鲁棒重试: 轨迹层最大重试={MAX_TRIAL_ATTEMPTS} 退避 base={RETRY_BACKOFF_BASE}s "
          f"cap={RETRY_BACKOFF_CAP}s (失败样本持续重试直到成功)")
    print(f"[INFO] max_model_len={MAX_MODEL_LEN} 早停prompt预算(REACT_PROMPT_TOKEN_BUDGET)="
          f"{os.environ.get('REACT_PROMPT_TOKEN_BUDGET')}")
    print(f"[INFO] schema 缓存: {_SCHEMA_CACHE_PATH} ({len(_SCHEMA_MAP)} 条)")
    print(f"[INFO] 样本总数: {total}; 每样本采样 {args.n_trials} 次; 并发 rollout: {max_inflight}")
    print(f"[INFO] max_turns={MAX_TURNS} 单轨迹硬超时={args.traj_timeout}s")
    print(f"[INFO] ★ 保存实际生成 SQL + 每条 0/1 奖励 -> {args.output}")

    results = load_existing(args.output)
    # 续跑时把仍含 None(未成功)的样本视为未完成, 重新采样, 确保最终全部成功。
    done_idx = set()
    for r in results:
        if "task_idx" not in r:
            continue
        rl = r.get("reward_list", [])
        if rl and all(x is not None for x in rl) and not r.get("timed_out"):
            done_idx.add(r["task_idx"])
    # 移除未完成的历史记录(将被重采覆盖)
    results = [r for r in results if r.get("task_idx") in done_idx]
    pending = [i for i in range(total) if i not in done_idx]
    if not pending:
        print(f"[INFO] 全部 {total} 个样本已成功完成。")
        summarize(results)
        return
    print(f"[INFO] 待采样 {len(pending)} / {total} (已成功 {len(done_idx)})")

    ctx = mp.get_context("fork")
    executor = ProcessPoolExecutor(max_workers=max_inflight, mp_context=ctx)

    # 队列元素: (si, ti, attempt)
    task_queue = deque((si, ti, 0) for si in pending for ti in range(args.n_trials))
    futures = {}                       # fut -> (si, ti, attempt)
    per_sample = defaultdict(lambda: [None] * args.n_trials)
    remaining = {si: args.n_trials for si in pending}   # 每样本仍需「定案」的 trial 数
    counter = 0
    stat_retry = 0
    stat_gaveup = 0

    def submit_next():
        nonlocal counter
        si, ti, attempt = task_queue.popleft()
        ep = endpoints[counter % len(endpoints)]
        counter += 1
        fut = executor.submit(_run_trial_task, (tasks[si], ep, args.db_dir, args.traj_timeout, attempt))
        futures[fut] = (si, ti, attempt)

    def finalize(si):
        rl = per_sample.pop(si)   # list[dict|None]
        task = tasks[si]
        reward_list = [(d["reward"] if isinstance(d, dict) else None) for d in rl]
        timed = any(d is None for d in rl)
        trials = []
        for ti, d in enumerate(rl):
            if isinstance(d, dict):
                trials.append({
                    "trial": ti,
                    "pred_sql": d.get("pred_sql", ""),
                    "reward": d["reward"],
                    "turns": d["turns"],
                    "llm_time": d["llm_time"],
                    "gen_tokens": d["gen_tokens"],
                    "trajectory": d.get("trajectory", []),
                })
            else:
                trials.append({
                    "trial": ti, "pred_sql": None, "reward": None,
                    "turns": None, "llm_time": None, "gen_tokens": None, "trajectory": None,
                })
        valid_d = [d for d in rl if isinstance(d, dict)]
        think_stats = {
            "turns": [d["turns"] for d in valid_d],
            "llm_time": [d["llm_time"] for d in valid_d],
            "gen_tokens": [d["gen_tokens"] for d in valid_d],
            "reasoning_tokens": [d["reasoning_tokens"] for d in valid_d],
            "reasoning_chars": [d["reasoning_chars"] for d in valid_d],
        }
        results.append({
            "task_idx": si,
            "db_id": task["db_id"],
            "question": task["question"],
            "gold_sql": task["query"],
            "query": task["query"],
            "evidence": task.get("evidence", ""),
            "reward_list": reward_list,
            "trials": trials,
            "timed_out": timed,
            "think_stats": think_stats,
        })
        valid = [r for r in reward_list if r is not None]
        avg = (sum(valid) / len(valid)) if valid else 0.0
        return timed, avg

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
                # 无进度看门狗: 长时间无任何完成 -> 重建进程池并把在途任务重新入队(不判负)
                if time.time() - last_progress > args.traj_timeout + 120:
                    logger.error(f"[stuck] {len(futures)} 个 worker 长时间无响应, 重新入队并重建进程池。")
                    for f, (si, ti, attempt) in list(futures.items()):
                        task_queue.append((si, ti, attempt + 1))
                    executor.shutdown(wait=False, cancel_futures=True)
                    futures.clear()
                    executor = ProcessPoolExecutor(max_workers=max_inflight, mp_context=ctx)
                    while task_queue and len(futures) < max_inflight:
                        submit_next()
                    last_progress = time.time()
                continue
            last_progress = time.time()
            for fut in done:
                si, ti, attempt = futures.pop(fut)
                try:
                    res = fut.result()
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"[trial] future 异常(可重试): {e}")
                    res = None

                if isinstance(res, dict):
                    # ★ 成功: 定案
                    per_sample[si][ti] = res
                    remaining[si] -= 1
                elif res == "PERMANENT":
                    # 不可重试(DB 缺失): 直接判负定案
                    per_sample[si][ti] = None
                    remaining[si] -= 1
                    stat_gaveup += 1
                else:
                    # ★ 可重试失败: 重新入队(带退避), 直到成功或达上限
                    if attempt + 1 < MAX_TRIAL_ATTEMPTS:
                        task_queue.append((si, ti, attempt + 1))
                        stat_retry += 1
                        if stat_retry % 20 == 0:
                            logger.info(f"[retry] 累计重试 {stat_retry} 次 (队列 {len(task_queue)} 待跑)")
                    else:
                        per_sample[si][ti] = None
                        remaining[si] -= 1
                        stat_gaveup += 1
                        logger.error(f"[giveup] idx={si} trial={ti} 达最大重试 {MAX_TRIAL_ATTEMPTS}, 判负。")

                # 保持进程池满载(优先消费队列, 含刚入队的重试任务)
                if task_queue and len(futures) < max_inflight:
                    submit_next()

                if remaining[si] == 0:
                    timed, avg = finalize(si)
                    completed += 1
                    since_save += 1
                    rate = completed / max(1e-9, time.time() - t_start) * 60
                    print(f"[{len(results)}/{total}] idx={si} db={tasks[si]['db_id']} "
                          f"avg={avg:.3f} timed_out={timed} retries={stat_retry} ~{rate:.1f}/min")

            # 队列可能因重试而重新有货, 补满进程池
            while task_queue and len(futures) < max_inflight:
                submit_next()

            if since_save >= args.save_every:
                results.sort(key=lambda r: r.get("task_idx", 0))
                save_results(args.output, results)
                since_save = 0
    finally:
        results.sort(key=lambda r: r.get("task_idx", 0))
        save_results(args.output, results)
        executor.shutdown(wait=False, cancel_futures=True)

    summarize(results)
    print(f"\n✅ 采样完成 (累计重试 {stat_retry} 次, 放弃 {stat_gaveup} 条), "
          f"实际生成 SQL + 0/1 奖励已保存: {args.output}")


def main() -> None:
    p = argparse.ArgumentParser(description="BIRD redundant-schema 采样(GLM-5.2 API + 鲁棒重试)")
    p.add_argument("--parquet", default=DEFAULT_PARQUET)
    p.add_argument("--db-dir", default=DEFAULT_DB_DIR)
    p.add_argument("--output", default=os.environ.get("OUTPUT", DEFAULT_OUTPUT))
    p.add_argument("--n-trials", type=int, default=int(os.environ.get("N_TRIALS", N_TRIALS)))
    p.add_argument("--concurrency", type=int, default=int(os.environ.get("CONCURRENCY", "8")),
                   help="并发 rollout 数(GLM API 无 GPU 概念; 过大易触发限流)")
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
        print("\n⚠️ 已中断, 结果已安全保存, 可重跑续跑(仍含 None 的样本会被重采)。")
