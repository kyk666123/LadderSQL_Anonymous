#!/usr/bin/env python3
"""BIRD train —— 基于 **redundant-schema agent** 的 16 条轨迹方差采样。

与 sample_bird_14b.py(golden schema 版) 的**唯一区别**是「prompt + schema 来源」：

  - prompt : original_agent/prompt_redundant_schema.REACT_SQL_BIRD_REDUNDANT_PROMPT
             (高召回冗余 schema 甄别提示，与 ReAct_agent_redundant_schema.py 训练时一致)
  - schema : apex-sql-schema-link-results/bird_train/schema_cache_train_rendered.json
             预渲染的候选 schema 文本，key = f"{db_id}|||{question}"
             (与 ReAct_Agent.LitAgent.get_table_info 的 truncate=="cache" 分支完全一致)

其余(ReAct 循环、SQL 执行、binary reward、流式续跑、进程池硬超时) 全部复用原逻辑。

支持三种模型配置(由环境变量控制，脚本层设定)：
  - qwen2.5-coder-14b        : SERVED_MODEL_NAME=qwen2.5-coder  (无 thinking)
  - qwen3-14b (开启思考)      : SERVED_MODEL_NAME=qwen3  ENABLE_THINKING=true   (vLLM 需 --reasoning-parser qwen3)
  - qwen3-14b (不开启思考)     : SERVED_MODEL_NAME=qwen3  ENABLE_THINKING=false

环境变量:
  SERVED_MODEL_NAME  vLLM --served-model-name (默认 qwen2.5-coder)
  ENABLE_THINKING    true/false/(unset)  -> 注入 extra_body.chat_template_kwargs.enable_thinking
  THINKING_BUDGET    思考 token 预算(仅 thinking=true 生效) -> extra_body.thinking_token_budget
                     (vLLM 达预算后强制输出 </think> 收尾; 需服务端 --reasoning-parser qwen3)
  MAX_TOKENS         单次生成上限 (默认 2048; thinking 建议 = 预算 + 答案余量)
  TEMPERATURE        采样温度 (默认 1.1)
  LLM_TIMEOUT        单次 LLM 请求超时秒 (默认 45; thinking 建议 120)
  SCHEMA_CACHE       预渲染 schema 缓存 json 路径

用法:
    python sample_bird_redundant.py --parquet ... --db-dir ... --output ... --n-trials 16 --num-gpus 8
断点续跑: 重复执行会跳过已完成样本, 并自动重跑标记为 timed_out 的样本。
"""
from __future__ import annotations

import os
import sys
import json
import time
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

from original_agent.ReAct_Agent import Agent  # type: ignore
from original_agent.prompt_redundant_schema import REACT_SQL_BIRD_REDUNDANT_PROMPT  # type: ignore
from reward_func.reward import binary_reward_bird  # type: ignore

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("sample_bird_redundant")
logger.setLevel(logging.INFO)

# ============================ 默认配置 ============================
DEFAULT_PARQUET = "/path/to/nl2sql_dataset/bird/bird_clean_data/train.parquet"
DEFAULT_DB_DIR = "/root/local_bird_db/train_databases"   # {db_id}/{db_id}.sqlite
DEFAULT_OUTPUT = "/path/to/sampling_outputs/redundant_bird_train_sampling.json"
DEFAULT_SCHEMA_CACHE = (
    "/path/to/schema_link_results/bird_train/"
    "schema_cache_train_rendered.json"
)

N_TRIALS = 16               # 每个样本采样次数
NUM_GPUS = 8                # vLLM 实例数(每卡一个)
BASE_PORT = 9000            # vLLM 起始端口 -> 9000..9007
MAX_TURNS = 8               # 必须与训练 --max-turns 对齐
EXECUTION_TRUNCATE = 2048
RECURSION_LIMIT = 100
SAVE_EVERY = 16

# ---- 由环境变量控制的运行时配置(脚本层设定，区分三种模型) ----
SERVED_MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", "qwen2.5-coder")
TEMPERATURE = float(os.environ.get("TEMPERATURE", "1.1"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "2048"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "16384"))
# ReAct 提前终止的 prompt token 预算 = 上下文窗口 - 单次生成上限。
# 覆盖 ReAct_Agent 默认的 12288(旧训练配置), 与本次采样的 max_model_len/max_tokens 对齐。
os.environ.setdefault("REACT_PROMPT_TOKEN_BUDGET", str(max(1024, MAX_MODEL_LEN - MAX_TOKENS)))
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "45"))
TRAJ_TIMEOUT = int(os.environ.get("TRAJ_TIMEOUT", "120"))
PARALLEL_PER_GPU = int(os.environ.get("PER_GPU", "10"))
_ENABLE_THINKING_ENV = os.environ.get("ENABLE_THINKING")  # "true"/"false"/None
_THINKING_BUDGET_ENV = os.environ.get("THINKING_BUDGET")  # "2048"/None (仅 thinking=true 生效)

# ---- 给 Agent 内部 LLM 调用注入: 请求级超时 + max_tokens 覆盖 + thinking 开关 ----
# 原 Agent.__init__ 里 init_chat_model(..., max_tokens=2048)，此处统一覆盖，
# 并按 ENABLE_THINKING 注入 extra_body.chat_template_kwargs.enable_thinking (Qwen3)，
# 若开启思考且设了 THINKING_BUDGET，再注入顶层采样参数 thinking_token_budget。
import original_agent.ReAct_Agent as _react_mod  # type: ignore  # noqa: E402
_ORIG_INIT_CHAT_MODEL = _react_mod.init_chat_model


def _init_chat_model_patched(*a, **k):
    k.setdefault("timeout", LLM_TIMEOUT)
    k["max_retries"] = 1  # 原为 3, 连接挂起时会把等待放大 3 倍
    k["max_tokens"] = MAX_TOKENS
    if _ENABLE_THINKING_ENV is not None:
        thinking_on = _ENABLE_THINKING_ENV.lower() == "true"
        extra_body = {"chat_template_kwargs": {"enable_thinking": thinking_on}}
        if thinking_on and _THINKING_BUDGET_ENV:
            # thinking_token_budget 是请求顶层采样参数(与 chat_template_kwargs 平级)
            extra_body["thinking_token_budget"] = int(_THINKING_BUDGET_ENV)
        k["extra_body"] = extra_body
    return _ORIG_INIT_CHAT_MODEL(*a, **k)


_react_mod.init_chat_model = _init_chat_model_patched


# ============================ redundant schema 缓存 ============================
def _make_key(db_id: str, question: str) -> str:
    # 与 ReAct_Agent.get_table_info 的 cache 分支完全一致: 原始 question, 3 个竖线
    return f"{db_id}|||{question}"


def _load_schema_cache(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(
            f"redundant schema 缓存不存在: {path} "
            f"(应为 build_rendered_cache_train.py 的产物, key=db_id|||question)"
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"[schema] 已加载 {len(data)} 条 redundant schema 缓存: {path}")
    return data


_SCHEMA_CACHE_PATH = Path(os.environ.get("SCHEMA_CACHE", DEFAULT_SCHEMA_CACHE))
# 模块级加载(fork 前), 子进程继承
_SCHEMA_MAP: Dict[str, str] = _load_schema_cache(_SCHEMA_CACHE_PATH)
# 归一化空白后的兜底索引(应对 parquet 与 APEX question 的极少数空白差异)
_SCHEMA_MAP_NORM: Dict[str, str] = {
    f"{k.split('|||', 1)[0]}|||{' '.join(k.split('|||', 1)[1].split())}": v
    for k, v in _SCHEMA_MAP.items() if "|||" in k
}


def build_endpoints(num_gpus: int) -> List[str]:
    return [f"http://localhost:{BASE_PORT + i}/v1" for i in range(num_gpus)]


def get_db_path(db_dir: str, db_id: str) -> str:
    return os.path.join(db_dir, db_id, f"{db_id}.sqlite")


def get_redundant_schema(db_id: str, question: str) -> str:
    """取预渲染的 redundant 候选 schema 文本(按 db_id+question)，不截断。"""
    s = _SCHEMA_MAP.get(_make_key(db_id, question))
    if s is None:
        s = _SCHEMA_MAP_NORM.get(f"{db_id}|||{' '.join(question.strip().split())}")
    if s is None:
        logger.warning(f"[schema] redundant schema 未命中: {db_id} | {question[:80]}")
        return "No candidate schema available."
    return s


def run_one_trial(task: Dict[str, Any], endpoint: str, db_dir: str) -> Optional[Dict[str, Any]]:
    """单条 rollout: 跑 redundant-schema agent 得到 SQL, 再算 0/1 reward。失败返回 None。"""
    db_id = task["db_id"]
    question = task["question"]
    ground_truth = task["query"]
    evidence = task.get("evidence", "") or ""

    db_file = get_db_path(db_dir, db_id)
    if not os.path.exists(db_file):
        logger.error(f"[trial] 数据库不存在: {db_file}")
        return None

    db_uri = f"sqlite:///{db_file}"
    db_schema = get_redundant_schema(db_id, question)   # redundant 候选 schema

    try:
        agent_obj = Agent(
            db_path=db_uri,
            db_schema=db_schema,
            max_turns=MAX_TURNS,
            endpoint=endpoint,
            verl_replacement={"model": SERVED_MODEL_NAME, "temperature": TEMPERATURE},
            execution_truncate=EXECUTION_TRUNCATE,
            prompt_template=REACT_SQL_BIRD_REDUNDANT_PROMPT,
            evidence=evidence,
        )
        result = agent_obj.graph().invoke({"question": question}, {"recursion_limit": RECURSION_LIMIT})
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[trial] db_id={db_id} agent 执行失败: {e}")
        return None

    pred_sql = result.get("query", "") if isinstance(result, dict) else ""
    try:
        reward = binary_reward_bird(pred_sql, ground_truth, db_file, raise_on_error=False)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[trial] db_id={db_id} reward 计算失败: {e}")
        return None

    # 汇总本条轨迹的思考时间/长度(来自 Agent 每轮 _turn_history; 非思考模型 reasoning=0)
    hist = getattr(agent_obj, "_turn_history", []) or []
    llm_time = sum(float(h.get("llm_time") or 0.0) for h in hist)
    gen_tokens = sum(int(h.get("gen_tokens") or 0) for h in hist)
    rsn_chars = sum(int(h.get("reasoning_chars") or 0) for h in hist)
    _rtoks = [h.get("reasoning_tokens") for h in hist if h.get("reasoning_tokens") is not None]
    rsn_tokens = int(sum(_rtoks)) if _rtoks else None
    return {
        "reward": float(reward),
        "turns": len(hist),
        "llm_time": round(llm_time, 3),
        "gen_tokens": gen_tokens,
        "reasoning_tokens": rsn_tokens,
        "reasoning_chars": rsn_chars,
    }


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
    print(f"[INFO] 模型: served={SERVED_MODEL_NAME} thinking={_ENABLE_THINKING_ENV} "
          f"think_budget={_THINKING_BUDGET_ENV} max_tokens={MAX_TOKENS} temp={TEMPERATURE} "
          f"llm_timeout={LLM_TIMEOUT}s")
    print(f"[INFO] max_model_len={MAX_MODEL_LEN} 早停prompt预算(REACT_PROMPT_TOKEN_BUDGET)="
          f"{os.environ.get('REACT_PROMPT_TOKEN_BUDGET')}")
    print(f"[INFO] schema 缓存: {_SCHEMA_CACHE_PATH} ({len(_SCHEMA_MAP)} 条)")
    print(f"[INFO] 样本总数: {total}; 每样本采样 {args.n_trials} 次 -> {len(endpoints)} 端点; "
          f"并发 rollout: {max_inflight}")
    print(f"[INFO] max_turns={MAX_TURNS} 单轨迹硬超时={args.traj_timeout}s 结果: {args.output}")

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
        rl = per_sample.pop(si)   # list[dict|None]
        task = tasks[si]
        reward_list = [(d["reward"] if isinstance(d, dict) else None) for d in rl]
        timed = any(d is None for d in rl)
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
            "query": task["query"],
            "evidence": task.get("evidence", ""),
            "reward_list": reward_list,
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
                if time.time() - last_progress > args.traj_timeout + 120:
                    logger.error(f"[stuck] {len(futures)} 个 worker 长时间无响应, 标记超时并重建进程池。")
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
    p = argparse.ArgumentParser(description="BIRD train redundant-schema 方差采样")
    p.add_argument("--parquet", default=DEFAULT_PARQUET)
    p.add_argument("--db-dir", default=DEFAULT_DB_DIR)
    p.add_argument("--output", default=os.environ.get("OUTPUT", DEFAULT_OUTPUT))
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
