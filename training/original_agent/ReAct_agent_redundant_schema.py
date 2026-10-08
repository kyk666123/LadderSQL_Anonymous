"""ReAct SQL agent variant for HIGH-RECALL, REDUNDANT schema training.

与 original_agent.ReAct_Agent 的唯一区别：
在 training_rollout / validation_rollout 中，把喂给 Agent 的 prompt 模板从
    REACT_SQL_BIRD_PROMPT / REACT_SQL_PROMPT
换成显式告知「schema 含冗余、需自行甄别」的
    REACT_SQL_BIRD_REDUNDANT_PROMPT / REACT_SQL_REDUNDANT_PROMPT

Agent 执行流程（ReAct 循环、SQL 执行、reward、轨迹日志）完全复用父类，未做任何改动。
本类子类化父 LitAgent 并只覆盖两个 rollout 方法，无全局副作用、线程安全。

用法：训练/评测脚本把
    from original_agent.ReAct_Agent import LitAgent
改为
    from original_agent.ReAct_agent_redundant_schema import LitAgent
其余构造参数与调用方式保持不变。
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import textwrap
import time
from pathlib import Path
from typing import Any, Dict

import agentlightning as agl
from typing import cast

from original_agent.ReAct_Agent import Agent, LitAgent as _BaseLitAgent
from original_agent.prompt import SUMMARIZE_PROMPT
from original_agent.prompt_redundant_schema import (
    REACT_SQL_BIRD_REDUNDANT_PROMPT,
    REACT_SQL_REDUNDANT_PROMPT,
)
from reward_func.reward import binary_reward, binary_reward_bird
from reward_func.composite_reward_dynamic import composite_reward_dynamic
from reward_func.composite_reward_dynamic_bird import composite_reward_dynamic_bird


logger = logging.getLogger(__name__)


class LitAgent(_BaseLitAgent):
    """LitAgent variant that feeds a high-recall / redundant schema prompt.

    Inherits everything (``__init__``, ``get_table_info``, trajectory logging,
    etc.) from the original ``LitAgent``; only the two rollout methods are
    overridden to swap the prompt template. The overridden bodies are identical
    to the parent's except for the single line that selects ``prompt_template``.
    """

    def training_rollout(
        self,
        task: Dict[str, Any],
        resources: agl.NamedResources,
        rollout: agl.Rollout,
    ) -> float | None:
        start_time = time.time()

        question = task["question"]
        db_id = task["db_id"]
        ground_truth = task["query"]
        evidence = task.get("evidence", "")  # BIRD特有字段

        llm: agl.LLM = cast(agl.LLM, resources["main_llm"])
        rollout_id = rollout.rollout_id

        if self.debug:
            print(f"[DEBUG][{rollout_id}] >>> training_rollout START | db_id={db_id} | question={question[:60]}...")

        # 根据dataset选择数据库路径（本地磁盘，不复制，直接读取）
        if self.dataset == "bird":
            db_path = Path("/root/local_bird_db/train_databases") / f"{db_id}/{db_id}.sqlite"
        else:
            db_path = Path(self.spider_dir) / f"database/{db_id}/{db_id}.sqlite"

        if not db_path.exists():
            logger.error(f"Database {db_path} does not exist. Skipping.")
            return None

        if self.debug:
            print(f"[DEBUG][{rollout_id}] Step 1: DB at {db_path} (direct read, no copy)")

        logger.info(f"[Rollout {rollout_id}] Question: {question}")
        logger.info(f"[Rollout {rollout_id}] Ground Truth: {ground_truth}")

        t0 = time.time()
        db_schema = self.get_table_info(
            is_extend=False,
            mode=rollout.mode,  # type: ignore
            db_path="sqlite:///" + db_path.as_posix(),
            rollout_db_id=db_id,
            db_schema="",
            rollout_question=question,
            light_schema={}
        )
        t1 = time.time()
        logger.info(f"[Rollout {rollout_id}] [TIMING] get_table_info: {t1-t0:.3f}s")
        if self.debug:
            print(f"[DEBUG][{rollout_id}] Step 3: get_table_info done ({t1-t0:.3f}s) | schema_len={len(db_schema)}")

        # Run the ReAct agent —— 仅此处的 prompt 模板与父类不同（含冗余 schema 甄别提示）
        prompt_template = REACT_SQL_BIRD_REDUNDANT_PROMPT if self.dataset == "bird" else REACT_SQL_REDUNDANT_PROMPT
        endpoint_url = llm.get_base_url(rollout.rollout_id, rollout.attempt.attempt_id)  # type: ignore
        if self.debug:
            print(f"[DEBUG][{rollout_id}] Step 4: Building agent graph | endpoint={endpoint_url} | model={llm.model}")

        agent_instance = Agent(
            "sqlite:///" + db_path.as_posix(),
            db_schema=db_schema,
            max_turns=self.max_turns,
            execution_truncate=self.execution_truncate,
            endpoint=endpoint_url,
            verl_replacement={"model": llm.model, **llm.sampling_parameters},
            prompt_template=prompt_template,
            evidence=evidence,
        )
        agent = agent_instance.graph()
        t2 = time.time()
        logger.info(f"[Rollout {rollout_id}] [TIMING] agent_init+graph: {t2-t1:.3f}s")
        if self.debug:
            print(f"[DEBUG][{rollout_id}] Step 5: Agent graph built ({t2-t1:.3f}s) | Starting invoke...")

        try:
            handler = self.tracer.get_langchain_handler()
            result = agent.invoke(  # type: ignore
                {"question": question},  # type: ignore
                {"callbacks": [handler] if handler else [], "recursion_limit": 100},
            )
        except Exception as e:
            elapsed = time.time() - start_time
            # 打印完整轨迹到 stdout（会出现在 wandb output.log）
            import traceback as tb_mod
            turn_history_str = ""
            for th in agent_instance._turn_history:
                turn_history_str += (
                    f"    Turn {th['turn']}: input_chars={th['input_chars']} "
                    f"est_tokens={th['estimated_tokens']} "
                    f"status={th['status']}\n"
                )
                if th['status'] == 'ok':
                    turn_history_str += f"      output_chars={th['output_chars']} preview={th['output_preview'][:150]}\n"
                else:
                    turn_history_str += f"      output={th.get('output', '')}\n"
            if not turn_history_str:
                turn_history_str = "    (no turns completed)\n"
            print(
                f"\n{'='*60}\n"
                f"[ROLLOUT FAILED] {rollout_id}\n"
                f"  db_id: {db_id}\n"
                f"  question: {question[:120]}\n"
                f"  elapsed: {elapsed:.1f}s\n"
                f"  error: {type(e).__name__}: {e}\n"
                f"  turn_history ({len(agent_instance._turn_history)} turns completed):\n"
                f"{turn_history_str}"
                f"  traceback:\n{textwrap.indent(tb_mod.format_exc(), '    ')}\n"
                f"{'='*60}"
            )
            logger.error(
                f"[Rollout {rollout_id}] agent.invoke FAILED after {elapsed:.1f}s | "
                f"{type(e).__name__}: {e}"
            )
            # 将失败的 rollout 也记录到轨迹文件，便于事后分析
            self._log_failed_trajectory(
                rollout_id, question, db_id, ground_truth,
                agent_instance._turn_history, type(e).__name__, str(e), elapsed
            )
            return
        t3 = time.time()
        logger.info(f"[Rollout {rollout_id}] [TIMING] agent_invoke: {t3-t2:.3f}s")
        if self.debug:
            print(f"[DEBUG][{rollout_id}] Step 6: agent.invoke done ({t3-t2:.3f}s) | query={result.get('query', '')[:80]}")

        # Reward calculation（直接读取本地DB，不复制）
        if self.reward_mode == "binary":
            _reward_fn = binary_reward_bird if self.dataset == "bird" else binary_reward
            reward = _reward_fn(
                result["query"], ground_truth, db_path.as_posix(), raise_on_error=False,
            )
        elif self.dataset == "bird":
            reward = composite_reward_dynamic_bird(
                result["query"], ground_truth, db_path.as_posix(), raise_on_error=False,
            )
        else:
            reward = composite_reward_dynamic(
                result["query"], ground_truth, db_path.as_posix(), raise_on_error=False,
            )

        total_time = time.time() - start_time

        logger.info(f"[Rollout {rollout_id}] [TIMING] reward: {time.time()-t3:.3f}s")
        logger.info(f"[Rollout {rollout_id}] [TIMING] total: {total_time:.3f}s")
        logger.info(f"[Rollout {rollout_id}] Final reward: {reward}")
        if self.debug:
            print(f"[DEBUG][{rollout_id}] Step 7: reward={reward} | total={total_time:.2f}s | DONE")

        # Persist the full ReAct trajectory to the dedicated trajectory file
        self._log_trajectory(rollout_id, "train", question, db_id, ground_truth, result, reward)

        # Emit tool call stats as annotation span for daemon-side aggregation
        _sql_exec = result.get("sql_exec_count", 0)
        _sql_err = result.get("sql_error_count", 0)
        try:
            agl.emit_annotation({
                "tool_stats.sql_execute.calls": _sql_exec,
                "tool_stats.sql_execute.errors": _sql_err,
                "tool_stats.sql_execute.successes": _sql_exec - _sql_err,
            })
        except Exception as exc:
            logger.warning(f"[Rollout {rollout_id}] Failed to emit tool stats annotation: {exc}")

        return reward

    def validation_rollout(
        self,
        task: Dict[str, Any],
        resources: agl.NamedResources,
        rollout: agl.Rollout,
    ) -> float | None:
        question = task["question"]
        db_id = task["db_id"]
        ground_truth = task.get("query") or task.get("SQL", "")
        # BIRD特有字段
        evidence = task.get("evidence", "")

        llm: agl.LLM = cast(agl.LLM, resources["main_llm"])

        # 根据dataset选择数据库路径（本地磁盘，直接读取，不复制）
        if self.dataset == "bird":
            db_path = Path("/root/local_bird_db/dev_databases") / f"{db_id}/{db_id}.sqlite"
        else:
            db_path = Path(self.spider_dir) / f"test_database/{db_id}/{db_id}.sqlite"

        if not db_path.exists():
            logger.error(f"Database {db_path} does not exist. Skipping.")
            return None

        rollout_id = rollout.rollout_id

        logger.info(f"[Rollout {rollout_id}] Question: {question}")
        logger.info(f"[Rollout {rollout_id}] Ground Truth: {ground_truth}")

        def single_sampling(endpoint_url: str | None = None) -> tuple[dict | None, str | None]:
            """单次ReAct推理，返回 (result, db_schema) 或 (None, None)。"""
            try:
                db_schema = self.get_table_info(
                    is_extend=False,
                    mode=rollout.mode,  # type: ignore
                    db_path="sqlite:///" + db_path.as_posix(),
                    rollout_db_id=db_id,
                    db_schema="",
                    rollout_question=question,
                    light_schema={}
                )

                # 负载均衡：优先使用传入的endpoint_url，否则fallback到llm.get_base_url
                _ep = endpoint_url
                if _ep is None:
                    _ep = llm.get_base_url(rollout.rollout_id, rollout.attempt.attempt_id)  # type: ignore

                # 仅此处的 prompt 模板与父类不同（含冗余 schema 甄别提示）
                prompt_template = REACT_SQL_BIRD_REDUNDANT_PROMPT if self.dataset == "bird" else REACT_SQL_REDUNDANT_PROMPT
                agent = Agent(
                    "sqlite:///" + db_path.as_posix(),
                    db_schema=db_schema,
                    max_turns=self.max_turns,
                    execution_truncate=self.execution_truncate,
                    endpoint=_ep,
                    verl_replacement=(
                        {
                            "model": llm.model,
                            # 强制val使用确定性解码（0.0），不看 llm.sampling_parameters
                            # 原因：框架 daemon 会把 train temperature（1.1）注入 sampling_parameters，
                            # 导致 dict.get("temperature", val_temperature) 始终返回1.1，val_temperature成为死代码。
                            "temperature": self.val_temperature,
                        }
                    ),
                    prompt_template=prompt_template,
                    evidence=evidence,
                ).graph()

                handler = self.tracer.get_langchain_handler()
                result = agent.invoke(  # type: ignore
                    {"question": question},  # type: ignore
                    {"callbacks": [handler] if handler else [], "recursion_limit": 100},
                )
                return result, db_schema
            except Exception as e:
                logger.exception(f"[Rollout {rollout_id}] Error during agent invocation: {e}")
                return None, None

        query = ""
        result: dict = {}

        if self.val_concurrent <= 1:
            # 单次推理
            t_start = time.time()
            result, _ = single_sampling()
            if result is not None:
                query = result["query"]
            else:
                query = ""
                result = {}
            logger.info(f"[Rollout {rollout_id}] [TIMING] single_sampling: {time.time()-t_start:.3f}s")
        else:
            # 并行推理（多路采样）
            # 负载均衡：如果有多个vLLM endpoint，将并发请求分散到不同实例
            endpoints = self.llm_endpoints if self.llm_endpoints else [llm.get_base_url(rollout.rollout_id, rollout.attempt.attempt_id)]  # type: ignore
            t_start = time.time()
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.val_concurrent) as executor:
                futures = [
                    executor.submit(single_sampling, endpoints[i % len(endpoints)])
                    for i in range(self.val_concurrent)
                ]
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())

            if self.eval_mode == "pass_at_n":
                # ===== Pass@N 模式 =====
                # 逐个验证每条SQL，只要有一条正确，reward即为1
                pass_at_n_rewards = []
                valid_results = [(r, s) for r, s in results if r is not None]
                _reward_fn = binary_reward_bird if self.dataset == "bird" else binary_reward
                for r, _ in valid_results:
                    individual_reward = _reward_fn(
                        r["query"], ground_truth, db_path.as_posix(), raise_on_error=False
                    )
                    pass_at_n_rewards.append(individual_reward)

                # 任意一条正确 → 1，全错 → 0
                reward = 1.0 if any(r == 1.0 for r in pass_at_n_rewards) else 0.0
                query = valid_results[0][0]["query"] if valid_results else ""
                result = valid_results[0][0] if valid_results else {}

                # 保存所有N条候选SQL及其individual reward（含完整轨迹）
                pass_at_n_candidates = []
                for idx, (r, _) in enumerate(valid_results):
                    pass_at_n_candidates.append({
                        "index": idx,
                        "query": r.get("query", ""),
                        "reward": pass_at_n_rewards[idx] if idx < len(pass_at_n_rewards) else None,
                        "num_turns": r.get("num_turns", 0),
                    })
                # 附加到 result 中，供 export_rollouts 导出
                result["pass_at_n_candidates"] = pass_at_n_candidates
                result["pass_at_n_final_reward"] = reward

                logger.info(
                    f"[Rollout {rollout_id}] Pass@{self.val_concurrent}: "
                    f"individual_rewards={pass_at_n_rewards} → final={reward}"
                )
                logger.info(f"[Rollout {rollout_id}] [TIMING] pass_at_n: {time.time()-t_start:.3f}s")

                # Persist ALL candidates' full trajectories (每条候选都保存完整轨迹)
                self._log_pass_at_n_trajectories(
                    rollout_id, question, db_id, ground_truth,
                    valid_results, pass_at_n_rewards, reward
                )

                _sql_exec = result.get("sql_exec_count", 0)
                _sql_err = result.get("sql_error_count", 0)
                try:
                    agl.emit_annotation({
                        "tool_stats.sql_execute.calls": _sql_exec,
                        "tool_stats.sql_execute.errors": _sql_err,
                        "tool_stats.sql_execute.successes": _sql_exec - _sql_err,
                        "pass_at_n.candidates": json.dumps(pass_at_n_candidates),
                        "pass_at_n.n": self.val_concurrent,
                        "pass_at_n.reward": reward,
                    })
                except Exception as exc:
                    logger.warning(f"[Rollout {rollout_id}] Failed to emit tool stats annotation: {exc}")
                return reward

            else:
                # ===== Best of N 模式（原逻辑）=====
                # 按执行结果分组，减少上下文长度
                execution_groups: dict[str, list[tuple[dict, str]]] = {}
                for r, db_schema in results:
                    if r is None:
                        continue
                    # 执行结果截断到2048字符，与其他地方对齐
                    exec_str = str(r.get("execution", ""))
                    if len(exec_str) > self.execution_truncate:
                        exec_str = exec_str[:self.execution_truncate] + "\n... (truncated)"
                    if exec_str not in execution_groups:
                        execution_groups[exec_str] = []
                    execution_groups[exec_str].append((r, db_schema))

                candidate_blocks = ""
                group_idx = 1
                for exec_key, group in execution_groups.items():
                    # 每组只保留一个执行结果样例
                    first_result, first_schema = group[0]
                    sql_list = [r["query"] for r, _ in group]
                    candidate_blocks += f"""
                    Candidate Group {group_idx}:
                    Schema: {first_schema}
                    SQL(s) producing this result ({len(sql_list)} total):
                    {chr(10).join(f'  - {sql}' for sql in sql_list)}
                    Result: {exec_key}\n\n
                    """
                    group_idx += 1

                if candidate_blocks:
                    prompt = SUMMARIZE_PROMPT.invoke(
                        {
                            "question": question,
                            "candidate_blocks": candidate_blocks
                        }
                    )
                    query = self.summarizer_llm.invoke(prompt).content
                else:
                    query = ""
                    logger.warning(f"[Rollout {rollout_id}] All concurrent samples failed, no candidates for summarization")

                # 用第一组的第一条result做trajectory log
                if execution_groups:
                    first_group = next(iter(execution_groups.values()))
                    result = first_group[0][0]
                else:
                    result = {}

                logger.info(f"[Rollout {rollout_id}] [TIMING] concurrent_sampling+summarize: {time.time()-t_start:.3f}s")

        logger.info(f"[Rollout {rollout_id}] Generated Query: {query}")

        # Log finish_reason for each LLM call
        if result:
            finish_reasons = [
                getattr(msg, 'response_metadata', {}).get('finish_reason')
                for msg in result.get("messages", [])
                if getattr(msg, 'response_metadata', {}).get('finish_reason') is not None
            ]
            truncated_count = sum(1 for fr in finish_reasons if fr == "length")
            logger.info(f"[Rollout {rollout_id}] finish_reasons: {finish_reasons}")
            if truncated_count > 0:
                logger.warning(
                    f"[Rollout {rollout_id}] {truncated_count}/{len(finish_reasons)} "
                    f"LLM calls truncated (finish_reason=length)"
                )

        t_start = time.time()

        # 直接读取DB计算reward，不复制（SELECT-only安全）
        _reward_fn = binary_reward_bird if self.dataset == "bird" else binary_reward
        reward = _reward_fn(query, ground_truth, db_path.as_posix(), raise_on_error=False)

        logger.info("[Rollout %s] Reward: %s", rollout_id, reward)
        logger.info("[Rollout %s] Time taken for reward calculation: %.2f seconds", rollout_id, time.time() - t_start)

        # Persist the full ReAct trajectory to the dedicated trajectory file
        self._log_trajectory(rollout_id, "val", question, db_id, ground_truth, result, reward)

        # Emit tool call stats as annotation span for daemon-side aggregation
        _sql_exec = result.get("sql_exec_count", 0)
        _sql_err = result.get("sql_error_count", 0)
        try:
            agl.emit_annotation({
                "tool_stats.sql_execute.calls": _sql_exec,
                "tool_stats.sql_execute.errors": _sql_err,
                "tool_stats.sql_execute.successes": _sql_exec - _sql_err,
            })
        except Exception as exc:
            logger.warning(f"[Rollout {rollout_id}] Failed to emit tool stats annotation: {exc}")

        return reward
