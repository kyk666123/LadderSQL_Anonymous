import asyncio
import agentlightning as agl
import json
import argparse
import pandas as pd
import os
from agentlightning.emitter import find_final_reward
from agentlightning import TraceToMessages, TracerTraceToTriplet
from pathlib import Path



def extract_question_from_messages(messages):
    if isinstance(messages, list):
        for block in messages:
            if isinstance(block, dict) and "messages" in block:
                inner = block.get("messages") or []
                for m in inner:
                    if not isinstance(m, dict):
                        continue
                    content = m.get("content")
                    if not isinstance(content, str):
                        continue
                    idx = content.find("Question:")
                    if idx != -1:
                        # 拿到 Question: 后面的部分
                        text = content[idx + len("Question:"):].strip()
                        # 只取第一行（避免后面跟 Query: / Execution result 等）
                        text = text.splitlines()[0].strip()
                        return text

    if isinstance(messages, list):
        for m in messages:
            if not isinstance(m, dict):
                continue
            content = m.get("content")
            if not isinstance(content, str):
                continue
            idx = content.find("Question:")
            if idx != -1:
                text = content[idx + len("Question:"):].strip()
                text = text.splitlines()[0].strip()
                return text

    return None


async def export_rollouts(mode: str, output_path: Path, client: agl.LightningStoreClient, dataset: str = "spider"):
    # 根据dataset参数选择数据路径
    if dataset == "bird":
        if mode == "bird_dev":
            data_path = Path("/path/to/nl2sql_dataset/bird/dev_20240627/dev.parquet")
        else:
            raise ValueError(f"Unknown mode '{mode}' for bird dataset")
    else:
        # Spider数据集路径（保持原有逻辑）
        data_path = Path(os.environ["SPIDER_DATA_DIR"]) / f"{mode}.parquet"
    
    print(f"Export rollouts - Dataset: {dataset}, Mode: {mode}")
    print(f"Export rollouts - Data path: {data_path}")
    data = pd.read_parquet(data_path)

    question_lookup = {
        row["question"]: {"db_id": row["db_id"], "query": row["query"]}
        for _, row in data.iterrows()
    }

    messages_adapter = TraceToMessages()
    triplets_adapter = TracerTraceToTriplet()

    try:
        rollouts = await client.query_rollouts(status_in=["succeeded"])

        if not rollouts:
            print("No rollouts found.")
            return

        print(f"Found {len(rollouts)} rollouts.")

        exported = []

        # 只统计“最终被导出”的rollouts的reward
        total_reward = 0.0
        valid_rewards = 0

        skipped_no_spans = 0
        skipped_no_question = 0
        skipped_question_not_found = 0

        for attempt in rollouts:
            rollout_id = attempt.rollout_id

            spans = await client.query_spans(rollout_id=rollout_id, attempt_id="latest")
            if not spans:
                print(f"[SKIP] No spans found for Rollout ID {rollout_id}.")
                skipped_no_spans += 1
                continue

            first_span = spans[0]
            attempt_id = getattr(first_span, "attempt_id", "latest")

            messages = messages_adapter.adapt(spans)
            triplets = triplets_adapter.adapt(spans)
            triplets_dict = [triplet.model_dump() for triplet in triplets]

            # Extract finish_reason for each LLM call from triplet raw_content
            llm_call_details = []
            for triplet in triplets:
                raw = triplet.response.get("raw_content", []) if isinstance(triplet.response, dict) else []
                if raw and isinstance(raw[0], dict):
                    fr = raw[0].get("finish_reason")
                    if fr is not None:
                        llm_call_details.append({"finish_reason": fr})
            finish_reasons = [d["finish_reason"] for d in llm_call_details]
            truncated_count = sum(1 for fr in finish_reasons if fr == "length")

            question_text = extract_question_from_messages(messages)
            if question_text is None:
                print(f"[SKIP] Could not extract question for rollout {rollout_id}")
                skipped_no_question += 1
                continue

            meta = question_lookup.get(question_text)
            if meta is None:
                print(
                    f"[SKIP] Question not found in {mode}.parquet for rollout {rollout_id}: {question_text!r}"
                )
                skipped_question_not_found += 1
                continue

            db_id = meta["db_id"]
            gold_query = meta["query"]

            final_reward = find_final_reward(spans)
            if final_reward is None:
                print(f"[WARN] Rollout ID {rollout_id} has no valid reward.")
            else:
                total_reward += final_reward
                valid_rewards += 1

            # 提取 pass@n 候选SQL信息（如果有）
            pass_at_n_candidates = None
            pass_at_n_n = None
            for span in spans:
                annotations = getattr(span, "annotations", None) or {}
                if "pass_at_n.candidates" in annotations:
                    try:
                        pass_at_n_candidates = json.loads(annotations["pass_at_n.candidates"])
                    except (json.JSONDecodeError, TypeError):
                        pass
                    pass_at_n_n = annotations.get("pass_at_n.n")
                    break

            entry = {
                "rollout_id": rollout_id,
                "attempt_id": attempt_id,
                "db_id": db_id,
                "question": question_text,
                "query": gold_query,
                "messages": messages,
                "triplets": triplets_dict,
                "final_reward": final_reward,
                "llm_calls": llm_call_details,
                "finish_reasons": finish_reasons,
                "truncated_count": truncated_count,
            }
            if pass_at_n_candidates is not None:
                entry["pass_at_n_candidates"] = pass_at_n_candidates
                entry["pass_at_n_n"] = pass_at_n_n

            exported.append(entry)

        if valid_rewards > 0:
            avg_final_reward = total_reward / valid_rewards
            print(
                f"Average final reward for {valid_rewards} exported rollouts with valid reward: {avg_final_reward}"
            )
            exported.append({"avg_final_reward": avg_final_reward})
        else:
            print("No valid rewards to compute average for exported rollouts.")

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(exported, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(exported)} rollouts to {output_path}")
        print(
            "Skipped summary: "
            f"no_spans={skipped_no_spans}, "
            f"no_question={skipped_no_question}, "
            f"question_not_found={skipped_question_not_found}"
        )

    finally:
        await client.close()



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("port", type=int)
    parser.add_argument("time_stamp", type=str)
    parser.add_argument("truncate", type=str, choices=["yes", "no"])
    parser.add_argument("mode", type=str, choices=["test", "test_dev_500", "train"])
    args = parser.parse_args()

    client = agl.LightningStoreClient(server_address=f"http://localhost:{args.port}")
    output_path = Path(f"../training_result/{args.time_stamp}/test_rolllouts_truncated_{args.truncate}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    asyncio.run(export_rollouts(mode=args.mode, output_path=output_path, client=client))