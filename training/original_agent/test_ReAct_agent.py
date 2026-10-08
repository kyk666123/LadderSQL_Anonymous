import json
import os
from datetime import datetime
import asyncio
from pathlib import Path
import argparse
from typing import Any, Dict, List, cast
import pandas as pd
import agentlightning as agl
from .ReAct_Agent import LitAgent
from utils.extract_rollouts import export_rollouts



async def test_sql_agent(mode: str, truncate: str, n_runners: int, port: int, time_stamp: str, max_turns: int, val_concurrent: int, val_temperature: float, dataset: str = "spider", vllm_port: int = 9000, vllm_num: int = 1, eval_mode: str = "best_of_n", schema_cache: str | None = None):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(f"./results/{time_stamp}/test_rollouts_{truncate}_truncated_{now}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = agl.LightningStoreClient(server_address=f"http://127.0.0.1:{port}")

    strategy = agl.ClientServerExecutionStrategy(
        role="both",          # 同一个脚本里起 algorithm + runners
        n_runners=n_runners,         # 和 n_workers 对齐
        managed_store=False,  # 关键：不要再包一层 LightningStoreServer
        # server_host / server_port 对你这种"外部 store"场景就不再重要了
    )

    # 根据dataset参数选择数据路径
    if dataset == "bird":
        # BIRD数据集路径
        if mode == "bird_dev":
            spider_data_path = Path("/path/to/nl2sql_dataset/bird/dev_20240627/dev.parquet")
        else:
            raise ValueError(f"Unknown mode '{mode}' for bird dataset")
    else:
        # Spider数据集路径（保持原有逻辑）
        spider_data_path = Path(os.environ.get("SPIDER_DATA_DIR", "data")) / f"{mode}.parquet"
    
    print(f"Dataset: {dataset}, Mode: {mode}")
    print(f"Data path: {spider_data_path}")
    if not os.path.exists(spider_data_path):
        raise FileNotFoundError(f"Data file {spider_data_path} does not exist.")
    df = pd.read_parquet(spider_data_path)  # type: ignore
    df = cast(List[Dict[str, Any]], df.to_dict(orient="records"))  # type: ignore
    
    # with open("/path/to/LadderSQL/examples/my_spider/agent/results/20260331/test_rollouts_new_llm_truncated_20260331_1504.json", "r") as f:
    #     existing_rollouts = json.load(f)
    
    # keys = []
    # for item in existing_rollouts:
    #     if "rollout_id" in item:
    #         question = item["question"]
    #         db_id = item["db_id"]
    #         key = f"{question}_{db_id}"
    #         keys.append(key)
    
    # new_df = []
    # for item in df:
    #     question = item["question"]
    #     db_id = item["db_id"]
    #     key = f"{question}_{db_id}"
    #     if key in keys:
    #         continue
    #     else:
    #         print(key)
    #         new_df.append(item)

    # print("Spider 1.0 Data Length:", len(new_df))
            
    print("Spider 1.0 Data Length:", len(df))

    # 多vLLM实例endpoint配置
    llm_endpoints = [f"http://localhost:{vllm_port + i}/v1" for i in range(vllm_num)]

    trainer = agl.Trainer(
        # n_runners=n_runners,
        initial_resources={
            "main_llm": agl.LLM(
                endpoint=llm_endpoints[0],
                model="qwen2.5-coder",
                sampling_parameters={"temperature": val_temperature},
            ),
        },
        # port = port
        store=client,
        strategy=strategy
    )
    
    agent = LitAgent(
        is_train=False,
        truncate=truncate, 
        max_turns=max_turns, 
        val_concurrent=val_concurrent,
        val_temperature=val_temperature,
        dataset=dataset,
        mode=mode,
        llm_endpoints=llm_endpoints,
        reward_mode="binary",  # test/eval 统一用执行准确率
        eval_mode=eval_mode,
        schema_cache_path=schema_cache,
    )

    # await asyncio.to_thread(trainer.dev, agent, val_dataset=new_df)
    
    await asyncio.to_thread(trainer.dev, agent, val_dataset=df)

    await export_rollouts(mode, output_path, client, dataset=dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, choices=["train", "test", "test_dev_500", "dev", "bird_dev"])
    parser.add_argument("time_stamp", type=str)
    parser.add_argument("--port", type=int, default=4747, help="LightningStore port")
    parser.add_argument("--truncate", type=str, choices=["yes", "no", "golden", "llm", "bilink", "new_llm", "cache", "glm-5-cache", "pipeline"], default="cache")
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--n-runners", type=int, default=8)
    parser.add_argument("--val-concurrent", type=int, default=1)
    parser.add_argument("--val-temperature", type=float, default=0.0)
    parser.add_argument("--dataset", type=str, choices=["spider", "bird"], default="spider")
    parser.add_argument("--vllm-port", type=int, default=9000, help="vLLM base port")
    parser.add_argument("--vllm-num", type=int, default=1, help="Number of vLLM instances (consecutive ports)")
    parser.add_argument("--eval-mode", type=str, choices=["best_of_n", "pass_at_n"], default="best_of_n", help="Evaluation mode: best_of_n (summarizer picks best) or pass_at_n (any correct = 1)")
    parser.add_argument("--schema-cache", type=str, default=None, help="Path to schema linking cache JSON file")
    args = parser.parse_args()

    asyncio.run(test_sql_agent(
        args.mode, 
        args.truncate, 
        args.n_runners, 
        args.port, 
        args.time_stamp, 
        args.max_turns, 
        args.val_concurrent,
        args.val_temperature,
        args.dataset,
        args.vllm_port,
        args.vllm_num,
        args.eval_mode,
        args.schema_cache,
        )) 