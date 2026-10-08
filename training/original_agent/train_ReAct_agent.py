from __future__ import annotations

import argparse
import os
import pandas as pd
import agentlightning as agl
from copy import deepcopy
from typing import Any, Dict, Optional
from datetime import datetime
from .ReAct_Agent import LitAgent



RL_TRAINING_CONFIG: Dict[str, Any] = {
    "algorithm": {
        "adv_estimator": "grpo",
        "use_kl_in_reward": False,
    },
    "data": {
        "train_files": "data/train.parquet",
        "val_files": "data/test_dev_500.parquet",
        "train_batch_size": 32, #
        "max_prompt_length": 4096,
        "max_response_length": 2048,
        "truncation": "error",
        "custom_cls": {
            "path": None,
            "name" : None
            },
        # "sampler": {
        #     "class_path": None,
        #     "class_name": None,
        # },
    },
    "actor_rollout_ref": {
        "rollout": {
            "tensor_model_parallel_size": 1,
            "n": 4,  # 
            "log_prob_micro_batch_size_per_gpu": 4,
            "multi_turn": {"format": "hermes"},
            "name": "vllm",
            "gpu_memory_utilization": 0.5,
            "engine_kwargs": {
                "vllm": {
                    "enable_auto_tool_choice": True,
                    "tool_call_parser": "hermes",
                }
            },
        },
        "actor": {
            "ppo_mini_batch_size": 32,
            "ppo_micro_batch_size_per_gpu": 4,
            "optim": {"lr": 1e-6},
            "use_kl_loss": True,  # [20260410] Set True to trigger ref_policy creation for OPSD (kl_loss_coef=0 so KL loss contributes nothing)
            "kl_loss_coef": 0.0,
            "entropy_coeff": 0,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.3,
            "fsdp_config": {
                "param_offload": True,
                "optimizer_offload": True,
            },
        },
        "ref": {
            "log_prob_micro_batch_size_per_gpu": 8,
            "fsdp_config": {"param_offload": True},
        },
        "model": {
            "path": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "use_remove_padding": True,
            "enable_gradient_checkpointing": True,
        },
    },
    "trainer": {
        "n_gpus_per_node": 1,
        "val_before_train": False,
        "critic_warmup": 0,
        "logger": ["console", "wandb"],
        "project_name": "AgentLightning",
        "experiment_name": "spider",
        "nnodes": 1,
        "test_freq": 32,
        "total_epochs": 2,
    },
}


def train(max_turns: int, truncate: str, dataset: str, active_agent: Optional[str], port: int, n_runners: int, config: Dict[str, Any], train_temperature: float, reward_mode: str = "composite", debug: bool = False, traj_mirror_dir: Optional[str] = None, agent_variant: str = "base", schema_cache_path: Optional[str] = None) -> None:
    print(f"Active agent: {active_agent}")
    print(f"Dataset: {dataset}")
    print(f"Agent variant: {agent_variant}  schema_truncate: {truncate}  schema_cache_path: {schema_cache_path}")
    if debug:
        print("[DEBUG] Debug mode enabled: using small data subset, verbose logging")

    # 选择 agent 变体: base = 原 ReAct_Agent.LitAgent; redundant = 高召回冗余 schema 甄别提示
    # redundant_rowtrunc = redundant + 行数版执行结果截断(RowTruncAgent)
    if agent_variant == "redundant":
        from .ReAct_agent_redundant_schema import LitAgent as _LitAgent
    elif agent_variant == "redundant_rowtrunc":
        from .ReAct_agent_redundant_rowtrunc_schema import LitAgent as _LitAgent
    else:
        from .ReAct_Agent import LitAgent as _LitAgent

    agent = _LitAgent(is_train=True, max_turns=max_turns, truncate=truncate, trained_agents=active_agent, train_temperature=train_temperature, dataset=dataset, reward_mode=reward_mode, debug=debug, traj_mirror_dir=traj_mirror_dir, schema_cache_path=schema_cache_path)  # type: ignore
    algorithm = agl.VERL(config)

    trainer = agl.Trainer(port=port, n_runners=n_runners, algorithm=algorithm, adapter={"agent_match": active_agent})
    print("Adapter agent match acknowledged:", trainer.adapter.agent_match)  # type: ignore

    train_data = pd.read_parquet(config["data"]["train_files"]).to_dict(orient="records")  # type: ignore
    val_data = pd.read_parquet(config["data"]["val_files"]).to_dict(orient="records")  # type: ignore

    if debug:
        train_data = train_data[:4]
        val_data = val_data[:2]
        print(f"[DEBUG] Sliced data: train={len(train_data)}, val={len(val_data)}")

    # ===== 临时切片：从 step 5 开始（索引64起），快速验证失败样本 =====
    _skip = int(os.environ.get("TRAIN_DATA_SKIP", "0"))
    if _skip > 0:
        train_data = train_data[_skip:]
        print(f"[DATA SKIP] Skipping first {_skip} samples, remaining: {len(train_data)}")
    # =================================================================

    trainer.fit(agent, train_dataset=train_data, val_dataset=val_data)  # type: ignore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--active-agent", type=str, default=None)
    parser.add_argument("--schema-truncate", type=str, choices=["truncate", "full", "golden", "llm", "cache"], required=True)
    parser.add_argument("--max-turns", type=int, required=True)
    parser.add_argument("--local-model-path", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=["spider", "bird"], required=True)
    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--val-data", type=str, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--train-batch-size", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--sample", type=int, required=True)
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--entropy-coeff", type=float, required=True)
    parser.add_argument("--n-gpus", type=int, required=True)
    parser.add_argument("--save-freq", type=int, required=True)
    parser.add_argument("--test-freq", type=int, required=True)
    parser.add_argument("--val-before-train", action="store_true")
    parser.add_argument("--max-prompt-length", type=int, required=True)
    parser.add_argument("--max-response-length", type=int, required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--prompt-truncate", type=str, choices=["left", "right", "error"], required=True)
    parser.add_argument("--n-runners", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--project-name", type=str, required=True)
    parser.add_argument("--default-local-dir", type=str, required=True)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seq-masking", action="store_true")
    parser.add_argument("--reward-mode", type=str, default="composite", choices=["composite", "binary"],
                        help="Training reward: composite (tiered) or binary (0/1)")
    parser.add_argument("--traj-mirror-dir", type=str, default=None,
                        help="Secondary directory to mirror trajectory logs (e.g. /path/to/...)")
    parser.add_argument("--debug", action="store_true", help="Debug mode: use small data subset, verbose logging")
    parser.add_argument("--agent-variant", type=str, default="base", choices=["base", "redundant", "redundant_rowtrunc"],
                        help="base=ReAct_Agent.LitAgent; redundant=高召回冗余schema甄别提示(ReAct_agent_redundant_schema); redundant_rowtrunc=redundant+行数版执行结果截断")
    parser.add_argument("--schema-cache-path", type=str, default=None,
                        help="truncate=cache 时的预渲染schema缓存json路径(key=db_id|||question)")
    # parser.add_argument("--repetition-penalty", type=float, default=1.0)
    args = parser.parse_args()
    
    config = deepcopy(RL_TRAINING_CONFIG)
    config["actor_rollout_ref"]["model"]["path"] = args.local_model_path
    config["data"]["train_files"] = args.train_data
    config["data"]["val_files"] = args.val_data
    config["trainer"]["total_epochs"] = args.epochs
    config["data"]["train_batch_size"] = args.train_batch_size
    config["actor_rollout_ref"]["actor"]["optim"]["lr"] = args.lr
    config["actor_rollout_ref"]["rollout"]["n"] = args.sample
    config["data"]["custom_cls"]["path"] = args.dataset_path
    config["data"]["custom_cls"]["name"] = args.dataset_name
    config["actor_rollout_ref"]["rollout"]["temperature"]=args.temperature
    # config["actor_rollout_ref"]["rollout"]["repeat_penalty"]=args.repetition_penalty
    config["actor_rollout_ref"]["actor"]["entropy_coeff"]=args.entropy_coeff
    config["trainer"]["n_gpus_per_node"] = args.n_gpus
    config["trainer"]["save_freq"] = args.save_freq
    config["trainer"]["test_freq"] = args.test_freq
    config["trainer"]["val_before_train"] = args.val_before_train
    config["data"]["max_prompt_length"] = args.max_prompt_length
    config["data"]["max_response_length"] = args.max_response_length
    config["actor_rollout_ref"]["rollout"]["engine_kwargs"]["vllm"]["max_model_len"] = args.max_model_len
    config["data"]["truncation"] = args.prompt_truncate
    config["trainer"]["project_name"] = args.project_name
    config["data"]["shuffle"] = args.shuffle
    if args.seq_masking:
        config["actor_rollout_ref"]["actor"]["seq_masking"] = True

    # Debug模式：减少sample数、batch、runners，加快调试循环
    if args.debug:
        config["actor_rollout_ref"]["rollout"]["n"] = 2
        config["data"]["train_batch_size"] = 2
        config["trainer"]["n_gpus_per_node"] = min(args.n_gpus, 2)
        # 同步缩小 PPO 相关 batch 参数，避免 batch 为空
        # total rollouts = train_batch_size * sample = 2 * 2 = 4
        config["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"] = 4
        config["actor_rollout_ref"]["actor"]["ppo_micro_batch_size_per_gpu"] = 2
        config["actor_rollout_ref"]["rollout"]["log_prob_micro_batch_size_per_gpu"] = 2
        config["actor_rollout_ref"]["ref"]["log_prob_micro_batch_size_per_gpu"] = 2
        print(f"[DEBUG] Override: sample=2, train_batch_size=2, ppo_mini=4, ppo_micro=2, n_gpus={config['trainer']['n_gpus_per_node']}")
    
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M")
    exp_name = f"{args.dataset}_{time_stamp}"
    config["trainer"]["experiment_name"] = exp_name
    
    # Checkpoint 保存路径
    config["trainer"]["default_local_dir"] = args.default_local_dir
    config["trainer"]["default_hdfs_dir"] = None
    
    print(f"Checkpoint dir: {args.default_local_dir}")
    
    if args.resume_from_checkpoint is not None:
        config["trainer"]["resume_mode"] = "resume_path"      # 启用从指定路径恢复
        config["trainer"]["resume_from_path"] = args.resume_from_checkpoint  # 指定 checkpoint 目录
    
    train(
        max_turns=args.max_turns, 
        truncate=args.schema_truncate, 
        dataset=args.dataset, 
        active_agent=args.active_agent, 
        port=args.port, 
        n_runners=args.n_runners, 
        config=config,
        train_temperature=args.temperature,
        reward_mode=args.reward_mode,
        debug=args.debug,
        traj_mirror_dir=args.traj_mirror_dir,
        agent_variant=args.agent_variant,
        schema_cache_path=args.schema_cache_path,
    )


if __name__ == "__main__":
    main()