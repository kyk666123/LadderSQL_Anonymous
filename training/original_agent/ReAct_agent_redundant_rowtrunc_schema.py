"""redundant schema 变体 + 行数版执行结果截断 (redundant_rowtrunc)。

复用 ReAct_agent_redundant_schema 的全部 rollout / prompt / reward 逻辑, 仅将其
rollout 内部实例化的 Agent 替换为 RowTruncAgent(按整行 + 整单元格截断执行结果)。

不修改基础 Agent, 也不修改已有 redundant 变体; 通过替换该模块的全局名 `Agent`
完成依赖注入 —— training_rollout / validation_rollout(single_sampling) 内部均按
模块全局名 `Agent(...)` 实例化, 替换后即切换为行数版子类。

每个训练/采样进程只加载并使用一个变体, 因此进程内替换是安全的。
"""
from __future__ import annotations

import original_agent.ReAct_agent_redundant_schema as _redundant
from original_agent.ReAct_Agent_rowtrunc import RowTruncAgent

# 依赖注入: 把 rollout 内部使用的 Agent 换成行数版子类
_redundant.Agent = RowTruncAgent

LitAgent = _redundant.LitAgent

__all__ = ["LitAgent"]
