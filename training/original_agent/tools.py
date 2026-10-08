"""tools.py — Unified tool registry for ReAct agent (v2).

集中管理 4 个工具（函数式调用风格）：

| Tool                  | Args                                | Purpose                                          |
|-----------------------|-------------------------------------|--------------------------------------------------|
| get_other_tables      | ()                                  | 列出初始 schema 之外的表（含描述）               |
| describe_table        | (table_name)                        | 单表完整列信息（type/desc/key/samples）          |
| search_column_values  | (kw1, kw2, ...)                     | 向量检索数据库中相似的值                         |
| find_relations        | (table_a, table_b)                  | 两表连接关系：DIRECT / INDIRECT via [X] / NOT_CONNECTED |

设计原则：
1. 每个工具是纯函数 + ToolResult（success/observation/data/error_type），便于诊断系统使用 data 做反馈。
2. ToolExecutor 持有 context（db_id/full_schema/initial_tables/chroma_root），通过 call(action_text)
   提供鲁棒的解析与错误反馈：parse_error / unknown_tool / arg_error / not_found / runtime_error
   全部以 Observation 文本回传 agent，绝不抛异常终止流程。
3. 工具调用格式严格使用 ``tool_name(arg1, arg2, ...)`` 函数式（不用 JSON）。
"""

from __future__ import annotations

import logging
import os
import re
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Result type
# =============================================================================


@dataclass
class ToolResult:
    """Unified tool result.

    Attributes:
        success: True if the tool ran successfully and returned valid data.
        observation: Plain-text Observation to feed back to the agent.
        data: Structured payload for downstream diagnostic / reward systems.
        error_type: One of ``parse_error`` / ``unknown_tool`` / ``arg_error``
            / ``not_found`` / ``runtime_error``; ``None`` on success.
    """

    success: bool
    observation: str
    data: Dict[str, Any] = field(default_factory=dict)
    error_type: Optional[str] = None

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.observation


# =============================================================================
# FK graph helpers (used by find_relations)
# =============================================================================


def build_fk_graph(full_schema: Dict[str, Any]) -> Dict[str, Set[str]]:
    """Build an undirected FK adjacency graph (lower-cased table names).

    Args:
        full_schema: One DB entry from ``db_full_schema.json`` (i.e. the value
            keyed by ``db_id``). Expected layout:
            ``{"tables": {tname: {"columns": {cname: {"key_info": "FOREIGN KEY -> t.c"}}}}}``.

    Returns:
        ``{table_lower: {neighbor_lower, ...}}``.
    """
    graph: Dict[str, Set[str]] = {}
    tables = full_schema.get("tables", {}) if full_schema else {}
    for tname, tinfo in tables.items():
        t_low = tname.lower()
        graph.setdefault(t_low, set())
        for cinfo in tinfo.get("columns", {}).values():
            key_info = cinfo.get("key_info") or ""
            m = re.search(r"FOREIGN KEY\s*->\s*(\w+)\.\w+", key_info, flags=re.IGNORECASE)
            if m:
                ref = m.group(1).lower()
                graph[t_low].add(ref)
                graph.setdefault(ref, set()).add(t_low)
    return graph


def find_fk_path(
    graph: Dict[str, Set[str]], a: str, b: str, max_hop: int = 3
) -> Optional[List[str]]:
    """BFS shortest FK path between two tables (inclusive of endpoints).

    Args:
        graph: Output of :func:`build_fk_graph`.
        a: Source table (any case).
        b: Target table (any case).
        max_hop: Maximum number of hops (edges) to consider.

    Returns:
        Lower-cased list of tables on the path, or ``None`` if disconnected.
    """
    a_low, b_low = a.lower(), b.lower()
    if a_low not in graph or b_low not in graph:
        return None
    if a_low == b_low:
        return [a_low]
    visited = {a_low}
    q = deque([(a_low, [a_low])])
    while q:
        node, path = q.popleft()
        if len(path) > max_hop:
            continue
        for nb in graph.get(node, set()):
            if nb in visited:
                continue
            new_path = path + [nb]
            if nb == b_low:
                return new_path
            visited.add(nb)
            q.append((nb, new_path))
    return None


def _restore_case(tables: Dict[str, Any], lower_path: List[str]) -> List[str]:
    """Map lower-cased table names back to their original case from ``tables``."""
    real_path: List[str] = []
    for node in lower_path:
        for tname in tables:
            if tname.lower() == node:
                real_path.append(tname)
                break
        else:
            real_path.append(node)
    return real_path


# =============================================================================
# Tool implementations (pure functions)
# =============================================================================


def get_other_tables(
    full_schema: Dict[str, Any],
    initial_tables: List[str],
) -> ToolResult:
    """List tables NOT in the initial schema, with descriptions."""
    if not full_schema:
        return ToolResult(False, "No additional table information available.", error_type="not_found")
    tables = full_schema.get("tables", {})
    initial_lower = {t.lower() for t in (initial_tables or [])}
    others: List[Tuple[str, str]] = []
    for tname, tinfo in tables.items():
        if tname.lower() in initial_lower:
            continue
        desc = tinfo.get("description") or "No description available"
        others.append((tname, desc))
    if not others:
        return ToolResult(
            True,
            "No other tables available. Your initial schema contains all tables in this database.",
            data={"other_tables": []},
        )
    lines = ["Tables not in your initial schema:"]
    for tname, desc in others:
        lines.append(f"- {tname}: {desc}")
    return ToolResult(
        True,
        "\n".join(lines),
        data={"other_tables": [t for t, _ in others]},
    )


def describe_table(table_name: str, full_schema: Dict[str, Any]) -> ToolResult:
    """Return full column list of a specific table."""
    if not full_schema:
        return ToolResult(
            False,
            f"No schema information available for table '{table_name}'.",
            error_type="not_found",
        )
    tables = full_schema.get("tables", {})
    target = None
    for tname, tinfo in tables.items():
        if tname.lower() == table_name.lower():
            target = (tname, tinfo)
            break
    if target is None:
        avail = list(tables.keys())[:10]
        return ToolResult(
            False,
            (
                f"Table '{table_name}' not found in the database. "
                f"Available tables (first 10): {avail}. "
                f"Hint: call get_other_tables() to discover tables outside your initial schema."
            ),
            data={"queried": table_name, "available_sample": avail},
            error_type="not_found",
        )
    tname, tinfo = target
    desc = tinfo.get("description") or ""
    columns = tinfo.get("columns", {})
    lines = [f"Table: {tname}", f"Description: {desc}", "Columns:"]
    for cname, cinfo in columns.items():
        ctype = cinfo.get("type", "unknown")
        cdesc = cinfo.get("description") or ""
        key = cinfo.get("key_info") or "None"
        samples = cinfo.get("samples") or []
        sample_str = f" (samples: {samples[:3]})" if samples else ""
        lines.append(f"  - {cname} ({ctype}): {cdesc} | Key: {key}{sample_str}")
    return ToolResult(
        True,
        "\n".join(lines),
        data={"table": tname, "columns": list(columns.keys())},
    )


def search_column_values(
    keywords: List[str],
    db_id: str,
    chroma_root: Optional[str] = None,
    threshold: float = 0.8,
    top_k: int = 5,
) -> ToolResult:
    """Vector-search column values across the entire database."""
    if not keywords:
        return ToolResult(
            False,
            "Error: search_column_values requires at least one keyword.",
            error_type="arg_error",
        )
    chroma_root = chroma_root or os.path.join(
        os.path.dirname(__file__), "chroma", "spider_test"
    )
    try:
        # retrieve.py 位置优先级: $AGL_RETRIEVE_DIR > original_agent/db_schema_preprocess
        preprocess_dir = os.environ.get(
            "AGL_RETRIEVE_DIR",
            os.path.join(os.path.dirname(__file__), "db_schema_preprocess"),
        )
        if preprocess_dir not in sys.path:
            sys.path.insert(0, preprocess_dir)
        from retrieve import DatabaseCellRetrieval  # type: ignore

        retriever = DatabaseCellRetrieval(
            database_literals=keywords,
            search_client=chroma_root,
            collection_name=db_id,
        )
        results = retriever.retrieve(threshold=threshold, k=top_k)
        if not results:
            return ToolResult(
                True,
                f"No similar values found in database '{db_id}' for keywords: {keywords}",
                data={"keywords": keywords, "matches": []},
            )
        lines = [f"Values found in database '{db_id}' (Top {len(results)} matches):"]
        for r in results:
            lines.append(
                f"  - Value: \"{r['content']}\" (Table: {r['table']}, Column: {r['column']})"
            )
        return ToolResult(
            True,
            "\n".join(lines),
            data={"keywords": keywords, "matches": results},
        )
    except Exception as e:
        logger.error("[search_column_values] Vector retrieval failed: %s", e)
        return ToolResult(
            False,
            f"Error searching values: {e}",
            error_type="runtime_error",
        )


def find_relations(
    table_a: str,
    table_b: str,
    full_schema: Dict[str, Any],
    max_hop: int = 3,
) -> ToolResult:
    """Check connection between two tables via FK graph.

    Returns one of:
      * ``DIRECT``: a single FK link between the two tables.
      * ``INDIRECT via [X1, X2, ...]``: connected through intermediate tables.
      * ``NOT_CONNECTED``: no FK path within ``max_hop`` hops.
    """
    if not full_schema:
        return ToolResult(False, "No schema information available.", error_type="not_found")
    tables = full_schema.get("tables", {})
    a_real = next((t for t in tables if t.lower() == table_a.lower()), None)
    b_real = next((t for t in tables if t.lower() == table_b.lower()), None)
    missing = [t for t, real in [(table_a, a_real), (table_b, b_real)] if real is None]
    if missing:
        return ToolResult(
            False,
            (
                f"Table(s) not found: {missing}. "
                f"Hint: call get_other_tables() to discover available tables."
            ),
            data={"missing": missing},
            error_type="not_found",
        )
    if a_real == b_real:
        return ToolResult(
            True,
            f"Same table: '{a_real}'. find_relations expects two distinct tables.",
            data={"relation": "SAME", "path": [a_real]},
        )
    graph = build_fk_graph(full_schema)
    a_low, b_low = a_real.lower(), b_real.lower()  # type: ignore[union-attr]
    # 1) DIRECT
    if b_low in graph.get(a_low, set()):
        return ToolResult(
            True,
            f"DIRECT: {a_real} <-> {b_real} (foreign key constraint).",
            data={"relation": "DIRECT", "path": [a_real, b_real], "bridges": []},
        )
    # 2) INDIRECT (BFS)
    path = find_fk_path(graph, a_low, b_low, max_hop=max_hop)
    if path is None:
        return ToolResult(
            True,
            (
                f"NOT_CONNECTED: {a_real} and {b_real} have no foreign-key path within "
                f"{max_hop} hops. They likely should NOT be joined directly."
            ),
            data={"relation": "NOT_CONNECTED", "path": None, "bridges": []},
        )
    real_path = _restore_case(tables, path)
    bridges = real_path[1:-1]
    return ToolResult(
        True,
        (
            f"INDIRECT via {bridges}: {' -> '.join(real_path)}. "
            f"To join {a_real} and {b_real}, you must go through these bridge tables."
        ),
        data={"relation": "INDIRECT", "path": real_path, "bridges": bridges},
    )


# =============================================================================
# Registry & robust dispatcher
# =============================================================================


TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "get_other_tables": {
        "description": "List tables NOT in your initial schema (with descriptions).",
        "usage": "get_other_tables()",
        "min_args": 0,
        "max_args": 0,
    },
    "describe_table": {
        "description": "Show full column list (type/desc/key/samples) of a specific table.",
        "usage": "describe_table(table_name)",
        "min_args": 1,
        "max_args": 1,
    },
    "search_column_values": {
        "description": "Vector-search similar values in the DB.",
        "usage": "search_column_values(keyword1, keyword2, ...)",
        "min_args": 1,
        "max_args": -1,
    },
    "find_relations": {
        "description": "Check connection between two tables: DIRECT / INDIRECT via [X] / NOT_CONNECTED.",
        "usage": "find_relations(table_a, table_b)",
        "min_args": 2,
        "max_args": 2,
    },
}


def _split_args(args_str: str) -> List[str]:
    """CSV-like split that respects single/double quotes."""
    out: List[str] = []
    cur = ""
    quote: Optional[str] = None
    for c in args_str:
        if quote:
            if c == quote:
                quote = None
            else:
                cur += c
        elif c in ("'", '"'):
            quote = c
        elif c == ",":
            out.append(cur.strip())
            cur = ""
        else:
            cur += c
    if cur.strip():
        out.append(cur.strip())
    return out


def parse_tool_call(text: str) -> Tuple[bool, Optional[str], Optional[List[str]], Optional[str]]:
    """Parse ``tool_name(arg1, arg2, ...)``.

    Returns ``(success, name, args_list, error_msg)``.
    """
    text = (text or "").strip()
    m = re.match(r"^(\w+)\s*\(\s*(.*?)\s*\)\s*$", text, flags=re.DOTALL)
    if not m:
        return (
            False,
            None,
            None,
            (
                f"Cannot parse tool call from: {text!r}. "
                f"Expected format: tool_name(arg1, arg2, ...). "
                f"Available tools: {list(TOOL_REGISTRY.keys())}."
            ),
        )
    name = m.group(1).strip().lower()
    args_str = m.group(2).strip()
    args = _split_args(args_str) if args_str else []
    return True, name, args, None


def _arity_msg(min_a: int, max_a: int) -> str:
    if min_a == max_a:
        return str(min_a)
    if max_a < 0:
        return f"{min_a}+"
    return f"{min_a}-{max_a}"


# =============================================================================
# Tool Executor
# =============================================================================


class ToolExecutor:
    """Stateful tool dispatcher that holds the per-rollout context.

    Lifecycle: instantiated once per question/rollout. Holds ``accumulated_schema``
    so that ``add_schema`` can incrementally grow the agent's working memory.
    """

    def __init__(
        self,
        db_id: str,
        full_schema: Dict[str, Any],
        initial_tables: Optional[List[str]] = None,
        chroma_root: Optional[str] = None,
    ) -> None:
        """Initialise the executor.

        Args:
            db_id: Database id (used as Chroma collection name).
            full_schema: ``db_full_schema.json[db_id]``.
            initial_tables: Tables already shown in the initial schema (for
                ``get_other_tables``).
            chroma_root: Override the default Chroma store path.
        """
        self.db_id = db_id
        self.full_schema = full_schema or {}
        self.initial_tables: List[str] = list(initial_tables or [])
        self.chroma_root = chroma_root

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def call(self, action_text: str) -> ToolResult:
        """Robustly parse and dispatch ``tool_name(args)`` text.

        Never raises: every failure is wrapped in a :class:`ToolResult` with
        the appropriate ``error_type`` so the agent can self-correct.
        """
        ok, name, args, err = parse_tool_call(action_text)
        if not ok:
            logger.warning("[ToolExecutor] parse_error: %s", err)
            return ToolResult(False, f"[Parse Error] {err}", error_type="parse_error")

        meta = TOOL_REGISTRY.get(name or "")
        if meta is None:
            avail = list(TOOL_REGISTRY.keys())
            return ToolResult(
                False,
                (
                    f"[Unknown Tool] '{name}' is not a valid tool. "
                    f"Available tools: {avail}. "
                    f"Hint: use exact spelling (snake_case)."
                ),
                error_type="unknown_tool",
            )

        n_args = len(args or [])
        min_a, max_a = meta["min_args"], meta["max_args"]
        if n_args < min_a or (max_a >= 0 and n_args > max_a):
            return ToolResult(
                False,
                (
                    f"[Argument Error] '{name}' expects {_arity_msg(min_a, max_a)} arg(s), "
                    f"got {n_args}. Usage: {meta['usage']}"
                ),
                error_type="arg_error",
            )

        try:
            if name == "get_other_tables":
                return get_other_tables(self.full_schema, self.initial_tables)
            if name == "describe_table":
                return describe_table(args[0], self.full_schema)  # type: ignore[index]
            if name == "search_column_values":
                return search_column_values(args or [], self.db_id, self.chroma_root)
            if name == "find_relations":
                return find_relations(args[0], args[1], self.full_schema)  # type: ignore[index]
        except Exception as e:  # pragma: no cover - defensive
            logger.exception("Tool '%s' raised exception", name)
            return ToolResult(
                False,
                f"[Runtime Error] Tool '{name}' failed: {e}",
                error_type="runtime_error",
            )

        return ToolResult(
            False,
            f"[Internal] Tool '{name}' is registered but not implemented.",
            error_type="runtime_error",
        )

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def list_tools(self) -> str:
        """Render the tool registry as a human-readable string (for prompt)."""
        lines = ["Available tools:"]
        for name, meta in TOOL_REGISTRY.items():
            lines.append(f"- {meta['usage']}: {meta['description']}")
        return "\n".join(lines)


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    "ToolResult",
    "ToolExecutor",
    "TOOL_REGISTRY",
    "get_other_tables",
    "describe_table",
    "search_column_values",
    "find_relations",
    "build_fk_graph",
    "find_fk_path",
    "parse_tool_call",
]
