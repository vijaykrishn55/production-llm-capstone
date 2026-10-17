from __future__ import annotations

import ast
import json
import logging
import math
import re
import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)
ALLOWED_SQL_FUNCTIONS = {"count", "sum", "avg", "min", "max", "round", "lower", "upper", "length", "substr", "coalesce", "ifnull", "date", "strftime"}

def _safe_pow(base, exponent):
    if not isinstance(exponent, (int, float)) or abs(exponent) > 100:
        raise ValueError("Exponent must have absolute value <= 100")
    return pow(base, exponent)


SAFE_FUNCTIONS = {
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "round": round,
    "len": len,
    "sorted": sorted,
    "sqrt": math.sqrt,
    "floor": math.floor,
    "ceil": math.ceil,
    "log": math.log,
    "pow": _safe_pow,
}
ALLOWED_AST_NODES = (
    ast.Expression, ast.Constant, ast.List, ast.Tuple, ast.Dict, ast.Set,
    ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
    ast.Mod, ast.Pow, ast.USub, ast.UAdd, ast.Call, ast.Name, ast.Load,
    ast.Subscript, ast.Slice, ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE,
    ast.Gt, ast.GtE, ast.BoolOp, ast.And, ast.Or, ast.IfExp,
)


def safe_python_execute(expression: str) -> dict[str, Any]:
    """Evaluate a restricted Python expression, never arbitrary statements or imports."""
    if not expression or len(expression) > 500:
        raise ValueError("Provide a Python expression of at most 500 characters")
    tree = ast.parse(expression, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_AST_NODES):
            raise ValueError(f"Unsupported Python syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in SAFE_FUNCTIONS:
            raise ValueError(f"Unknown name: {node.id}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in SAFE_FUNCTIONS:
                raise ValueError("Only approved math/data functions may be called")
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and node.value.bit_length() > 256:
            raise ValueError("Integer constant is too large")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            def is_sequence_operand(operand: ast.AST) -> bool:
                return isinstance(operand, (ast.List, ast.Tuple, ast.Set)) or (
                    isinstance(operand, ast.Constant) and isinstance(operand.value, (str, bytes))
                )

            if is_sequence_operand(node.left) and isinstance(node.right, ast.Constant):
                count_node = node.right
            elif is_sequence_operand(node.right) and isinstance(node.left, ast.Constant):
                count_node = node.left
            else:
                count_node = None
            if count_node is not None and isinstance(count_node.value, int) and abs(count_node.value) > 10_000:
                raise ValueError("Sequence repetition is too large")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            if not isinstance(node.right, ast.Constant) or not isinstance(node.right.value, (int, float)) or abs(node.right.value) > 100:
                raise ValueError("Exponent must be a numeric constant with absolute value <= 100")
    value = eval(compile(tree, "<safe-python-expression>", "eval"), {"__builtins__": {}}, SAFE_FUNCTIONS)  # noqa: S307
    return {"result": value, "result_type": type(value).__name__}


class ToolRegistry:
    def __init__(self, db_path: str, retriever: HybridRetriever):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.retriever = retriever
        self._initialize_database()

    def _connect(self, readonly: bool = False) -> sqlite3.Connection:
        if readonly:
            connection = sqlite3.connect(f"file:{self.db_path.resolve()}?mode=ro", uri=True, timeout=3)
            connection.execute("PRAGMA query_only = ON")
            deadline = time.monotonic() + 1.5
            connection.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 5_000)

            def authorizer(action, arg1, arg2, _database, _trigger):
                if action == sqlite3.SQLITE_SELECT:
                    return sqlite3.SQLITE_OK
                if action == sqlite3.SQLITE_READ and arg1 == "sales":
                    return sqlite3.SQLITE_OK
                if action == sqlite3.SQLITE_FUNCTION and str(arg2 or arg1 or "").lower() in ALLOWED_SQL_FUNCTIONS:
                    return sqlite3.SQLITE_OK
                if hasattr(sqlite3, "SQLITE_RECURSIVE") and action == sqlite3.SQLITE_RECURSIVE:
                    return sqlite3.SQLITE_OK
                return sqlite3.SQLITE_DENY

            connection.set_authorizer(authorizer)
        else:
            connection = sqlite3.connect(self.db_path, timeout=3)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS sales (id INTEGER PRIMARY KEY, product TEXT NOT NULL, region TEXT NOT NULL, amount REAL NOT NULL, sold_on TEXT NOT NULL)")
            count = connection.execute("SELECT COUNT(*) FROM sales").fetchone()[0]
            if count == 0:
                connection.executemany(
                    "INSERT INTO sales(product, region, amount, sold_on) VALUES (?, ?, ?, ?)",
                    [
                        ("Cloud Migration", "North", 12000.0, "2026-01-10"),
                        ("AI Platform", "West", 18500.0, "2026-02-05"),
                        ("Cloud Migration", "South", 9400.0, "2026-02-19"),
                        ("Observability", "North", 7600.0, "2026-03-02"),
                    ],
                )

    def sql_query(self, query: str) -> dict[str, Any]:
        cleaned = query.strip()
        if not cleaned or len(cleaned) > 2_000:
            raise ValueError("SQL must contain 1-2000 characters")
        if not cleaned.lower().startswith(("select ", "with ")):
            raise ValueError("Only read-only SELECT queries are allowed")
        if ";" in cleaned.rstrip(";"):
            raise ValueError("Only one SQL statement is allowed")
        forbidden = ("insert", "update", "delete", "drop", "alter", "attach", "detach", "pragma", "vacuum", "create", "replace")
        if re.search(r"\b(" + "|".join(forbidden) + r")\b", cleaned, flags=re.IGNORECASE):
            raise ValueError("Query contains a prohibited SQL operation")
        cleaned = cleaned.rstrip(";")
        if not re.search(r"\blimit\s+\d+\b", cleaned, flags=re.IGNORECASE):
            cleaned += " LIMIT 50"
        with self._connect(readonly=True) as connection:
            rows = connection.execute(cleaned).fetchmany(51)
        return {"rows": [dict(row) for row in rows[:50]], "row_count": min(len(rows), 50)}

    def execute(self, action: str, arguments: dict[str, Any]) -> Any:
        if action == "python_exec":
            return safe_python_execute(str(arguments.get("expression", "")))
        if action == "sql_query":
            return self.sql_query(str(arguments.get("query", "")))
        if action == "rag_search":
            query = str(arguments.get("query", "")).strip()
            if not query:
                raise ValueError("query is required")
            rows = self.retriever.search(query, top_k=5)
            return [{"source": row["source"], "text": row["text"], "score": row["score"]} for row in rows]
        raise ValueError("Unknown tool")
