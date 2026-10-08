"""Pure SQL extraction and normalization from LLM responses (Req 8)."""
from __future__ import annotations

import re
from typing import Literal

ExtractionStatus = Literal["ok", "fallback", "truncated", "missing"]

PLACEHOLDER_SQL = "SELECT 1 FROM __invalid__"

_WS_RE = re.compile(r"\s+")
_THINK_CLOSED_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)
_SQL_BLOCK_RE = re.compile(r"```sql\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_STMT_RE = re.compile(r"(?is)\b(?:SELECT|WITH)\b.*?(?=;|```|\n\s*\n|$)")


def normalize_sql(sql: str) -> str:
    """Collapse whitespace to single spaces, strip, and drop trailing semicolons."""
    s = _WS_RE.sub(" ", sql).strip()
    while True:
        t = s.rstrip(";").strip()
        if t == s:
            return s
        s = t


def _strip_think(text: str) -> str:
    t = _THINK_CLOSED_RE.sub("", text)
    return _THINK_OPEN_RE.sub("", t)  # unclosed <think>: drop to end


def extract_sql(text: str | None, finish_reason: str | None) -> tuple[str, ExtractionStatus]:
    """Return (single-line SQL, status). Output is always non-empty and single-line."""
    t = _strip_think(text or "")

    for body in reversed(_SQL_BLOCK_RE.findall(t)):
        sql = normalize_sql(body)
        if sql:
            return sql, "ok"

    for m in reversed(list(_STMT_RE.finditer(t))):
        sql = normalize_sql(m.group(0))
        if sql:
            return sql, "fallback"

    return PLACEHOLDER_SQL, ("truncated" if finish_reason == "length" else "missing")
