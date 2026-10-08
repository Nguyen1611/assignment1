"""Per-example wrapper around the official Spider evaluator (Req 2.7, 10, 11).

EX/EM come from functions imported from the official `test-suite-sql-eval`
clone; `evaluate_one` mirrors one iteration of the official `evaluate()` loop
for etype="all", plug_value=False, keep_distinct=False.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator, Literal

import pandas as pd

from common import config

LEVELS = ["easy", "medium", "hard", "extra", "all"]
EXEC_TIMEOUT_S = 60.0

# Copied verbatim from the official evaluate() (empty-SQL substitute).
EMPTY_SQL = {
    "except": None,
    "from": {
        "conds": [],
        "table_units": []
    },
    "groupBy": [],
    "having": [],
    "intersect": None,
    "limit": None,
    "orderBy": [],
    "select": [
        False,
        []
    ],
    "union": None,
    "where": []
}


class EvaluationError(RuntimeError):
    """Raised where the official script itself would crash (gold parse, EM)."""

    def __init__(self, example_id, message: str):
        self.example_id = example_id
        super().__init__(f"example {example_id}: {message}")


# ---------------------------------------------------------------- official import
def import_official() -> SimpleNamespace:
    d = str(config.OFFICIAL_DIR)
    if d not in sys.path:
        sys.path.insert(0, d)
    import evaluation as ev  # noqa: E402  (pulls in process_sql, exec_eval)
    import exec_eval as xe  # noqa: E402
    return SimpleNamespace(
        get_schema=ev.get_schema, Schema=ev.Schema, get_sql=ev.get_sql,
        Evaluator=ev.Evaluator, eval_exec_match=xe.eval_exec_match,
        build_valid_col_units=ev.build_valid_col_units,
        rebuild_sql_val=ev.rebuild_sql_val, rebuild_sql_col=ev.rebuild_sql_col,
        build_foreign_key_map_from_json=ev.build_foreign_key_map_from_json,
        postprocess=xe.postprocess,
    )


@contextmanager
def official_cwd() -> Iterator[None]:
    """chdir to OFFICIAL_DIR (exec_eval uses relative 'tmp/'); restore after."""
    prev = os.getcwd()
    (Path(config.OFFICIAL_DIR) / "tmp").mkdir(exist_ok=True)
    os.chdir(config.OFFICIAL_DIR)
    try:
        yield
    finally:
        os.chdir(prev)


# ---------------------------------------------------------------- execution
@dataclass
class ExecResult:
    status: Literal["ok", "error", "timeout"]
    n_rows: int | None
    error: str | None


def execute_sql(db_file: Path, sql: str, timeout_s: float, count_rows: bool = True) -> ExecResult:
    """Read-only, interruptible execution via a progress handler."""
    try:
        conn = sqlite3.connect(f"file:{Path(db_file)}?mode=ro", uri=True, check_same_thread=False)
    except sqlite3.Error as e:
        return ExecResult("error", None, f"{type(e).__name__}: {e}")
    conn.text_factory = lambda b: b.decode(errors="ignore")
    deadline = time.monotonic() + timeout_s
    timed_out = {"v": False}

    def _handler():
        if time.monotonic() > deadline:
            timed_out["v"] = True
            return 1
        return 0

    conn.set_progress_handler(_handler, 10_000)
    try:
        cur = conn.cursor()
        cur.execute(sql)
        if count_rows:
            return ExecResult("ok", len(cur.fetchall()), None)
        cur.fetchone()
        return ExecResult("ok", None, None)
    except Exception as e:  # sqlite3.Error, Warning (multiple statements), etc.
        if timed_out["v"]:
            return ExecResult("timeout", None, f"timeout after {timeout_s}s")
        return ExecResult("error", None, f"{type(e).__name__}: {e}")
    finally:
        conn.close()


# ---------------------------------------------------------------- per example
@dataclass
class EvalRow:
    example_id: int
    db_id: str
    difficulty: str
    ex: int
    em: int
    parse_ok: bool
    parse_error: str | None
    exec_status: str
    exec_error: str | None
    n_rows: int | None


def evaluate_one(api, evaluator, kmaps, db_dir, p_str_raw: str, g_str: str, db_name: str,
                 example_id: int = -1) -> EvalRow:
    p_str = p_str_raw.replace("value", "1")
    db = os.path.join(str(db_dir), db_name, db_name + ".sqlite")
    try:
        schema = api.Schema(api.get_schema(db))
        g_sql = api.get_sql(schema, g_str)
        hardness = evaluator.eval_hardness(g_sql)
    except Exception as e:
        raise EvaluationError(example_id, f"gold failed: {type(e).__name__}: {e}") from e
    try:
        p_sql = api.get_sql(schema, p_str)
        parse_error = None
    except Exception as e:  # official uses bare except
        p_sql = deepcopy(EMPTY_SQL)
        parse_error = f"{type(e).__name__}: {e}"

    pre = execute_sql(Path(db), api.postprocess(p_str), timeout_s=EXEC_TIMEOUT_S)
    if pre.status == "timeout":
        ex = 0  # official would hang; recorded as exec_status="timeout"
    else:
        ex = int(bool(api.eval_exec_match(db=db, p_str=p_str, g_str=g_str, plug_value=False,
                                          keep_distinct=False,
                                          progress_bar_for_each_datapoint=False)))
    try:
        kmap = kmaps[db_name]
        g_valid = api.build_valid_col_units(g_sql["from"]["table_units"], schema)
        g_sql = api.rebuild_sql_val(g_sql)
        g_sql = api.rebuild_sql_col(g_valid, g_sql, kmap)
        p_valid = api.build_valid_col_units(p_sql["from"]["table_units"], schema)
        p_sql = api.rebuild_sql_val(p_sql)
        p_sql = api.rebuild_sql_col(p_valid, p_sql, kmap)
        em = int(evaluator.eval_exact_match(p_sql, g_sql))
    except Exception as e:
        raise EvaluationError(example_id, f"exact match failed: {type(e).__name__}: {e}") from e
    return EvalRow(example_id, db_name, hardness, ex, em, parse_error is None, parse_error,
                   pre.status, pre.error, pre.n_rows)


def _run_without_event_loop(fn):
    """exec_eval calls asyncio.run(), which fails inside Jupyter's running loop;
    in that case run `fn` in a worker thread that has no event loop."""
    import asyncio
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return fn()
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(fn).result()


def _official_pred_str(line: str) -> str:
    # The official reader does l.strip().split('\t') and uses element 0.
    return line.strip().split("\t")[0]


# ---------------------------------------------------------------- aggregation
def aggregate(rows) -> pd.DataFrame:
    """Pure: index = LEVELS, columns count/ex/em (0 when a level is empty, as official)."""
    out = {}
    for lvl in LEVELS:
        sel = [r for r in rows if lvl == "all" or _get(r, "difficulty") == lvl]
        n = len(sel)
        out[lvl] = {
            "count": n,
            "ex": (sum(_get(r, "ex") for r in sel) / n) if n else 0.0,
            "em": (sum(_get(r, "em") for r in sel) / n) if n else 0.0,
        }
    df = pd.DataFrame.from_dict(out, orient="index")[["count", "ex", "em"]]
    df["count"] = df["count"].astype(int)
    return df


def _get(r, k):
    return r[k] if isinstance(r, dict) else getattr(r, k)


@dataclass
class EvalSummary:
    rows: list[EvalRow]
    table: pd.DataFrame
    per_example: pd.DataFrame
    summary: dict = field(default_factory=dict)


def evaluate_predictions(pred_sqls: list[str], gold: list[tuple[str, str]],
                         extraction: list[dict] | None, usage: list[dict] | None,
                         out_dir: Path, run_name: str | None = None) -> EvalSummary:
    if len(pred_sqls) != len(gold):
        raise ValueError(f"{len(pred_sqls)} predictions vs {len(gold)} gold")
    from common.extract import PLACEHOLDER_SQL

    api = import_official()
    db_dir = Path(config.DATA_DIR) / "database"
    kmaps = api.build_foreign_key_map_from_json(str(Path(config.DATA_DIR) / "tables.json"))
    evaluator = api.Evaluator()
    def _loop() -> list[EvalRow]:
        out: list[EvalRow] = []
        sink = io.StringIO()
        with official_cwd():
            for i, (p, (g_str, db)) in enumerate(zip(pred_sqls, gold)):
                with contextlib.redirect_stdout(sink):
                    out.append(evaluate_one(api, evaluator, kmaps, db_dir, _official_pred_str(p),
                                            g_str, db, example_id=i))
                sink.seek(0); sink.truncate()
        return out

    rows = _run_without_event_loop(_loop)

    recs = []
    for i, r in enumerate(rows):
        d = asdict(r)
        d["gold_sql"] = gold[i][0]
        d["pred_sql"] = pred_sqls[i]
        ext = (extraction[i] if extraction and i < len(extraction) else {}) or {}
        d["extraction_status"] = ext.get("extraction_status", ext.get("status"))
        for k, v in ext.items():
            if k not in d and k not in ("status",):
                d[k] = v
        use = (usage[i] if usage and i < len(usage) else {}) or {}
        for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "latency_s",
                  "cost_usd", "provider", "finish_reason", "reasoning_missing"):
            d[k] = use.get(k)
        for k, v in use.items():
            d.setdefault(k, v)
        d["invalid"] = bool(d["extraction_status"] in {"missing", "truncated"}
                            or r.exec_status != "ok")
        recs.append(d)
    per_example = pd.DataFrame(recs)
    table = aggregate(rows)
    summary = {
        "run_name": run_name or Path(out_dir).name,
        "n": len(rows),
        "levels": {lvl: {"count": int(table.loc[lvl, "count"]), "ex": float(table.loc[lvl, "ex"]),
                         "em": float(table.loc[lvl, "em"])} for lvl in LEVELS},
        "n_placeholder": int(sum(p.strip() == PLACEHOLDER_SQL for p in pred_sqls)),
        "n_invalid": int(per_example["invalid"].sum()) if len(per_example) else 0,
        "n_parse_error": int(sum(not r.parse_ok for r in rows)),
        "cli_check": None,
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_example.to_csv(out_dir / "per_example.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return EvalSummary(rows, table, per_example, summary)


def record_cli_check(out_dir: Path, mismatches: list) -> None:
    """Store the CLI totals check in summary.json (Req 11.4)."""
    p = Path(out_dir) / "summary.json"
    s = json.loads(p.read_text(encoding="utf-8"))
    s["cli_check"] = {"passed": not mismatches, "mismatches": [asdict(m) for m in mismatches]}
    p.write_text(json.dumps(s, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- official CLI
def run_official_cli(gold: Path, pred: Path, timeout_s: float = 7200.0) -> dict[str, dict[str, float]]:
    (Path(config.OFFICIAL_DIR) / "tmp").mkdir(exist_ok=True)
    cmd = [sys.executable, "evaluation.py", "--gold", str(Path(gold).resolve()),
           "--pred", str(Path(pred).resolve()), "--db", str(Path(config.DATA_DIR) / "database"),
           "--table", str(Path(config.DATA_DIR) / "tables.json"), "--etype", "all"]
    r = subprocess.run(cmd, cwd=config.OFFICIAL_DIR, capture_output=True, text=True,
                       timeout=timeout_s)
    if r.returncode != 0:
        raise EvaluationError("cli", f"official CLI exited {r.returncode}: {r.stderr[-2000:]}")
    return parse_cli_output(r.stdout)


_NUM = r"[-+]?\d+(?:\.\d+)?"


def parse_cli_output(stdout: str) -> dict[str, dict[str, float]]:
    """Pure: {'count'|'execution'|'exact match': {level: value}} from print_scores output."""
    lines = stdout.splitlines()
    out: dict[str, dict[str, float]] = {}
    seen_exec_hdr = seen_em_hdr = False
    for ln in lines:
        s = ln.strip()
        if "EXECUTION ACCURACY" in s and "TURN" not in s:
            seen_exec_hdr = True
            continue
        if "EXACT MATCHING ACCURACY" in s and "TURN" not in s:
            seen_em_hdr = True
            continue
        for label, ready in (("count", True), ("execution", seen_exec_hdr),
                             ("exact match", seen_em_hdr)):
            if label in out or not ready:
                continue
            m = re.match(rf"^{re.escape(label)}((?:\s+{_NUM})+)\s*$", s)
            if m:
                vals = m.group(1).split()
                if len(vals) >= 5:
                    out[label] = {lvl: (int(v) if label == "count" else float(v))
                                  for lvl, v in zip(LEVELS, vals[:5])}
    missing = {"count", "execution", "exact match"} - set(out)
    if missing:
        raise ValueError(f"could not parse CLI output rows: {sorted(missing)}")
    return out


@dataclass
class Mismatch:
    metric: str
    level: str
    wrapper: float
    cli: float


def compare_totals(wrapper: pd.DataFrame, cli: dict, tol: float = 1e-3) -> list[Mismatch]:
    """Pure: counts must match exactly; EX/EM within `tol` (CLI prints 3 decimals)."""
    out = []
    for lvl in LEVELS:
        for metric, col, exact in (("count", "count", True), ("execution", "ex", False),
                                   ("exact match", "em", False)):
            w = float(wrapper.loc[lvl, col])
            c = float(cli[metric][lvl])
            if (w != c) if exact else (abs(w - c) > tol + 1e-12):
                out.append(Mismatch(metric, lvl, w, c))
    return out
