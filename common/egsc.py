"""Execution-Guided Self-Correction (EGSC) for Qwen2.5-7B (Req 16).

Pure decision rules (needs_retry, feedback_text, select_final) are kept apart from
the I/O loop (run_egsc) so they can be property-tested.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from . import data as data_mod
from .config import RunConfig
from .data import Example
from .evaluate import ExecResult, execute_sql
from .extract import PLACEHOLDER_SQL, extract_sql, normalize_sql
from .llm_client import LLMClient
from .prompts import build_messages, build_retry_messages
from .runner import write_predictions
from .schema import serialize_schema

TRIGGERS = {"error", "timeout"}
PRIMARY_RUN = "qwen25_7b_egsc"
EMPTY_RUN = "qwen25_7b_egsc_empty"


# ---------------------------------------------------------------- pure rules
def needs_retry(res: ExecResult, retry_on_empty: bool) -> bool:
    """Primary: error/timeout only. Variant additionally retries ok-with-zero-rows."""
    if res.status in TRIGGERS:
        return True
    return bool(retry_on_empty and res.status == "ok" and res.n_rows == 0)


def feedback_text(res: ExecResult, timeout_s: float = 5.0) -> str:
    if res.status == "error":
        return f"Execution error: {res.error}"
    if res.status == "timeout":
        t = int(timeout_s) if float(timeout_s).is_integer() else timeout_s
        return f"Execution timed out after {t} seconds."
    if res.status == "ok" and res.n_rows == 0:
        return "The query executed but returned 0 rows."
    raise ValueError(f"no feedback for a non-triggering result: {res!r}")


def select_final(original: str, attempts: list[tuple[str, ExecResult]],
                 retry_on_empty: bool) -> tuple[str, int]:
    """attempts[0] is the original. Returns (sql, attempt index)."""
    if not attempts:
        return original, 0
    ok = [i for i, (_, r) in enumerate(attempts) if r.status == "ok"]
    if retry_on_empty:
        nonempty = [i for i in ok if r_rows(attempts[i][1]) >= 1]
        if nonempty:
            return attempts[nonempty[-1]][0], nonempty[-1]
    if ok:
        return attempts[ok[-1]][0], ok[-1]
    return original, 0


def r_rows(res: ExecResult) -> int:
    return res.n_rows if res.n_rows is not None else 0


# ---------------------------------------------------------------- I/O loop
@dataclass(frozen=True)
class EGSCInput:
    """Projection of Example without gold SQL (Req 16.3)."""
    example_id: int
    question: str
    db_id: str

    @classmethod
    def from_example(cls, ex: Example) -> "EGSCInput":
        return cls(ex.example_id, ex.question, ex.db_id)


@dataclass
class EGSCStats:
    n_examples: int = 0
    n_skipped: int = 0
    n_triggered: int = 0
    n_llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw_path: str = ""
    preds_path: str = ""
    final_sqls: list[str] = field(default_factory=list)


def check_run_name(cfg: RunConfig) -> None:
    expected = {PRIMARY_RUN: False, EMPTY_RUN: True}
    if cfg.run_name not in expected:
        raise ValueError(f"EGSC run_name must be one of {sorted(expected)}, got {cfg.run_name!r}")
    if expected[cfg.run_name] != cfg.retry_on_empty:
        raise ValueError(
            f"run_name {cfg.run_name!r} requires retry_on_empty={expected[cfg.run_name]}, "
            f"got {cfg.retry_on_empty}")


def _exec_dict(r: ExecResult) -> dict:
    return asdict(r)


def _load_done(raw_path: Path) -> dict[int, dict]:
    done: dict[int, dict] = {}
    if not raw_path.exists():
        return done
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                done.setdefault(int(rec["example_id"]), rec)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue  # partial trailing line from an interrupted write
    return done


def _row_count_exec(executor, db: Path, sql: str, timeout_s: float) -> ExecResult:
    """Execute under the EGSC timeout; n_rows is needed for the zero-row rule."""
    return executor(db, sql, timeout_s)


def run_egsc(cfg: RunConfig, examples: list[Example], original_preds: list[str],
             client: LLMClient,
             executor: Callable[[Path, str, float], ExecResult] = execute_sql,
             db_resolver: Callable[[str], Path] | None = None) -> EGSCStats:
    check_run_name(cfg)
    if len(examples) != len(original_preds):
        raise ValueError("examples and original_preds differ in length")
    resolve = db_resolver or data_mod.db_path
    inputs = [EGSCInput.from_example(ex) for ex in examples]  # gold never read after this

    out_dir = Path(cfg.output_dir())
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw.jsonl"
    preds_path = out_dir / "preds.txt"
    done = _load_done(raw_path)
    stats = EGSCStats(n_examples=len(inputs), raw_path=str(raw_path), preds_path=str(preds_path))
    schema_cache: dict[str, str] = {}
    to = cfg.egsc_exec_timeout_s

    with open(raw_path, "a", encoding="utf-8") as log:
        for inp, orig in zip(inputs, original_preds):
            if inp.example_id in done:
                stats.n_skipped += 1
                continue
            db = Path(resolve(inp.db_id))
            if inp.db_id not in schema_cache:
                schema_cache[inp.db_id] = serialize_schema(db, cfg.sample_rows, cfg.value_trunc)
            base = build_messages(inp.question, schema_cache[inp.db_id], cfg.prompt_version)

            r0 = _row_count_exec(executor, db, orig, to)
            attempts: list[tuple[str, ExecResult]] = [(orig, r0)]
            log_attempts = [{"k": 0, "sql": orig, "exec": _exec_dict(r0)}]
            cur, res, k = orig, r0, 0
            triggered = needs_retry(r0, cfg.retry_on_empty)
            while needs_retry(res, cfg.retry_on_empty) and k < cfg.egsc_max_retries:
                k += 1
                fb = feedback_text(res, to)
                llm = client.complete(build_retry_messages(base, cur, fb), cfg)
                if llm.status == "ok":
                    sql, status = extract_sql(llm.text, llm.finish_reason)
                else:
                    sql, status = PLACEHOLDER_SQL, "missing"
                sql = normalize_sql(sql) or PLACEHOLDER_SQL
                res = _row_count_exec(executor, db, sql, to)
                attempts.append((sql, res))
                stats.n_llm_calls += 1
                stats.prompt_tokens += llm.prompt_tokens or 0
                stats.completion_tokens += llm.completion_tokens or 0
                log_attempts.append({
                    "k": k, "sql": sql, "extraction_status": status, "feedback": fb,
                    "exec": _exec_dict(res), "llm_status": llm.status, "llm_error": llm.error,
                    "prompt_tokens": llm.prompt_tokens, "completion_tokens": llm.completion_tokens,
                    "latency_s": llm.latency_s, "finish_reason": llm.finish_reason,
                })
                cur = sql
            final_sql, idx = select_final(orig, attempts, cfg.retry_on_empty)
            rec = {
                "example_id": inp.example_id, "db_id": inp.db_id, "run_name": cfg.run_name,
                "config_fingerprint": cfg.fingerprint(), "retry_on_empty": cfg.retry_on_empty,
                "original_sql": orig, "triggered": triggered, "n_llm_calls": k,
                "attempts": log_attempts, "final_sql": final_sql, "final_attempt": idx,
            }
            log.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log.flush()
            done[inp.example_id] = rec

    finals = []
    for inp in inputs:
        rec = done[inp.example_id]
        finals.append(rec["final_sql"])
        if rec.get("triggered"):
            stats.n_triggered += 1
    write_predictions(preds_path, finals)
    stats.final_sqls = finals
    return stats
