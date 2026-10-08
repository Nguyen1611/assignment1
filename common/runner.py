"""Resumable, thread-pooled runner: raw JSONL logs and prediction files (Req 5.5, 7, 9)."""
from __future__ import annotations

import dataclasses
import functools
import json
import os
import re
import threading
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import RunConfig
from .data import Example, db_path
from .extract import PLACEHOLDER_SQL, extract_sql
from .prompts import build_messages, messages_sha256
from .schema import serialize_schema

if TYPE_CHECKING:  # avoid a hard runtime dependency on llm_client
    from .llm_client import LLMClient, LLMResult


class ConfigMismatchError(RuntimeError):
    """A raw log contains successes produced under a different Run_Config."""

    def __init__(self, log_path: Path, expected: str, found: set[str]):
        self.log_path, self.expected, self.found = log_path, expected, found
        super().__init__(
            f"{log_path} holds records with config fingerprint(s) {sorted(found)}, "
            f"expected {expected}. Use a new run_name or delete the log."
        )


class NonRetryableBatchError(RuntimeError):
    """The whole first batch failed with a non-retryable 4xx error."""


@dataclass
class LogState:
    successes: dict[int, dict] = field(default_factory=dict)
    bad_lines: list[int] = field(default_factory=list)
    n_lines: int = 0
    n_errors: int = 0


@dataclass
class RunStats:
    n_total: int
    n_skipped: int
    n_ok: int
    n_error: int
    providers: Counter
    bad_lines: list[int] = field(default_factory=list)
    stopped_early: bool = False


def default_log_path(cfg: RunConfig, phase: str = "full") -> Path:
    name = "raw.jsonl" if phase == "full" else f"raw_{phase}.jsonl"
    return cfg.output_dir() / name


def _result_dict(result: Any) -> dict:
    if dataclasses.is_dataclass(result):
        return dataclasses.asdict(result)
    if isinstance(result, dict):
        return dict(result)
    return dict(vars(result))


def make_record(ex: Example, cfg: RunConfig, messages: list[dict], result: "LLMResult",
                phase: str, extra_fields: dict | None = None) -> dict:
    """Pure: one raw-log record (Req 7.1, 5.5)."""
    rec: dict = {
        "example_id": ex.example_id,
        "db_id": ex.db_id,
        "question": ex.question,
        "phase": phase,
        "run_name": cfg.run_name,
        "model_id": cfg.model_id,
        "prompt_version": cfg.prompt_version,
        "config_fingerprint": cfg.fingerprint(),
        "run_config": cfg.to_dict(),
        "messages_sha256": messages_sha256(messages),
    }
    rec.update(_result_dict(result))
    if extra_fields:
        rec.update(extra_fields)
    return rec


def load_log(log_path: Path) -> LogState:
    """Parse a JSONL log; bad lines are recorded (1-based) and skipped; first success wins."""
    state = LogState()
    p = Path(log_path)
    if not p.exists():
        return state
    with open(p, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            state.n_lines += 1
            try:
                rec = json.loads(line)
                eid = int(rec["example_id"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                state.bad_lines.append(lineno)
                continue
            if rec.get("status") == "ok":
                state.successes.setdefault(eid, rec)
            else:
                state.n_errors += 1
    return state


_4XX_RE = re.compile(r"\b(4\d\d)\b")


def _is_non_retryable_4xx(rec: dict) -> bool:
    if rec.get("status") == "ok":
        return False
    code = rec.get("status_code") or rec.get("http_status")
    if code is None:
        m = _4XX_RE.search(str(rec.get("error") or ""))
        code = int(m.group(1)) if m else None
    try:
        code = int(code)
    except (TypeError, ValueError):
        return False
    return 400 <= code < 500 and code not in (408, 429)


def _check_preflight(cfg: RunConfig) -> None:
    from . import preflight  # lazy: implemented in a later task

    preflight.require_passed(cfg)


def run(cfg: RunConfig, examples: list[Example], client: "LLMClient",
        log_path: Path | None = None, phase: str = "full",
        require_preflight: bool | None = None, skip_completed: bool = True,
        extra_fields: dict | None = None, show_progress: bool = True) -> RunStats:
    if require_preflight is None:
        require_preflight = cfg.reasoning_enabled and phase == "full"
    if require_preflight:
        _check_preflight(cfg)

    log_path = Path(log_path) if log_path is not None else default_log_path(cfg, phase)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    state = load_log(log_path)
    if state.bad_lines:
        print(f"[runner] skipped unparseable lines in {log_path}: {state.bad_lines}")
    fp = cfg.fingerprint()
    found = {r.get("config_fingerprint") for r in state.successes.values()}
    if found - {fp}:
        raise ConfigMismatchError(log_path, fp, found - {fp})

    done = set(state.successes) if skip_completed else set()
    todo = [e for e in examples if e.example_id not in done]
    n_skipped = len(examples) - len(todo)

    @functools.lru_cache(maxsize=None)
    def schema_for(db_id: str) -> str:
        return serialize_schema(db_path(db_id), cfg.sample_rows, cfg.value_trunc)

    def work(ex: Example):
        msgs = build_messages(ex.question, schema_for(ex.db_id), cfg.prompt_version)
        return ex, msgs, client.complete(msgs, cfg)

    lock = threading.Lock()
    providers: Counter = Counter()
    n_ok = n_err = 0
    stopped = False
    first_batch = max(1, min(cfg.threads, len(todo)))
    first_results: list[dict] = []

    pbar = None
    if show_progress:
        from tqdm.auto import tqdm
        pbar = tqdm(total=len(examples), initial=n_skipped, desc=f"{cfg.run_name}[{phase}]")

    with open(log_path, "a", encoding="utf-8") as f, \
            ThreadPoolExecutor(max_workers=max(1, cfg.threads)) as pool:
        it = iter(todo)
        pending = set()
        # Submit only the first batch; more are submitted as results arrive.
        for ex in it:
            pending.add(pool.submit(work, ex))
            if len(pending) >= first_batch:
                break
        while pending:
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in finished:
                ex, msgs, result = fut.result()
                rec = make_record(ex, cfg, msgs, result, phase, extra_fields)
                line = json.dumps(rec, ensure_ascii=False) + "\n"
                with lock:
                    f.write(line)
                    f.flush()
                    os.fsync(f.fileno())
                if rec.get("status") == "ok":
                    n_ok += 1
                    providers[rec.get("provider")] += 1
                    if pbar:
                        pbar.update(1)
                else:
                    n_err += 1
                if len(first_results) < first_batch:
                    first_results.append(rec)
                    if len(first_results) == first_batch and all(
                            _is_non_retryable_4xx(r) for r in first_results):
                        stopped = True
            if stopped:
                for p in pending:
                    p.cancel()
                break
            for ex in it:
                pending.add(pool.submit(work, ex))
                if len(pending) >= first_batch:
                    break
    if pbar:
        pbar.close()

    stats = RunStats(len(examples), n_skipped, n_ok, n_err, providers,
                     bad_lines=list(state.bad_lines), stopped_early=stopped)
    if stopped:
        raise NonRetryableBatchError(
            f"All {first_batch} first-batch calls failed with non-retryable 4xx errors; "
            f"stopping. First error: {first_results[0].get('error')}"
        )
    return stats


def write_predictions(path: Path, sqls: list[str]) -> None:
    for i, s in enumerate(sqls):
        if not s or not s.strip() or "\n" in s or "\r" in s or "\t" in s:
            raise ValueError(f"prediction {i} is empty or contains a newline/tab: {s!r}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(sqls) + "\n")


def read_predictions(path: Path) -> list[str]:
    with open(path, encoding="utf-8", newline="") as f:
        return f.read().splitlines()


def build_prediction_list(examples: list[Example], log_state: LogState
                          ) -> tuple[list[str], list[dict], int]:
    """(sqls in example order, extraction rows, n_missing_records). Missing → Placeholder_SQL."""
    sqls: list[str] = []
    rows: list[dict] = []
    n_missing = 0
    for ex in examples:
        rec = log_state.successes.get(ex.example_id)
        if rec is None:
            n_missing += 1
            sql, status = PLACEHOLDER_SQL, "no_record"
        else:
            sql, status = extract_sql(rec.get("text"), rec.get("finish_reason"))
        sqls.append(sql)
        rows.append({"example_id": ex.example_id, "pred_sql": sql,
                     "extraction_status": status, "has_record": rec is not None})
    if n_missing:
        print(f"[runner] {n_missing} example(s) had no successful record; wrote Placeholder_SQL")
    return sqls, rows, n_missing
