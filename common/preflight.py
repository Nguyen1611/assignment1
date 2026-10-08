"""Reasoning-trace preflight: checks, report, and full-run gate (Req 21)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from .config import RunConfig
from .data import Example
from .extract import extract_sql

if TYPE_CHECKING:
    from .llm_client import LLMClient

CHECKS = ("api_ok", "reasoning_nonempty", "reasoning_raw_not_summary",
          "reasoning_length_consistent", "reasoning_tokens_positive",
          "sql_extractable", "provider_matches_pin")

RAW_SOURCES = {"details_text", "reasoning_field", "inline_think"}
_CHECK_DETAIL = {
    "api_ok": "status == 'ok'",
    "reasoning_nonempty": "non-empty raw reasoning text",
    "reasoning_raw_not_summary": "raw reasoning, not summary/encrypted-only",
    "reasoning_length_consistent": "len(reasoning)/reasoning_tokens >= min_chars_per_token",
    "reasoning_tokens_positive": "reasoning_tokens > 0",
    "sql_extractable": "Extraction_Status in {ok, fallback}",
    "provider_matches_pin": "served provider within cfg.provider_order",
}


class PreflightRequiredError(RuntimeError):
    """No passing, current preflight report exists (Req 21.8)."""

    def __init__(self, report_path: Path, reason: str = ""):
        self.report_path = Path(report_path)
        super().__init__(
            f"A passing, current preflight report is required before the full run: "
            f"{self.report_path}" + (f" ({reason})" if reason else "")
            + ". Run preflight.run_preflight(cfg, examples, client) first."
        )


class PreflightFailedError(RuntimeError):
    """Preflight ran and at least one check failed (Req 21.7)."""

    def __init__(self, report: "PreflightReport", report_path: Path):
        self.report = report
        self.report_path = Path(report_path)
        lines = [f"Preflight FAILED (report: {self.report_path}); "
                 f"providers served: {report.providers_served}"]
        for c in report.checks:
            if not c["passed"]:
                lines.append(f"  - {c['name']}: failing ids {c['failing_ids']}")
        lines.append("Change provider_order or the Run_Config and rerun the preflight.")
        super().__init__("\n".join(lines))


@dataclass
class PreflightReport:
    config_fingerprint: str
    run_config: dict
    provider_order: list[str] | None
    providers_served: list[str]
    example_ids: list[int]
    checks: list[dict]
    passed: bool
    preflight_run_id: str
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PreflightReport":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "PreflightReport":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _pin(cfg: RunConfig) -> set[str]:
    return {p.lower() for p in (cfg.provider_order or ())}


def _record_checks(rec: dict, cfg: RunConfig, min_chars_per_token: float) -> dict[str, bool]:
    reasoning = rec.get("reasoning") or ""
    tokens = rec.get("reasoning_tokens") or 0
    types = set(rec.get("reasoning_detail_types") or [])
    summary_like = bool(types & {"reasoning.summary", "reasoning.encrypted"})
    raw_ok = rec.get("reasoning_source") in RAW_SOURCES and (
        not summary_like or "reasoning.text" in types)
    ext = rec.get("extraction_status")
    if ext is None:
        _, ext = extract_sql(rec.get("text"), rec.get("finish_reason"))
    provider = rec.get("provider")
    return {
        "api_ok": rec.get("status") == "ok",
        "reasoning_nonempty": reasoning.strip() != "",
        "reasoning_raw_not_summary": raw_ok,
        "reasoning_length_consistent": tokens > 0 and len(reasoning) / tokens >= min_chars_per_token,
        "reasoning_tokens_positive": tokens > 0,
        "sql_extractable": ext in ("ok", "fallback"),
        "provider_matches_pin": provider is not None and provider.lower() in _pin(cfg),
    }


def check_records(records: list[dict], cfg: RunConfig,
                  min_chars_per_token: float = 1.5) -> list[dict]:
    """Pure: one {name, passed, failing_ids, detail} per check (Req 21.2-21.6).

    A check passes only if every record passes it; an empty record list fails all."""
    failing: dict[str, list[int]] = {c: [] for c in CHECKS}
    for rec in records:
        for name, ok in _record_checks(rec, cfg, min_chars_per_token).items():
            if not ok:
                failing[name].append(rec.get("example_id"))
    return [{"name": c, "passed": bool(records) and not failing[c],
             "failing_ids": failing[c], "detail": _CHECK_DETAIL[c]} for c in CHECKS]


def report_path(cfg: RunConfig) -> Path:
    return cfg.output_dir() / "preflight.json"


def raw_log_path(cfg: RunConfig) -> Path:
    return cfg.output_dir() / "preflight_raw.jsonl"


def sample_examples(cfg: RunConfig, examples: list[Example]) -> list[Example]:
    """First min(preflight_n, N) examples (Req 21.1)."""
    return list(examples[:min(cfg.preflight_n, len(examples))])


def run_preflight(cfg: RunConfig, examples: list[Example], client: "LLMClient",
                  raise_on_fail: bool = True, show_progress: bool = False) -> PreflightReport:
    from . import runner

    sample = sample_examples(cfg, examples)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    log = raw_log_path(cfg)
    runner.run(cfg, sample, client, log_path=log, phase="preflight",
               require_preflight=False, skip_completed=False,
               extra_fields={"preflight_run_id": run_id}, show_progress=show_progress)
    records = []
    with open(log, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("preflight_run_id") == run_id:
                records.append(rec)
    checks = check_records(records, cfg)
    served = sorted({r["provider"] for r in records if r.get("provider")})
    report = PreflightReport(
        config_fingerprint=cfg.fingerprint(),
        run_config=cfg.to_dict(),
        provider_order=list(cfg.provider_order) if cfg.provider_order else None,
        providers_served=served,
        example_ids=[e.example_id for e in sample],
        checks=checks,
        passed=all(c["passed"] for c in checks),
        preflight_run_id=run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    path = report_path(cfg)
    report.save(path)
    if not report.passed:
        err = PreflightFailedError(report, path)
        print(str(err))
        if raise_on_fail:
            raise err
    return report


def load_report(cfg: RunConfig) -> PreflightReport | None:
    path = report_path(cfg)
    if not path.is_file():
        return None
    try:
        return PreflightReport.load(path)
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def is_current(report: PreflightReport, cfg: RunConfig) -> bool:
    """Pure: passed, same fingerprint, served providers within the pin (Req 21.11)."""
    served = {p.lower() for p in report.providers_served}
    return (bool(report.passed)
            and report.config_fingerprint == cfg.fingerprint()
            and bool(served) and served <= _pin(cfg))


def require_passed(cfg: RunConfig) -> PreflightReport:
    path = report_path(cfg)
    report = load_report(cfg)
    if report is None:
        raise PreflightRequiredError(path, "missing")
    if not report.passed:
        raise PreflightRequiredError(path, "failed")
    if not is_current(report, cfg):
        raise PreflightRequiredError(path, "stale: Run_Config or provider changed")
    return report
