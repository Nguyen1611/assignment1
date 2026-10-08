"""Shared model-notebook pipeline: run → predictions → evaluate → CLI check → provider report."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import data, env, evaluate, report, runner
from .config import RunConfig

USAGE_KEYS = ("prompt_tokens", "completion_tokens", "reasoning_tokens", "latency_s",
              "provider", "finish_reason", "reasoning_missing")


@dataclass
class PipelineResult:
    stats: runner.RunStats
    eval: evaluate.EvalSummary
    cli: dict
    mismatches: list
    providers: pd.DataFrame
    pred_path: Path
    gold_path: Path
    output_dir: Path
    n_missing: int = 0
    extra: dict = field(default_factory=dict)


def usage_rows(examples, state: runner.LogState) -> list[dict]:
    """Per-example usage fields from the first successful record (empty dict if none)."""
    out = []
    for ex in examples:
        rec = state.successes.get(ex.example_id) or {}
        out.append({k: rec.get(k) for k in USAGE_KEYS if k in rec})
    return out


def run_model_pipeline(cfg: RunConfig, client, endpoints: list[dict] | None = None,
                       examples=None, run_official: bool = True,
                       record_env: bool = True, show_progress: bool = True) -> PipelineResult:
    """Full pipeline for one Run_Config. `cfg.limit` controls subset and output roots."""
    limit = cfg.limit
    examples = examples if examples is not None else data.load_examples(limit)
    stats = runner.run(cfg, examples, client, show_progress=show_progress)
    print(f"[pipeline] run: {stats}")

    log_path = runner.default_log_path(cfg)
    state = runner.load_log(log_path)
    sqls, extraction, n_missing = runner.build_prediction_list(examples, state)
    out_dir = cfg.output_dir()
    pred_path = out_dir / "preds.txt"
    runner.write_predictions(pred_path, sqls)

    # Per design Directory Layout: everything for a run lives in outputs/<run_name>/.
    res_dir = out_dir
    gold = data.gold_lines(limit)[:len(examples)]
    ev = evaluate.evaluate_predictions(sqls, gold, extraction, usage_rows(examples, state),
                                       res_dir, run_name=cfg.run_name)
    gold_path = data.write_gold(out_dir / "gold.txt", len(examples))

    cli, mismatches = {}, []
    if run_official:
        cli = evaluate.run_official_cli(gold_path, pred_path)
        mismatches = evaluate.compare_totals(ev.table, cli)
        evaluate.record_cli_check(res_dir, mismatches)
        if mismatches:
            raise AssertionError(f"wrapper vs Official_CLI totals differ: {mismatches}")
        print("[pipeline] wrapper totals match the Official_CLI.")

    records = list(state.successes.values())
    prov = report.provider_report(records, endpoints or [], cfg.provider_order)
    prov.to_csv(res_dir / "providers.csv", index=False)
    if record_env:
        # Count from the log (not stats) so resumed runs report all served calls.
        served = Counter(str(r.get("provider") or "unknown") for r in records)
        env.record_models(cfg.model_id, served, endpoints or [], limit=limit)
    return PipelineResult(stats, ev, cli, mismatches, prov, pred_path, gold_path, res_dir,
                          n_missing)
