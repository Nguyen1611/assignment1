"""Report generation: table/figure export, Task 1 tables, provider report, EGSC tables.

All table builders are pure (DataFrame in, DataFrame out) except the explicit
writers `export_table` / `save_figure` and the optional disk loading in
`load_per_example` / `egsc_main_table`.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path

import pandas as pd

from common import config
from common.evaluate import LEVELS, aggregate

DIFF_LEVELS: tuple[str, ...] = tuple(l for l in LEVELS if l != "all")
EXTRACTION_STATUSES: tuple[str, ...] = ("ok", "fallback", "missing", "truncated")
EGSC_PRIMARY_RUN = "qwen25_7b_egsc"
EGSC_EMPTY_RUN = "qwen25_7b_egsc_empty"
ABLATION_LABEL = "ablation: retry_on_empty=true"


# ---------------------------------------------------------------- export
def _latex_escape(s: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}")):
        s = s.replace(a, b) if a != "\\" else s.replace(a, "\x00")
    return s.replace("\x00", r"\textbackslash{}")


def export_table(df: pd.DataFrame, name: str, caption: str, out_dir: Path) -> dict[str, Path]:
    """Write `<name>.csv`, `<name>.md` (italic caption line above), `<name>.tex`
    (caption + label `tab:<name>`). Returns {ext: path}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {ext: out_dir / f"{name}.{ext}" for ext in ("csv", "md", "tex")}
    df.to_csv(paths["csv"], index=False)
    try:
        md_body = df.to_markdown(index=False)
    except ImportError:  # tabulate missing
        md_body = df.to_string(index=False)
    paths["md"].write_text(f"*{caption}*\n\n{md_body}\n", encoding="utf-8")
    shown = df.copy()
    shown.columns = [_latex_escape(str(c)) for c in shown.columns]
    for c in shown.columns:
        if shown[c].dtype == object:
            shown[c] = shown[c].map(lambda v: _latex_escape(str(v)) if isinstance(v, str) else v)
    styler = shown.style.hide(axis="index").format(precision=3, na_rep="--")
    tex = styler.to_latex(caption=_latex_escape(caption), label=f"tab:{name}",
                          hrules=True, position="htbp")
    paths["tex"].write_text(tex, encoding="utf-8")
    return paths


def save_figure(fig, name: str, out_dir: Path) -> dict[str, Path]:
    """Save a matplotlib figure as PNG (300 dpi) and PDF."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"png": out_dir / f"{name}.png", "pdf": out_dir / f"{name}.pdf"}
    fig.savefig(paths["png"], dpi=300, bbox_inches="tight")
    fig.savefig(paths["pdf"], bbox_inches="tight")
    return paths


# ---------------------------------------------------------------- helpers
def _as_dict(obj) -> dict:
    if obj is None:
        return {}
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Mapping):
        return dict(obj)
    return dict(vars(obj))


def load_per_example(run_name: str, limit: int | None = None) -> pd.DataFrame:
    """Read `outputs[_limitN]/<run_name>/per_example.csv`."""
    return pd.read_csv(config.out_root(limit) / run_name / "per_example.csv")


def _rows(df: pd.DataFrame) -> list[dict]:
    return df[["difficulty", "ex", "em"]].to_dict("records")


# ---------------------------------------------------------------- provider report (11.1)
def provider_report(records: list[dict], endpoints: list[dict],
                    provider_order: Sequence[str] | None = None) -> pd.DataFrame:
    """Served-provider call counts joined with endpoint quantization (Req 12.5, 12.6, 22.5).

    The pin is `provider_order` if given, else taken from the records' `run_config`.
    Columns: provider, calls, share, quantization, in_pin, flag. `flag` is
    "outside pin" for providers not in the pin (case-insensitive), "" otherwise.
    """
    ok = [r for r in records if r.get("status", "ok") == "ok"]
    counts = Counter(str(r.get("provider") or "unknown") for r in ok)
    if provider_order is None:
        for r in records:
            po = (r.get("run_config") or {}).get("provider_order")
            if po:
                provider_order = po
                break
    pin = {p.lower() for p in provider_order} if provider_order else None
    quant: dict[str, object] = {}
    for ep in endpoints or []:
        name = ep.get("provider_name") or ep.get("provider") or ep.get("name")
        if name:
            quant[str(name).lower()] = ep.get("quantization")
    total = sum(counts.values())
    rows = []
    for prov, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        in_pin = True if pin is None else prov.lower() in pin
        rows.append({
            "provider": prov, "calls": int(n), "share": n / total if total else 0.0,
            "quantization": quant.get(prov.lower()) or "not reported",
            "in_pin": in_pin, "flag": "" if in_pin else "outside pin",
        })
    return pd.DataFrame(rows, columns=["provider", "calls", "share", "quantization",
                                       "in_pin", "flag"])


# ---------------------------------------------------------------- Task 1 tables (12.1)
def cost_usd(prompt_tokens, completion_tokens, model_id: str) -> float:
    """Pure. (prompt*input_per_m + completion*output_per_m) / 1e6.
    completion_tokens already includes reasoning tokens. Returns NaN when the
    model has no price entry or the prices are not filled in (never silently 0)."""
    p = config.PRICES.get(model_id) or {}
    pin, pout = p.get("input_per_m"), p.get("output_per_m")
    if pin is None or pout is None:
        return float("nan")
    return (float(prompt_tokens or 0) * pin + float(completion_tokens or 0) * pout) / 1e6


def price_note(model_ids: Sequence[str]) -> str:
    """Caption fragment with price source/date per model, or a note if missing (Req 13.7)."""
    parts = []
    for m in dict.fromkeys(model_ids):
        p = config.PRICES.get(m)
        if not p or p.get("input_per_m") is None or p.get("output_per_m") is None:
            parts.append(f"{m}: price not set in config.PRICES (cost shown as NaN)")
        else:
            parts.append(f"{m}: ${p['input_per_m']}/M in, ${p['output_per_m']}/M out "
                         f"({p.get('source')}, {p.get('date')})")
    return "Prices: " + "; ".join(parts) + "."


def metadata_table(models: Mapping[str, Mapping], access: str = "API (OpenRouter)") -> pd.DataFrame:
    """Model metadata (Req 13.1). `models` maps model_id -> {"providers": {name: {"calls",
    "quantization"}}} (the `models` section of Environment_Metadata)."""
    rows = []
    for mid, info in models.items():
        meta = config.MODEL_META.get(mid, {})
        provs = (info or {}).get("providers", {}) or {}
        rows.append({
            "model_id": mid,
            "params": meta.get("params"),
            "type": meta.get("type"),
            "quantization": ", ".join(sorted({str(v.get("quantization") or "not reported")
                                              for v in provs.values()})) or "not reported",
            "provider": ", ".join(f"{p} ({v.get('calls', 0)})" for p, v in provs.items()) or "--",
            "access": access,
        })
    return pd.DataFrame(rows, columns=["model_id", "params", "type", "quantization",
                                       "provider", "access"])


def accuracy_table(per_example: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """EX/EM overall and per difficulty with counts (Req 13.2); one row per model label."""
    rows = []
    for label, df in per_example.items():
        agg = aggregate(_rows(df))
        row = {"model": label}
        for lvl in LEVELS:
            row[f"n_{lvl}"] = int(agg.loc[lvl, "count"])
            row[f"ex_{lvl}"] = float(agg.loc[lvl, "ex"])
            row[f"em_{lvl}"] = float(agg.loc[lvl, "em"])
        rows.append(row)
    return pd.DataFrame(rows)


def _mean(df: pd.DataFrame, col: str) -> float:
    if col not in df:
        return float("nan")
    s = pd.to_numeric(df[col], errors="coerce")
    return float(s.mean()) if s.notna().any() else float("nan")


def efficiency_table(per_example: Mapping[str, pd.DataFrame],
                     model_ids: Mapping[str, str]) -> pd.DataFrame:
    """Mean latency/tokens and total cost per model (Req 13.3, 13.7)."""
    rows = []
    for label, df in per_example.items():
        mid = model_ids[label]
        pt = pd.to_numeric(df.get("prompt_tokens", pd.Series(dtype=float)), errors="coerce").fillna(0)
        ct = pd.to_numeric(df.get("completion_tokens", pd.Series(dtype=float)), errors="coerce").fillna(0)
        cost = cost_usd(float(pt.sum()), float(ct.sum()), mid)
        rows.append({
            "model": label, "n": len(df),
            "mean_latency_s": _mean(df, "latency_s"),
            "mean_input_tokens": _mean(df, "prompt_tokens"),
            "mean_output_tokens": _mean(df, "completion_tokens"),
            "mean_reasoning_tokens": _mean(df, "reasoning_tokens"),
            "total_cost_usd": cost,
        })
    return pd.DataFrame(rows)


def invalid_table(per_example: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Count and rate of each Extraction_Status and of non-executable predictions (Req 13.4)."""
    rows = []
    for label, df in per_example.items():
        n = len(df)
        row = {"model": label, "n": n}
        st = df["extraction_status"].fillna("unknown") if "extraction_status" in df else pd.Series(dtype=str)
        for s in EXTRACTION_STATUSES:
            k = int((st == s).sum())
            row[f"{s}_count"] = k
            row[f"{s}_rate"] = k / n if n else 0.0
        k = int((df["exec_status"] != "ok").sum()) if "exec_status" in df else 0
        row["non_exec_count"] = k
        row["non_exec_rate"] = k / n if n else 0.0
        row.update(parse_stats(df))
        rows.append(row)
    return pd.DataFrame(rows)


def parse_stats(df: pd.DataFrame) -> dict:
    """Predictions the official Spider parser cannot parse (scored EM=0 by the official script).

    unparseable_count/rate, EM on the parseable subset, and EX on the unparseable subset
    (shows how many EM=0 parse failures are nonetheless execution-correct).
    """
    n = len(df)
    if "parse_ok" not in df or n == 0:
        return {"unparseable_count": 0, "unparseable_rate": float("nan"),
                "em_parseable": float("nan"), "ex_unparseable": float("nan")}
    ok = df["parse_ok"].astype(str).str.lower().isin(["true", "1"])
    k = int((~ok).sum())
    em = pd.to_numeric(df["em"], errors="coerce")
    ex = pd.to_numeric(df["ex"], errors="coerce")
    return {
        "unparseable_count": k,
        "unparseable_rate": k / n,
        "em_parseable": float(em[ok].mean()) if ok.any() else float("nan"),
        "ex_unparseable": float(ex[~ok].mean()) if k else float("nan"),
    }


def environment_table(env: Mapping) -> pd.DataFrame:
    """Two-column Item/Value methodology table from Environment_Metadata (Req 22.5, 22.8)."""
    items: list[tuple[str, object]] = []
    m = env.get("machine", {}) or {}
    for k, lab in (("os", "OS"), ("kernel", "Kernel"), ("is_wsl", "Running under WSL"),
                   ("distro", "Distribution"), ("cpu_model", "CPU"),
                   ("logical_cpus", "Logical cores"), ("ram_gib", "RAM (GiB)"),
                   ("ram_note", "RAM note"), ("python", "Python")):
        if k in m:
            items.append((lab, m[k]))
    for pkg, ver in (env.get("packages", {}) or {}).items():
        items.append((f"{pkg} version", ver if ver is not None else "not installed"))
    items.append(("Official script commit", env.get("official_commit") or "not available"))
    inf = env.get("inference", {}) or {}
    if inf:
        items.append(("Inference", f"{inf.get('access', 'api')} via {inf.get('service')} "
                                   f"on remote provider hardware not controlled by the student"))
        items.append(("Local machine role", inf.get("local_role")))
    for mid, info in (env.get("models", {}) or {}).items():
        provs = (info or {}).get("providers", {}) or {}
        desc = ", ".join(f"{p} ({v.get('calls')} calls, quant: {v.get('quantization') or 'not reported'})"
                         for p, v in provs.items())
        items.append((f"Providers for {mid}", desc or "--"))
    for r in env.get("runs", []) or []:
        items.append((f"Run {r.get('name')}",
                      f"{r.get('started_at')} -> {r.get('finished_at')} ({r.get('status')})"))
    return pd.DataFrame([{"item": k, "value": "" if v is None else str(v)} for k, v in items],
                        columns=["item", "value"])


def preflight_table(report) -> pd.DataFrame:
    """Preflight provider, overall status, and per-check results (Req 21.12)."""
    d = _as_dict(report)
    rows = [
        {"item": "provider(s) served", "result": ", ".join(d.get("providers_served") or []) or "--",
         "failing_ids": ""},
        {"item": "example ids", "result": ", ".join(map(str, d.get("example_ids") or [])),
         "failing_ids": ""},
        {"item": "overall", "result": "PASS" if d.get("passed") else "FAIL", "failing_ids": ""},
    ]
    for c in d.get("checks") or []:
        rows.append({"item": c.get("name"), "result": "PASS" if c.get("passed") else "FAIL",
                     "failing_ids": ", ".join(map(str, c.get("failing_ids") or []))})
    if d.get("timestamp"):
        rows.append({"item": "timestamp", "result": d["timestamp"], "failing_ids": ""})
    return pd.DataFrame(rows, columns=["item", "result", "failing_ids"])


# ---------------------------------------------------------------- EGSC (15.8)
def _level_row(label: str, df: pd.DataFrame) -> dict:
    agg = aggregate(_rows(df))
    row = {"system": label}
    for lvl in LEVELS:
        row[f"ex_{lvl}"] = float(agg.loc[lvl, "ex"])
        row[f"em_{lvl}"] = float(agg.loc[lvl, "em"])
    row["n"] = int(agg.loc["all", "count"])
    row.update(parse_stats(df))
    return row


def egsc_main_table(orig: pd.DataFrame, primary: pd.DataFrame | None = None,
                    limit: int | None = None) -> pd.DataFrame:
    """Original vs Primary_EGSC EX/EM overall and per difficulty (Req 17.1, 17.9).
    When `primary` is None it is read only from `outputs/qwen25_7b_egsc/per_example.csv`."""
    if primary is None:
        primary = load_per_example(EGSC_PRIMARY_RUN, limit)
    return pd.DataFrame([_level_row("original", orig), _level_row("primary EGSC", primary)])


def egsc_ablation_table(orig: pd.DataFrame, variant: pd.DataFrame | None = None,
                        limit: int | None = None) -> pd.DataFrame:
    """Original vs EGSC_Empty_Variant, labelled as an ablation (Req 17.7)."""
    if variant is None:
        variant = load_per_example(EGSC_EMPTY_RUN, limit)
    return pd.DataFrame([_level_row("original", orig),
                         _level_row(f"EGSC ({ABLATION_LABEL})", variant)])


def ablation_caption(base: str) -> str:
    return f"{base} [{ABLATION_LABEL}]"


def egsc_counts(orig_rows, final_rows, triggered) -> dict:
    """Pure. orig_rows/final_rows: sequences of EX values (0/1); triggered: n_retries
    (int) or bools per example. fixed = 0->1, broken = 1->0, unchanged = equal."""
    o = [int(x) for x in orig_rows]
    f = [int(x) for x in final_rows]
    t = [bool(int(x)) if not isinstance(x, bool) else x for x in triggered]
    if not (len(o) == len(f) == len(t)):
        raise ValueError("orig_rows, final_rows and triggered must have equal length")
    fixed = sum(a == 0 and b == 1 for a, b in zip(o, f))
    broken = sum(a == 1 and b == 0 for a, b in zip(o, f))
    return {"n": len(o), "triggered": sum(t), "fixed": fixed, "broken": broken,
            "unchanged": len(o) - fixed - broken}


def mcnemar_exact(orig_ex, final_ex) -> dict:
    """Exact (binomial) McNemar test on paired EX. b = 0->1 (fixed), c = 1->0 (broken)."""
    o = [int(x) for x in orig_ex]
    f = [int(x) for x in final_ex]
    if len(o) != len(f):
        raise ValueError("orig_ex and final_ex must have equal length")
    b = sum(a == 0 and y == 1 for a, y in zip(o, f))
    c = sum(a == 1 and y == 0 for a, y in zip(o, f))
    if b + c == 0:
        p = 1.0
    else:
        from scipy.stats import binomtest
        p = float(binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue)
    return {"b": b, "c": c, "n_discordant": b + c, "p": min(1.0, p)}
