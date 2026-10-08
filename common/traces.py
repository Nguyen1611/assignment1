"""Task 2: reasoning-trace feature extraction (Req 14) and statistics (Req 15)."""
from __future__ import annotations

import json
import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from common import config
from common.data import Example

SELF_CORRECTION_MARKERS: tuple[str, ...] = tuple(config.SELF_CORRECTION_MARKERS)

DIFFICULTY_ORDER: tuple[str, ...] = ("easy", "medium", "hard", "extra")
DIFFICULTY_RANK: dict[str, int] = {d: i + 1 for i, d in enumerate(DIFFICULTY_ORDER)}

# Feature columns used by the statistics helpers.
# length_tokens: Qwen3-tokenizer count of the trace text; length_chars: character count (robustness check).
FEATURES: tuple[str, ...] = (
    "length_tokens", "length_chars", "marker_count", "markers_per_1k", "sql_drafts",
    "table_coverage", "column_coverage",
)
# table_coverage is excluded from the logit: it is 1.0 for all but one trace on the full dev set
# (near-zero variance), which causes quasi-separation and forces the regularized fallback. It is
# still reported in the descriptive tables, the rank tests and the box plots.
# length_chars is excluded as it is collinear with length_tokens (robustness check only).
LOGIT_FEATURES: tuple[str, ...] = (
    "length_tokens", "markers_per_1k", "sql_drafts", "column_coverage",
)

_B = r"(?<![A-Za-z0-9_])"   # left word boundary
_E = r"(?![A-Za-z0-9_])"    # right word boundary


# =================================================================== features
@dataclass(frozen=True)
class TraceFeatures:
    example_id: int
    excluded: bool
    reason: str | None
    length_tokens: int | None
    length_chars: int | None
    provider_reasoning_tokens: int | None
    marker_count: int | None
    markers_per_1k: float | None
    sql_drafts: int | None
    table_coverage: float | None
    column_coverage: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def compile_marker_regex(markers: Sequence[str]) -> re.Pattern:
    """Alternation of markers; words escaped individually and joined by \\s+."""
    alts = []
    for m in sorted({m.strip() for m in markers if m and m.strip()}, key=len, reverse=True):
        alts.append(r"\s+".join(re.escape(w) for w in m.split()))
    if not alts:
        return re.compile(r"(?!x)x")  # matches nothing
    return re.compile(_B + "(?:" + "|".join(alts) + ")" + _E, re.IGNORECASE)


_DEFAULT_MARKER_RE = compile_marker_regex(SELF_CORRECTION_MARKERS)


def count_markers(trace: str, markers: Sequence[str] = SELF_CORRECTION_MARKERS) -> int:
    if not trace:
        return 0
    rx = _DEFAULT_MARKER_RE if tuple(markers) == SELF_CORRECTION_MARKERS else compile_marker_regex(markers)
    return len(rx.findall(trace))


_FENCE_RE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[^\n]*\n?(.*?)```", re.DOTALL)
_FENCE_BODY_RE = re.compile(r"\b(?:SELECT|WITH)\b", re.IGNORECASE)
_INLINE_SELECT_RE = re.compile(r"\bSELECT\b[^`]*?\bFROM\b")
_INLINE_WITH_RE = re.compile(r"\bWITH\s+\w+\s+AS\s*\(")


def count_sql_drafts(trace: str) -> int:
    """Fenced ```sql / bare ``` blocks containing SELECT/WITH, plus uppercase inline statements."""
    if not trace:
        return 0
    n = 0
    for m in _FENCE_RE.finditer(trace):
        lang = m.group(1).lower()
        if lang in ("", "sql", "sqlite") and _FENCE_BODY_RE.search(m.group(2)):
            n += 1
    outside = _FENCE_RE.sub(" ", trace)
    n += len(_INLINE_SELECT_RE.findall(outside))
    n += len(_INLINE_WITH_RE.findall(outside))
    return n


_KEY_RE = re.compile(r"^__(.+)__$")


def _name_maps(db_id: str, tables_json: Any) -> tuple[dict[str, str], dict[str, str]]:
    """Lower-cased original name → original name for tables and 'table.column' keys."""
    if isinstance(tables_json, (str, Path)):
        with open(tables_json, encoding="utf-8") as f:
            tables_json = json.load(f)
    if isinstance(tables_json, dict):
        entry = tables_json[db_id]
    else:
        entry = next(t for t in tables_json if t["db_id"] == db_id)
    tnames = entry["table_names_original"]
    tmap = {t.lower(): t for t in tnames}
    cmap = {}
    for tidx, cname in entry["column_names_original"]:
        if tidx < 0:
            continue
        cmap[f"{tnames[tidx].lower()}.{cname.lower()}"] = cname
    return tmap, cmap


def _walk_keys(node: Any, out: set[str]) -> None:
    if isinstance(node, str):
        if _KEY_RE.match(node):
            out.add(node)
    elif isinstance(node, dict):
        for v in node.values():
            _walk_keys(v, out)
    elif isinstance(node, (list, tuple)):
        for v in node:
            _walk_keys(v, out)


def gold_schema_items(gold_sql: str, db_id: str, api, tables_json) -> tuple[set[str], set[str]]:
    """(tables, columns) referenced by the parsed gold SQL, as original names. `*` excluded."""
    from common.data import db_path

    schema = api.Schema(api.get_schema(str(db_path(db_id))))
    parsed = api.get_sql(schema, gold_sql)
    keys: set[str] = set()
    _walk_keys(parsed, keys)
    tmap, cmap = _name_maps(db_id, tables_json)
    tables: set[str] = set()
    cols: set[str] = set()
    for k in keys:
        inner = _KEY_RE.match(k).group(1)
        if inner == "all":
            continue
        if "." in inner:
            t, c = inner.split(".", 1)
            cols.add(cmap.get(inner, c))
            tables.add(tmap.get(t, t))
        else:
            tables.add(tmap.get(inner, inner))
    return tables, cols


def coverage(trace: str, items: set[str]) -> float | None:
    if not items:
        return None
    trace = trace or ""
    hit = sum(
        1 for name in items
        if re.search(_B + re.escape(name) + _E, trace, re.IGNORECASE)
    )
    return hit / len(items)


def _exclusion_reason(record: dict) -> str | None:
    if record.get("reasoning_missing"):
        return "reasoning_missing"
    trace = record.get("reasoning")
    if trace is None or not str(trace).strip():
        return "empty_trace"
    return None


def extract_features(record: dict, example: Example, gold_items,
                     token_counter=None) -> TraceFeatures:
    """gold_items: (tables, columns) for this example, from gold_schema_items.

    Length is measured from the trace text (`token_counter`, default: Qwen3 tokenizer). The provider's
    `reasoning_tokens` is kept only as `provider_reasoning_tokens` (an estimate on some providers).
    """
    reason = _exclusion_reason(record)
    if reason is not None:
        return TraceFeatures(example.example_id, True, reason, None, None, None, None, None, None, None, None)
    if token_counter is None:
        from common.tokenizer import count_tokens as token_counter
    trace = str(record["reasoning"])
    tokens = max(int(token_counter(trace)), 1)
    ptok = record.get("reasoning_tokens")
    mc = count_markers(trace)
    tables, cols = gold_items
    return TraceFeatures(
        example_id=example.example_id, excluded=False, reason=None,
        length_tokens=tokens, length_chars=len(trace),
        provider_reasoning_tokens=None if ptok is None else int(ptok), marker_count=mc, markers_per_1k=mc / tokens * 1000,
        sql_drafts=count_sql_drafts(trace),
        table_coverage=coverage(trace, set(tables)),
        column_coverage=coverage(trace, set(cols)),
    )


# ================================================================= statistics
def rank_biserial(x: Sequence[float], y: Sequence[float]) -> float:
    """2*U_x/(n_x*n_y) - 1; positive when x tends to be larger."""
    from scipy.stats import mannwhitneyu

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    u = mannwhitneyu(x, y, alternative="two-sided").statistic
    return float(2 * u / (len(x) * len(y)) - 1)


def holm(pvals: Sequence[float]) -> list[float]:
    """Holm-adjusted p-values; NaN entries are passed through untouched."""
    from statsmodels.stats.multitest import multipletests

    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    if ok.any():
        out[ok] = multipletests(p[ok], method="holm")[1]
    return out.tolist()


def describe(df: pd.DataFrame, features: Sequence[str], by: str) -> pd.DataFrame:
    rows = []
    for f in features:
        for g, s in df.groupby(by, sort=True)[f]:
            s = pd.to_numeric(s, errors="coerce").dropna()
            q1, q3 = (s.quantile(0.25), s.quantile(0.75)) if len(s) else (np.nan, np.nan)
            rows.append({"feature": f, by: g, "count": int(len(s)),
                         "mean": s.mean(), "median": s.median(),
                         "q1": q1, "q3": q3, "iqr": q3 - q1})
    return pd.DataFrame(rows)


def _clean(df: pd.DataFrame, f: str, other: str) -> pd.DataFrame:
    d = df[[f, other]].copy()
    d[f] = pd.to_numeric(d[f], errors="coerce")
    return d.dropna()


def mann_whitney_table(df: pd.DataFrame, features: Sequence[str], group: str = "ex") -> pd.DataFrame:
    from scipy.stats import mannwhitneyu

    rows = []
    for f in features:
        d = _clean(df, f, group)
        x = d.loc[d[group] == 1, f].to_numpy()
        y = d.loc[d[group] == 0, f].to_numpy()
        if len(x) and len(y):
            res = mannwhitneyu(x, y, alternative="two-sided")
            u, p, r = float(res.statistic), float(res.pvalue), float(2 * res.statistic / (len(x) * len(y)) - 1)
        else:
            u = p = r = float("nan")
        rows.append({"feature": f, "n_correct": len(x), "n_incorrect": len(y),
                     "n_dropped": int(len(df) - len(d)), "U": u, "rank_biserial": r, "p_raw": p})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_raw"]) if len(out) else []
    return out


def kruskal_table(df: pd.DataFrame, features: Sequence[str], group: str = "difficulty") -> pd.DataFrame:
    from scipy.stats import kruskal

    rows = []
    for f in features:
        d = _clean(df, f, group)
        samples = [d.loc[d[group] == lvl, f].to_numpy() for lvl in DIFFICULTY_ORDER]
        samples = [s for s in samples if len(s)]
        try:
            h, p = kruskal(*samples) if len(samples) >= 2 else (np.nan, np.nan)
        except ValueError:  # all values identical
            h, p = np.nan, np.nan
        rows.append({"feature": f, "n": int(len(d)), "n_groups": len(samples),
                     "n_dropped": int(len(df) - len(d)), "H": float(h), "p_raw": float(p)})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_raw"]) if len(out) else []
    return out


def spearman_table(df: pd.DataFrame, features: Sequence[str], group: str = "difficulty") -> pd.DataFrame:
    from scipy.stats import spearmanr

    rows = []
    for f in features:
        d = _clean(df, f, group)
        rank = d[group].map(DIFFICULTY_RANK)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if len(d) >= 3:
                res = spearmanr(d[f], rank)
                rho, p = float(res.statistic if hasattr(res, "statistic") else res[0]), float(res.pvalue)
            else:
                rho, p = np.nan, np.nan
        rows.append({"feature": f, "n": int(len(d)), "n_dropped": int(len(df) - len(d)),
                     "rho": rho, "p_raw": p})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_raw"]) if len(out) else []
    return out


@dataclass
class LogitResult:
    table: pd.DataFrame
    n: int
    regularized: bool
    label: str
    converged: bool


def fit_logit(df: pd.DataFrame, features: Sequence[str] = LOGIT_FEATURES,
              outcome: str = "ex", difficulty: str = "difficulty",
              alpha: float = 0.01) -> LogitResult:
    """Logit of outcome on z-scored features + C(difficulty, Treatment('easy')).

    Falls back to L1-regularized fit on perfect separation / non-convergence.
    """
    import statsmodels.formula.api as smf
    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    try:
        from statsmodels.tools.sm_exceptions import PerfectSeparationError
    except ImportError:  # pragma: no cover
        PerfectSeparationError = Exception  # type: ignore

    cols = list(features) + [outcome, difficulty]
    d = df[cols].copy()
    for f in features:
        d[f] = pd.to_numeric(d[f], errors="coerce")
    d = d.dropna().reset_index(drop=True)
    zs = []
    for f in features:
        sd = d[f].std(ddof=1)
        z = f"z_{f}"
        d[z] = (d[f] - d[f].mean()) / sd if sd and sd > 0 else 0.0
        zs.append(z)
    d[outcome] = d[outcome].astype(int)
    levels = [lvl for lvl in DIFFICULTY_ORDER if lvl in set(d[difficulty])]
    d[difficulty] = pd.Categorical(d[difficulty], categories=levels)
    terms = list(zs)
    if len(levels) >= 2 and "easy" in levels:
        terms.append(f"C({difficulty}, Treatment('easy'))")
    elif len(levels) >= 2:
        terms.append(f"C({difficulty})")
    formula = f"{outcome} ~ " + (" + ".join(terms) if terms else "1")
    model = smf.logit(formula, data=d)

    regularized = False
    converged = True
    res = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            res = model.fit(disp=0, maxiter=200)
        converged = bool(res.mle_retvals.get("converged", True))
        if not converged:
            res = None
    except (PerfectSeparationError, ConvergenceWarning, np.linalg.LinAlgError, ValueError) as e:
        if type(e).__name__ not in ("PerfectSeparationError", "ConvergenceWarning",
                                    "LinAlgError", "ValueError"):
            raise
        res = None
    except Exception as e:  # newer statsmodels may raise PerfectSeparationWarning as error
        if "separation" not in str(e).lower() and "Separation" not in type(e).__name__:
            raise
        res = None

    if res is None:
        regularized = True
        converged = False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = model.fit_regularized(method="l1", alpha=alpha, disp=0, maxiter=1000)

    params = res.params
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            bse = res.bse
            pv = res.pvalues
            ci = res.conf_int()
        except Exception:
            bse = pd.Series(np.nan, index=params.index)
            pv = pd.Series(np.nan, index=params.index)
            ci = pd.DataFrame({0: np.nan, 1: np.nan}, index=params.index)
    table = pd.DataFrame({
        "term": params.index,
        "coef": params.values,
        "se": np.asarray(bse, dtype=float),
        "z": np.asarray(params.values, dtype=float) / np.asarray(bse, dtype=float),
        "p": np.asarray(pv, dtype=float),
        "odds_ratio": np.exp(params.values),
        "or_ci_low": np.exp(np.asarray(ci.iloc[:, 0], dtype=float)),
        "or_ci_high": np.exp(np.asarray(ci.iloc[:, 1], dtype=float)),
    }).reset_index(drop=True)
    table["n"] = int(len(d))
    label = ("Logistic regression (L1-regularized, alpha=%g; SEs/p-values not valid)" % alpha
             if regularized else "Logistic regression (maximum likelihood)")
    return LogitResult(table=table, n=int(len(d)), regularized=regularized,
                       label=label, converged=converged)
