"""Generate notebooks 04 (Task 1 summary), 05 (Task 2 traces), 06 (Task 3 EGSC).

Run: .venv/bin/python notebooks/_mk_analysis_nbs.py
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent

SETUP = '''import os, sys, json
from pathlib import Path
ROOT = Path.cwd().resolve()
if not (ROOT / "common").is_dir():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import config
# Optional override of the project root for outputs/results (used for offline fixture tests).
if os.environ.get("A1_ROOT_OVERRIDE"):
    config.ROOT_DIR = Path(os.environ["A1_ROOT_OVERRIDE"])

LIMIT = None  # set to an int for a smoke run; falls back to A1_LIMIT
LIMIT = config.resolve_limit(LIMIT)
OUT = config.out_root(LIMIT)
RES = config.results_root(LIMIT)
print("LIMIT =", LIMIT, "| outputs:", OUT, "| results:", RES)'''

RUNS = '''RUNS = {  # run_name -> (label, model_id)
    "qwen25_7b": ("Qwen2.5-7B-Instruct", "qwen/qwen-2.5-7b-instruct"),
    "qwen25_72b": ("Qwen2.5-72B-Instruct", "qwen/qwen-2.5-72b-instruct"),
    "qwen3_32b_thinking": ("Qwen3-32B (thinking)", "qwen/qwen3-32b"),
}'''


def nb(cells):
    n = nbf.v4.new_notebook()
    n.cells = [nbf.v4.new_markdown_cell(c[3:]) if c.startswith("MD:") else nbf.v4.new_code_cell(c)
               for c in cells]
    n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return n


# ============================================================ 04
NB04 = [
    "MD:# 04 · Task 1 summary (offline)\n\nBuilds the Task 1 tables and figure from saved "
    "`per_example.csv` files. No API calls; the API key is never loaded.",
    SETUP,
    RUNS + '''
from common import report, traces, env as env_mod
from common.preflight import PreflightReport

per_ex, model_ids = {}, {}
for run, (label, mid) in RUNS.items():
    p = OUT / run / "per_example.csv"
    if not p.is_file():
        print(f"[skip] {run}: {p} not found (run its notebook first)")
        continue
    per_ex[label] = pd.read_csv(p)
    model_ids[label] = mid
    print(f"[ok]   {run}: {len(per_ex[label])} rows")''',
    "MD:## Model metadata",
    '''env_meta = env_mod.load_env(env_mod.env_path(LIMIT))
models = {mid: (env_meta.get("models", {}) or {}).get(mid, {}) for mid in model_ids.values()}
meta = report.metadata_table(models)
report.export_table(meta, "task1_metadata", "Model metadata: parameters, type, served providers "
                    "(call counts) and reported quantization.", RES)
meta''',
    "MD:## Accuracy (EX / EM by difficulty)",
    '''if per_ex:
    acc = report.accuracy_table(per_ex)
    report.export_table(acc, "task1_accuracy", "Execution accuracy (EX) and exact-set match (EM) "
                        "on Spider dev, overall and per difficulty, with example counts.", RES)
    display(acc)
else:
    print("No runs available.")''',
    "MD:## Significance: bootstrap CIs and pairwise McNemar (EX)",
    '''if len(per_ex) >= 2:
    rng = np.random.default_rng(0)  # fixed seed for the bootstrap
    srt = {lab: df.sort_values("example_id").reset_index(drop=True) for lab, df in per_ex.items()}
    ci_rows = []
    for lab, df in srt.items():
        ex = df["ex"].to_numpy(dtype=float)
        bs = ex[rng.integers(0, len(ex), (2000, len(ex)))].mean(axis=1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        ci_rows.append({"model": lab, "n": len(ex), "ex_all": ex.mean(), "ci95_low": lo, "ci95_high": hi})
    ci = pd.DataFrame(ci_rows)
    report.export_table(ci, "task1_ex_ci", "Overall EX with 95% percentile bootstrap confidence "
                        "intervals (2,000 resamples of the 1,034 examples, seed 0).", RES)
    labs = list(srt); pairs = []
    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            a, b = srt[labs[i]], srt[labs[j]]
            assert (a["example_id"].to_numpy() == b["example_id"].to_numpy()).all()
            m = report.mcnemar_exact(a["ex"], b["ex"])  # b: only model B correct, c: only A
            pairs.append({"model_a": labs[i], "model_b": labs[j], "ex_a": a["ex"].mean(),
                          "ex_b": b["ex"].mean(), "only_a_correct": m["c"],
                          "only_b_correct": m["b"], "mcnemar_p": m["p"]})
    pw = pd.DataFrame(pairs)
    pw["p_holm"] = traces.holm(pw["mcnemar_p"])
    report.export_table(pw, "task1_pairwise_mcnemar", "Pairwise exact McNemar tests on per-example "
                        "EX (same 1,034 examples); Holm-adjusted across the three pairs.", RES)
    display(ci); display(pw)''',
    "MD:## Efficiency and cost",
    '''if per_ex:
    eff = report.efficiency_table(per_ex, model_ids)
    report.export_table(eff, "task1_efficiency", "Mean latency and token usage per example, and total "
                        "cost. Token counts are provider-reported: mean_output_tokens is completion_tokens and "
                        "mean_reasoning_tokens is the provider's reasoning_tokens, which is an estimate on some providers "
                        "(DeepInfra) and can exceed completion_tokens; Task 2 measures trace length from the text. "
                        + report.price_note(list(model_ids.values())), RES)
    display(eff)''',
    "MD:## Invalid outputs",
    '''if per_ex:
    inv = report.invalid_table(per_ex)
    report.export_table(inv, "task1_invalid", "Counts and rates of each SQL extraction status and of "
                        "non-executable predictions. unparseable = predictions the official Spider "
                        "parser cannot parse (scored EM=0 by the official script; EX unaffected); "
                        "em_parseable = EM on the parseable subset; ex_unparseable = EX on the "
                        "unparseable subset.", RES)
    display(inv)''',
    "MD:## Environment and preflight",
    '''envt = report.environment_table(env_meta)
report.export_table(envt, "task1_environment", "Experimental environment: local machine (orchestration "
                    "and evaluation only), packages, and remote inference providers.", RES)
display(envt)

pf_path = OUT / "qwen3_32b_thinking" / "preflight.json"
if pf_path.is_file():
    pft = report.preflight_table(PreflightReport.load(pf_path))
    report.export_table(pft, "task1_preflight", "Qwen3-32B reasoning-trace preflight: served provider "
                        "and per-check results.", RES)
    display(pft)
else:
    print(f"[skip] preflight: {pf_path} not found")''',
    "MD:## EX by difficulty",
    '''if per_ex:
    levels = ["easy", "medium", "hard", "extra", "all"]
    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.8 / len(per_ex)
    x = np.arange(len(levels))
    for i, (label, row) in enumerate(acc.set_index("model").iterrows()):
        ax.bar(x + i * w - 0.4 + w / 2, [row[f"ex_{l}"] for l in levels], w, label=label)
    ax.set_xticks(x, levels); ax.set_ylim(0, 1); ax.set_ylabel("Execution accuracy")
    ax.set_title("EX by difficulty")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    fig.tight_layout()
    print(report.save_figure(fig, "task1_ex_by_difficulty", RES))
    plt.close(fig)''',
]

# ============================================================ 05
NB05 = [
    "MD:# 05 · Task 2 reasoning-trace analysis (offline)\n\nJoins the Qwen3-32B raw log with its "
    "evaluated `per_example.csv`, extracts trace features, and runs the statistical tests.",
    SETUP,
    '''from common import report, traces, evaluate, data
from common.tokenizer import tokenizer_info
RUN = "qwen3_32b_thinking"
run_dir = OUT / RUN
pe = pd.read_csv(run_dir / "per_example.csv")
recs = {}
with open(run_dir / "raw.jsonl", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("phase", "full") != "full":
            continue
        # keep the last successful record per example (else the last one)
        eid = int(r["example_id"])
        if eid not in recs or r.get("status", "ok") == "ok":
            recs[eid] = r
print(len(pe), "evaluated rows;", len(recs), "raw records")''',
    "MD:## Gold schema items and features",
    '''examples = {ex.example_id: ex for ex in data.load_examples(LIMIT)}
api = evaluate.import_official()
tables_json = config.DATA_DIR / "tables.json"
feats = []
with evaluate.official_cwd():
    for eid in pe["example_id"]:
        ex = examples[int(eid)]
        try:
            items = traces.gold_schema_items(ex.gold_sql, ex.db_id, api, tables_json)
        except Exception as e:  # parser failure: no coverage
            print(f"gold parse failed for {eid}: {e}")
            items = (set(), set())
        rec = recs.get(int(eid), {"reasoning_missing": True})
        feats.append(traces.extract_features(rec, ex, items).to_dict())
fdf = pd.DataFrame(feats).merge(pe[["example_id", "difficulty", "ex"]], on="example_id")
excl = (fdf[fdf.excluded].groupby("reason").size().rename("count").reset_index()
        if fdf.excluded.any() else pd.DataFrame({"reason": [], "count": []}))
excl = pd.concat([excl, pd.DataFrame([{"reason": "included", "count": int((~fdf.excluded).sum())}])])
report.export_table(excl, "task2_exclusions", "Examples excluded from trace analysis, by reason.", RES)
df = fdf[~fdf.excluded].copy()
df["correct"] = df["ex"].map({1: "correct", 0: "incorrect"})
FEATS = list(traces.FEATURES)
display(excl)

# Length validation: tokenizer count of the trace text vs provider-reported numbers.
comp = pd.Series({int(r["example_id"]): r.get("completion_tokens") for r in recs.values()})
v = df.assign(completion_tokens=df["example_id"].map(comp))
val = pd.DataFrame([{
    "n": len(v),
    "mean_text_tokens": v["length_tokens"].mean(),
    "mean_provider_reasoning_tokens": v["provider_reasoning_tokens"].mean(),
    "mean_provider_completion_tokens": v["completion_tokens"].mean(),
    "mean_chars": v["length_chars"].mean(),
    "provider_reasoning_gt_completion": int((v["provider_reasoning_tokens"] > v["completion_tokens"]).sum()),
    "text_tokens_gt_completion": int((v["length_tokens"] > v["completion_tokens"]).sum()),
    "spearman_text_vs_provider": v["length_tokens"].corr(v["provider_reasoning_tokens"], method="spearman"),
}])
info = tokenizer_info()
report.export_table(val, "task2_length_validation",
    f"Trace length measured by tokenizing the reasoning text ({info['tokenizer']}, tokenizers "
    f"{info['tokenizers_version']}) versus provider-reported counts. Provider "
    "reasoning_tokens is an estimate (DeepInfra) and can exceed completion_tokens, so length_tokens "
    "(text) is used in all tests; length_chars is reported as a robustness check.", RES)
display(val)''',
    "MD:## Descriptive statistics",
    '''d_corr = traces.describe(df, FEATS, "correct")
d_diff = traces.describe(df, FEATS, "difficulty")
report.export_table(d_corr, "task2_desc_by_correctness", "Trace features by EX correctness "
                    "(count, mean, median, quartiles).", RES)
report.export_table(d_diff, "task2_desc_by_difficulty", "Trace features by difficulty "
                    "(count, mean, median, quartiles).", RES)
display(d_corr); display(d_diff)''',
    "MD:## Tests (Holm-adjusted)",
    '''mwu = traces.mann_whitney_table(df, FEATS, "ex")
kw = traces.kruskal_table(df, FEATS, "difficulty")
sp = traces.spearman_table(df, FEATS, "difficulty")
report.export_table(mwu, "task2_mannwhitney", "Mann-Whitney U, correct vs incorrect, with rank-biserial "
                    "effect size; Holm-adjusted p-values across features.", RES)
report.export_table(kw, "task2_kruskal", "Kruskal-Wallis across difficulty levels; Holm-adjusted.", RES)
report.export_table(sp, "task2_spearman", "Spearman correlation with ordinal difficulty "
                    "(easy=1..extra=4); Holm-adjusted.", RES)
display(mwu); display(kw); display(sp)''',
    "MD:## Logistic regression",
    '''lr = traces.fit_logit(df)
report.export_table(lr.table, "task2_logit", f"{lr.label}: EX on z-scored trace features plus "
                    f"difficulty (reference: easy), n={lr.n}. table_coverage is omitted (1.0 for all but "
                    "one trace, causing quasi-separation); length_chars is omitted (collinear "
                    "with length_tokens).", RES)
print(lr.label, "| converged:", lr.converged)
lr.table''',
    "MD:## Box plots",
    '''def boxes(by, order, name, title):
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for ax in axes.flat[len(FEATS):]:
        ax.axis("off")
    for ax, f in zip(axes.flat, FEATS):
        groups = [pd.to_numeric(df.loc[df[by] == g, f], errors="coerce").dropna() for g in order]
        ax.boxplot(groups)
        ax.set_xticks(range(1, len(order) + 1), order)
        ax.set_title(f, fontsize=9)
    fig.suptitle(title); fig.tight_layout()
    print(report.save_figure(fig, name, RES)); plt.close(fig)

boxes("correct", ["correct", "incorrect"], "task2_box_by_correctness", "Trace features by correctness")
boxes("difficulty", ["easy", "medium", "hard", "extra"], "task2_box_by_difficulty",
      "Trace features by difficulty")''',
]

# ============================================================ 06
NB06 = [
    "MD:# 06 · Task 3 execution-guided self-correction (🌐🔑)\n\nCells marked 🌐 call the "
    "OpenRouter API and need `OPENROUTER_API_KEY`. EGSC never sees gold SQL; gold is used only "
    "in the evaluation cells.",
    SETUP,
    '''from common import report, data, evaluate
from common.config import RunConfig
from common.egsc import run_egsc
from common.runner import read_predictions

cfg = RunConfig(run_name="qwen25_7b_egsc", model_id="qwen/qwen-2.5-7b-instruct",
                temperature=0.0, max_tokens=512, retry_on_empty=False, limit=LIMIT)
examples = data.load_examples(LIMIT)
orig_dir = OUT / "qwen25_7b"
orig_preds = read_predictions(orig_dir / "preds.txt")
orig_pe = pd.read_csv(orig_dir / "per_example.csv")
assert len(orig_preds) == len(examples), (len(orig_preds), len(examples))
print(cfg.run_name, cfg.fingerprint()[:12], len(examples), "examples")''',
    "MD:## 🌐 Run EGSC (resumable)",
    '''import time
from common.llm_client import OpenRouterClient
client = OpenRouterClient(api_key=config.load_api_key(required=True))
t0 = time.monotonic()
stats = run_egsc(cfg, examples, orig_preds, client)
egsc_wall_s = time.monotonic() - t0
print({k: v for k, v in vars(stats).items() if k != "final_sqls"}, "| wall s:", round(egsc_wall_s, 1))''',
    "MD:## Evaluate final predictions",
    '''def load_egsc(run_dir):
    recs = {}
    with open(run_dir / "raw.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    r = json.loads(line); recs.setdefault(int(r["example_id"]), r)
                except (json.JSONDecodeError, KeyError):
                    pass
    return recs

def evaluate_egsc(run_cfg):
    run_dir = run_cfg.output_dir()
    finals = read_predictions(run_dir / "preds.txt")
    s = evaluate.evaluate_predictions(finals, data.gold_lines(LIMIT), None, None, run_dir,
                                      run_name=run_cfg.run_name)
    gold_path = data.write_gold(run_dir / "gold.txt", LIMIT)
    cli = evaluate.run_official_cli(gold_path, run_dir / "preds.txt")
    mism = evaluate.compare_totals(s.table, cli)
    evaluate.record_cli_check(run_dir, mism)
    print("CLI totals check:", "PASS" if not mism else f"FAIL {mism}")
    return s.per_example, load_egsc(run_dir)

egsc_pe, egsc_recs = evaluate_egsc(cfg)''',
    "MD:## Results",
    '''main = report.egsc_main_table(orig_pe, egsc_pe)
report.export_table(main, "task3_egsc_main", "Qwen2.5-7B original vs primary EGSC (retry on execution "
                    "error/timeout, up to 2 retries): EX/EM overall and per difficulty. unparseable = "
                    "predictions the official parser cannot parse (EM=0); em_parseable = EM on the "
                    "parseable subset; ex_unparseable = EX on the unparseable subset.", RES)
display(main)

def summarize(pe_final, recs, label):
    o = orig_pe.sort_values("example_id"); f = pe_final.sort_values("example_id")
    trig = [bool(recs[int(e)]["triggered"]) for e in o["example_id"]]
    counts = report.egsc_counts(o["ex"], f["ex"], trig)
    mc = report.mcnemar_exact(o["ex"], f["ex"])
    pt = sum(a.get("prompt_tokens") or 0 for r in recs.values() for a in r["attempts"][1:])
    ct = sum(a.get("completion_tokens") or 0 for r in recs.values() for a in r["attempts"][1:])
    lat = sum(a.get("latency_s") or 0 for r in recs.values() for a in r["attempts"][1:])
    calls = sum(r["n_llm_calls"] for r in recs.values())
    row = {"system": label, **counts, "extra_llm_calls": calls, "extra_prompt_tokens": pt,
           "extra_completion_tokens": ct, "extra_llm_time_s": lat,
           "extra_cost_usd": report.cost_usd(pt, ct, cfg.model_id),
           "mcnemar_b_fixed": mc["b"], "mcnemar_c_broken": mc["c"], "mcnemar_p": mc["p"]}
    return pd.DataFrame([row])

cnt = summarize(egsc_pe, egsc_recs, "primary EGSC")
report.export_table(cnt, "task3_egsc_counts", "Primary EGSC: triggered/fixed/broken counts, extra "
                    "LLM calls, tokens, time and cost, and exact McNemar test on paired EX. "
                    + report.price_note([cfg.model_id]), RES)
display(cnt)''',
    "MD:## Examples: fixed and broken",
    '''def show(kind, n=2):
    o = orig_pe.set_index("example_id")["ex"]; f = egsc_pe.set_index("example_id")["ex"]
    want = (0, 1) if kind == "fixed" else (1, 0)
    ids = [e for e in o.index if (o[e], f[e]) == want][:n]
    exs = {ex.example_id: ex for ex in examples}
    print(f"=== {kind}: {len(ids)} shown ===")
    for e in ids:
        r, ex = egsc_recs[int(e)], exs[int(e)]
        print(f"\\n[{e}] {ex.db_id}: {ex.question}\\n  original: {r['original_sql']}")
        for a in r["attempts"][1:]:
            print(f"  feedback: {a.get('feedback')}\\n  retry {a['k']}: {a['sql']}")
        print(f"  final:    {r['final_sql']}\\n  gold:     {ex.gold_sql}")
show("fixed"); show("broken")''',
    "MD:## Ablation: retry on empty result (🌐, off by default)\n\nSeparate run "
    "`qwen25_7b_egsc_empty` with `retry_on_empty=True`. Reported only as an ablation.",
    '''RUN_EMPTY_ABLATION = False''',
    '''if RUN_EMPTY_ABLATION:
    cfg_empty = RunConfig(run_name="qwen25_7b_egsc_empty", model_id="qwen/qwen-2.5-7b-instruct",
                          temperature=0.0, max_tokens=512, retry_on_empty=True, limit=LIMIT)
    run_egsc(cfg_empty, examples, orig_preds, client)
    empty_pe, empty_recs = evaluate_egsc(cfg_empty)
    abl = report.egsc_ablation_table(orig_pe, empty_pe)
    report.export_table(abl, "task3_egsc_ablation_empty", report.ablation_caption(
        "Qwen2.5-7B original vs EGSC variant that also retries zero-row results"), RES)
    acnt = summarize(empty_pe, empty_recs, f"EGSC ({report.ABLATION_LABEL})")
    report.export_table(acnt, "task3_egsc_ablation_empty_counts", report.ablation_caption(
        "EGSC zero-row variant: counts, extra cost and McNemar test"), RES)
    display(abl); display(acnt)
else:
    print("Ablation skipped (RUN_EMPTY_ABLATION=False).")''',
]

if __name__ == "__main__":
    for name, cells in (("04_task1_summary", NB04), ("05_task2_traces", NB05), ("06_task3_egsc", NB06)):
        n = nb(cells)
        nbf.validate(n)
        nbf.write(n, HERE / f"{name}.ipynb")
        print("wrote", name)
