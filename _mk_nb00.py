import nbformat as nbf
nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
md("# 00 Setup data\nDownloads Spider, installs NLTK data, verifies the dataset, and runs a gold-vs-gold smoke check of the official evaluator."),
code('''import sys, os
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
from common import config, env, data
from common.config import OFFICIAL_DIR, DATA_DIR, resolve_limit

# Run_Config
RUN_NAME = "00_setup_data"
LIMIT = resolve_limit(None)
SPIDER_GDRIVE_ID = "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"  # from taoyds/spider README
print("ROOT", ROOT, "DATA_DIR", DATA_DIR, "OFFICIAL_DIR", OFFICIAL_DIR, "LIMIT", LIMIT)'''),
md("## Download Spider (idempotent)"),
code('''import shutil, zipfile

def data_present():
    return all((DATA_DIR / p).exists() for p in ("dev.json", "tables.json", "dev_gold.sql", "database"))

with env.track_run(RUN_NAME + ":download", LIMIT):
    if data_present():
        print("Spider data already present; skipping download.")
    else:
        import gdown
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        zpath = DATA_DIR.parent / "spider_data.zip"
        if not zpath.exists():
            gdown.download(id=SPIDER_GDRIVE_ID, output=str(zpath), quiet=False)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(DATA_DIR.parent / "_extract")
        ex = DATA_DIR.parent / "_extract"
        src = next(p.parent for p in ex.rglob("dev.json") if "__MACOSX" not in p.parts)
        for item in src.iterdir():
            dst = DATA_DIR / item.name
            if not dst.exists():
                shutil.move(str(item), str(dst))
        shutil.rmtree(ex)
        zpath.unlink()
        assert data_present(), "Extraction did not produce expected files"
    print(sorted(p.name for p in DATA_DIR.iterdir()))'''),
md("If `gdown` fails, download `spider_data.zip` manually from the Spider site/Google Drive link in the taoyds/spider README and extract so that `data/spider/dev.json`, `tables.json`, `dev_gold.sql`, and `database/` exist."),
code('''import nltk
for pkg in ("punkt", "punkt_tab"):
    nltk.download(pkg, quiet=True)
print("nltk ok")'''),
md("## Verify dataset"),
code('''summary = data.verify_dataset()
print(summary["n_examples"], summary["n_gold"], summary["n_databases"])
(OFFICIAL_DIR / "tmp").mkdir(exist_ok=True)'''),
md("## Gold-vs-gold Official_CLI smoke check"),
code('''import subprocess, re, tempfile
with env.track_run(RUN_NAME + ":gold_vs_gold", LIMIT):
    tmpd = Path(tempfile.mkdtemp())
    gold = data.write_gold(tmpd / "gold.txt", LIMIT)
    pred = tmpd / "pred.txt"
    pred.write_text("".join(s + "\\n" for s, _ in data.gold_lines(LIMIT)), encoding="utf-8")
    r = subprocess.run([sys.executable, "evaluation.py", "--gold", str(gold), "--pred", str(pred),
                        "--db", str(DATA_DIR / "database"), "--table", str(DATA_DIR / "tables.json"),
                        "--etype", "all"], cwd=OFFICIAL_DIR, capture_output=True, text=True)
    out = r.stdout
    print(out[-3000:]); print(r.stderr[-2000:])
    def overall(label):
        line = next(l for l in out.splitlines() if l.strip().startswith(label))
        return float(line.split()[-1])
    ex, em = overall("execution"), overall("exact match")
    print("EX", ex, "EM", em)
    assert ex == 1.0 and em == 1.0'''),
md("## Wrapper-vs-CLI validation (Req 11.1–11.3)\nDeterministic mix of 50 predictions on the first 50 dev examples: gold, perturbed (LIMIT 1 / wrong projection), unparsable text, erroring SQL, and Placeholder_SQL. The wrapper totals must match the official CLI."),
code('''import re
import pandas as pd
from common import evaluate
from common.extract import PLACEHOLDER_SQL

N_VAL = 50
val_gold = data.gold_lines(N_VAL)

def make_pred(i, g):
    kind = ["gold", "limit", "unparsable", "error", "placeholder",
            "gold", "wrong_column", "unparsable", "error", "gold"][i % 10]
    if kind == "gold":
        return kind, g
    if kind == "limit":
        return kind, (re.sub(r"(?i)\\bLIMIT\\s+\\d+", "LIMIT 2", g) if re.search(r"(?i)\\bLIMIT\\b", g)
                      else g + " LIMIT 1")
    if kind == "wrong_column":
        return kind, re.sub(r"(?is)^SELECT\\s+.*?\\s+FROM\\s", "SELECT * FROM ", g, count=1)
    if kind == "unparsable":
        return kind, "I cannot answer this question"
    if kind == "error":
        return kind, "SELECT no_such_column FROM no_such_table"
    return kind, PLACEHOLDER_SQL

kinds, val_preds = zip(*(make_pred(i, g) for i, (g, _) in enumerate(val_gold)))
val_dir = config.ROOT_DIR / "outputs" / "_wrapper_cli_validation"
val_dir.mkdir(parents=True, exist_ok=True)
val_gold_path = data.write_gold(val_dir / "gold.txt", N_VAL)
val_pred_path = val_dir / "preds.txt"
val_pred_path.write_text("".join(p + "\\n" for p in val_preds), encoding="utf-8")
print(pd.Series(kinds).value_counts().to_dict())'''),
code('''with env.track_run(RUN_NAME + ":wrapper_vs_cli", LIMIT):
    ext = [{"extraction_status": "missing" if k == "placeholder" else "ok", "kind": k} for k in kinds]
    res = evaluate.evaluate_predictions(list(val_preds), val_gold, ext, None, val_dir,
                                        run_name="wrapper_cli_validation")
    cli = evaluate.run_official_cli(val_gold_path, val_pred_path)
    mismatches = evaluate.compare_totals(res.table, cli)
    evaluate.record_cli_check(val_dir, mismatches)

val_table = pd.DataFrame({
    "level": evaluate.LEVELS,
    "count_wrapper": [int(res.table.loc[l, "count"]) for l in evaluate.LEVELS],
    "count_cli": [int(cli["count"][l]) for l in evaluate.LEVELS],
    "ex_wrapper": [round(float(res.table.loc[l, "ex"]), 4) for l in evaluate.LEVELS],
    "ex_cli": [cli["execution"][l] for l in evaluate.LEVELS],
    "em_wrapper": [round(float(res.table.loc[l, "em"]), 4) for l in evaluate.LEVELS],
    "em_cli": [cli["exact match"][l] for l in evaluate.LEVELS],
})
res_dir = config.results_root(None)
res_dir.mkdir(parents=True, exist_ok=True)
val_table.to_csv(res_dir / "wrapper_cli_validation.csv", index=False)
(res_dir / "wrapper_cli_validation.md").write_text("\\n".join(["| " + " | ".join(val_table.columns) + " |", "|" + "---|" * len(val_table.columns)]
    + ["| " + " | ".join(str(v) for v in r) + " |" for r in val_table.itertuples(index=False)]) + "\\n", encoding="utf-8")
print(val_table.to_string(index=False))
print("parse errors:", res.summary["n_parse_error"], "invalid:", res.summary["n_invalid"])
assert mismatches == [], mismatches
print("Wrapper totals match the official CLI.")'''),
md("## Environment snapshot"),
code('''import json
print(json.dumps(env.snapshot(), indent=2))'''),
]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "notebooks/00_setup_data.ipynb")
