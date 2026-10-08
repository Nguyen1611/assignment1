"""Generate notebooks 01, 02, 03 (model runs). Run: python notebooks/_mk_model_nbs.py"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent

SETUP = '''import sys
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
import pandas as pd
from IPython.display import display
from common import config, env, data, preflight
from common.config import RunConfig, resolve_limit, load_api_key
from common.llm_client import OpenRouterClient, fetch_endpoints
from common.pipeline import run_model_pipeline'''

ENDPOINTS = '''# Available providers + reported quantization; choose PROVIDER_ORDER from this table.
endpoints = fetch_endpoints(MODEL_ID)
display(pd.DataFrame(endpoints))'''

QWEN25_CFG = '''# Run_Config (single source of truth for this notebook)
RUN_NAME = "{run}"
MODEL_ID = "{model}"
LIMIT = resolve_limit(None)  # None = full dev set; or set A1_LIMIT
PROVIDER_ORDER = {pin}  # pinned so one quantization serves every call; see the fetch_endpoints table below

def make_cfg():
    return RunConfig(run_name=RUN_NAME, model_id=MODEL_ID, temperature=0.0, max_tokens=512,
                     reasoning_enabled=False, provider_order=tuple(PROVIDER_ORDER),
                     limit=LIMIT)

cfg = make_cfg()
cfg.to_dict()'''

QWEN3_CFG = '''# Run_Config (single source of truth for this notebook)
RUN_NAME = "qwen3_32b_thinking"
MODEL_ID = "qwen/qwen3-32b"
LIMIT = resolve_limit(None)  # None = full dev set; or set A1_LIMIT
PROVIDER_ORDER = ("DeepInfra",)  # REQUIRED -- pick from the fetch_endpoints table below

def make_cfg():
    if not PROVIDER_ORDER:
        raise ValueError("PROVIDER_ORDER must be set for the reasoning run: choose a provider "
                         "from the fetch_endpoints table and set PROVIDER_ORDER = ('Name',).")
    return RunConfig(run_name=RUN_NAME, model_id=MODEL_ID, temperature=0.6, top_p=0.95,
                     top_k=20, max_tokens=16384, reasoning_enabled=True,
                     provider_order=tuple(PROVIDER_ORDER), allow_fallbacks=False, limit=LIMIT)

print("MODEL_ID", MODEL_ID, "LIMIT", LIMIT, "PROVIDER_ORDER", PROVIDER_ORDER)'''

CLIENT = '''client = OpenRouterClient(api_key=load_api_key(required=True))'''

PREFLIGHT = '''# Preflight on the first min(preflight_n, N) examples; the full run is blocked unless it passes.
cfg = make_cfg()
examples = data.load_examples(LIMIT)
with env.track_run(RUN_NAME + ":preflight", LIMIT, extra={"provider_order": list(cfg.provider_order)}):
    try:
        pf = preflight.run_preflight(cfg, examples, client)
    except preflight.PreflightFailedError as e:
        rep = e.report
        failing = [c for c in rep.checks if not c["passed"]]
        display(pd.DataFrame(failing))
        raise SystemExit(f"Preflight FAILED -- full run not started. Report: {e.report_path}. "
                         "Fix the provider pin or config and re-run this cell.")
display(pd.DataFrame(pf.checks))
print("Preflight passed; providers served:", pf.providers_served)'''

RUN = '''cfg = make_cfg()
with env.track_run(RUN_NAME, LIMIT, extra={"fingerprint": cfg.fingerprint()}):
    res = run_model_pipeline(cfg, client, endpoints)
display(res.eval.table)
print("missing records:", res.n_missing, "| CLI check passed:", not res.mismatches)'''

PROV = '''# Served providers vs. the pin, with reported quantization (Req 12.5, 12.6, 22.5)
display(res.providers)
if (res.providers["flag"] == "outside pin").any():
    print("WARNING: some calls were served by providers outside the pin.")'''


def nb(title, intro, cells):
    n = nbf.v4.new_notebook()
    n.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
    n.cells = [nbf.v4.new_markdown_cell(f"# {title}\n{intro}")]
    for kind, src in cells:
        n.cells.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
    return n


def qwen25(run, model, title, pin):
    return nb(title, f"Zero-shot Text-to-SQL on Spider dev with `{model}` (temperature 0, 512 tokens).", [
        ("code", SETUP),
        ("code", QWEN25_CFG.format(run=run, model=model, pin=pin)),
        ("md", "## Provider endpoints"), ("code", ENDPOINTS),
        ("md", "## Run, extract, predict, evaluate, Official_CLI check"),
        ("code", CLIENT), ("code", RUN),
        ("md", "## Provider report"), ("code", PROV),
    ])


def main():
    out = {
        "01_qwen25_7b.ipynb": qwen25("qwen25_7b", "qwen/qwen-2.5-7b-instruct", "01 Qwen2.5-7B-Instruct", '("Phala",)'),
        "02_qwen25_72b.ipynb": qwen25("qwen25_72b", "qwen/qwen-2.5-72b-instruct", "02 Qwen2.5-72B-Instruct", '("DeepInfra",)'),
        "03_qwen3_32b_thinking.ipynb": nb(
            "03 Qwen3-32B (thinking)",
            "Reasoning on (temperature 0.6, top_p 0.95, top_k 20, 16384 tokens), pinned provider. "
            "Preflight must pass before the full run.", [
                ("code", SETUP), ("code", QWEN3_CFG),
                ("md", "## Provider endpoints"), ("code", ENDPOINTS),
                ("md", "## Preflight gate"), ("code", CLIENT), ("code", PREFLIGHT),
                ("md", "## Full run, extract, predict, evaluate, Official_CLI check"), ("code", RUN),
                ("md", "## Provider report"), ("code", PROV),
            ]),
    }
    for name, n in out.items():
        nbf.validate(n)
        nbf.write(n, HERE / name)
        print("wrote", name)


if __name__ == "__main__":
    main()
