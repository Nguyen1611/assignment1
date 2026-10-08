# CISC 451 Assignment 1: Spider Text-to-SQL benchmark

Zero-shot Text-to-SQL on the Spider dev set (1,034 examples) with three OpenRouter models
(Qwen2.5-7B-Instruct, Qwen2.5-72B-Instruct, Qwen3-32B thinking), reasoning-trace analysis (Task 2),
and execution-guided self-correction (EGSC, Task 3). Evaluation uses the official
`test-suite-sql-eval` evaluator, cross-checked against its CLI.

## Setup

```bash
cd assignment1
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then edit .env
```

`.env`:

- `OPENROUTER_API_KEY=...` (required for 🔑 cells). `.env` is git-ignored; never commit it.
- `SPIDER_EVAL_DIR=...` (optional). Path to the `test-suite-sql-eval` checkout containing
  `evaluation.py`. Default: `/home/nguyen/School/CISC_451/test-suite-sql-eval`.

## Notebook order

Run from `notebooks/` in order 00 → 06. Legend: 🌐 = network access, 🔑 = needs `OPENROUTER_API_KEY`. **Also notebook_export contain the exported run that I use in the report**

| Notebook | Purpose | Network / key cells |
|---|---|---|
| `00_setup_data` | Download Spider, NLTK data, verify dataset, gold-vs-gold check, wrapper-vs-CLI validation | 🌐 download cell (skipped if data present), 🌐 `nltk.download` |
| `01_qwen25_7b` | Qwen2.5-7B run + evaluation | 🌐 `fetch_endpoints`, 🔑 client + run cells |
| `02_qwen25_72b` | Qwen2.5-72B run + evaluation | 🌐 `fetch_endpoints`, 🔑 client + run cells |
| `03_qwen3_32b_thinking` | Qwen3-32B preflight gate, then full run | 🌐 `fetch_endpoints`, 🔑 client, preflight, run cells |
| `04_task1_summary` | Task 1 tables, figures, cost | offline |
| `05_task2_traces` | Task 2 reasoning-trace features + tests | offline |
| `06_task3_egsc` | Task 3 EGSC vs baseline, McNemar | 🔑 EGSC generation cells |

Notebooks 01–03 and 06 resume from `raw.jsonl`: re-running a cell skips examples that already have
a successful record, so an interrupted run can be continued by re-running the same cell.

### Choosing `PROVIDER_ORDER`

Each model notebook shows a `fetch_endpoints` table (provider, quantization, context, pricing).
Pick a provider and set e.g. `PROVIDER_ORDER = ("DeepInfra",)` in the Run_Config cell. It is
required for Qwen3-32B (no fallbacks) and optional for the Qwen2.5 runs. The provider report at the
end of each notebook flags calls served outside the pin.

### Prices (before notebook 04)

Fill `PRICES` in `common/config.py` (`input_per_m`, `output_per_m` in USD per million tokens) from
each model's OpenRouter page, and record `source` and `date`. Notebook 04 uses these for cost.

### Expected cost and runtime (estimates, unverified)

- Total API cost for all full runs: roughly under $3 (depends on provider prices and Qwen3
  reasoning length).
- Runtime: Qwen2.5 runs take minutes to tens of minutes each; Qwen3-32B thinking is the slowest
  (long traces, possibly hours). Real numbers are recorded in `results/environment.json`.

## Quick LIMIT runs

Set `A1_LIMIT=N` to use the first N dev examples. Outputs go to separate directories
(`outputs_limitN/`, `results_limitN/`), so full-run artifacts are never overwritten. A notebook's
own `LIMIT = <int>` overrides the env var. Notebook 00's wrapper-vs-CLI validation always uses
its own fixed set of the first 50 dev examples, independent of LIMIT. The Qwen3 preflight uses
the first `min(preflight_n, N)` examples.

## Quick live check (A1_LIMIT=10)

Do this before any full run. It costs a few cents.

```bash
cd assignment1
source .venv/bin/activate
export A1_LIMIT=10
jupyter lab            # launch from this shell so notebooks inherit A1_LIMIT
```

1. Run `00_setup_data` (all cells).
2. Run `01_qwen25_7b` and `02_qwen25_72b`: set `PROVIDER_ORDER` from the endpoints table (or leave
   `None`), then Run All.
3. Run `03_qwen3_32b_thinking`: set `PROVIDER_ORDER` (required), Run All. Preflight runs on the
   first `min(preflight_n, 10)` examples.
4. Run `04_task1_summary`, `05_task2_traces`, `06_task3_egsc`.
5. Check `outputs_limit10/<run>/summary.json` and `results_limit10/` tables and figures, and that
   every model notebook printed "wrapper totals match the Official_CLI".

Headless alternative (same order):

```bash
for nb in 00_setup_data 01_qwen25_7b 02_qwen25_72b 03_qwen3_32b_thinking \
          04_task1_summary 05_task2_traces 06_task3_egsc; do
  A1_LIMIT=10 jupyter nbconvert --to notebook --execute --inplace notebooks/$nb.ipynb || break
done
```

(Edit `PROVIDER_ORDER` in notebook 03 first; `--inplace` stores outputs in the notebooks.)

## Suggested order

1. `A1_LIMIT=10` quick live check (above).
2. Qwen3 preflight on the full config (`unset A1_LIMIT`, run notebook 03 up to the preflight cell).
3. Full runs: 00 → 06 without `A1_LIMIT`.

## Tests

```bash
source .venv/bin/activate
python -m pytest -q
```

Tests are offline (no API calls).

## Output locations

- `outputs[_limitN]/<run>/`: `raw.jsonl`, `preds.txt`, `gold.txt`, `per_example.csv`,
  `summary.json`, `providers.csv`; Qwen3 also `preflight.json`, `preflight_raw.jsonl`.
  `raw.jsonl` files are git-ignored.
- `results[_limitN]/`: `task1_*`, `task2_*`, `task3_*` tables (`.csv/.md/.tex`) and figures
  (`.png/.pdf`), `environment.json`.
- `results/wrapper_cli_validation.{csv,md}`: wrapper-vs-official-CLI validation.
- `METHODOLOGY.md`: report item → artifact map. `AI_USAGE.md`: AI usage declaration (draft).
