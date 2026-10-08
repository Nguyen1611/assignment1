# Methodology checklist: report item → artifact

Paths are relative to `assignment1/`. For a LIMIT run, read `outputs_limitN/` and `results_limitN/`
instead of `outputs/` and `results/`.

| Report item | Where it comes from |
|---|---|
| **Task 1: models** (names, OpenRouter IDs, params, access) | `notebooks/01–03` Run_Config cells; `results/task1_metadata.*` (nb 04) |
| Quantization / serving provider | `fetch_endpoints` table in nb 01–03; `outputs/<run>/providers.csv`; `results/environment.json` (models section) |
| Evaluation data / split | Spider dev, all 1,034 examples (`data/spider/dev.json`, `dev_gold.sql`); verified in nb 00 "Verify dataset" |
| Sampling | No sampling: full dev set in file order (LIMIT runs = first N). No seed needed for selection; the API `seed` (default 0, `common/config.py` RunConfig) is recorded in each run's config/fingerprint |
| Prompt template | `common/prompts.py`, version `v1`; per-call `messages_sha256` in `raw.jsonl` |
| Schema representation | `common/schema.py` (schema text built from `tables.json`/SQLite) |
| Few-shot | None (zero-shot) |
| SQL extraction rules | `common/extract.py` (`extract_sql`, `_strip_think`, `normalize_sql`); status per example in `per_example.csv`; failures → Placeholder_SQL |
| Decoding | Qwen2.5: temperature 0, max_tokens 512, reasoning off (nb 01/02 Run_Config). Qwen3-32B: temperature 0.6, top_p 0.95, top_k 20, max_tokens 16384, reasoning on, pinned provider, no fallbacks (nb 03) |
| Hardware / runtime | `results/environment.json` (`common/env.py`); `results/task1_environment.*` and `task1_efficiency.*` (nb 04) |
| Metrics | EX and EM via official evaluator wrapper (`common/evaluate.py`), cross-checked against the official CLI on every run (`record_cli_check` output in `outputs/<run>/`) |
| Invalid-output handling | Placeholder_SQL counted as wrong; `results/task1_invalid.*` (nb 04); `summary.json` (`n_parse_error`, `n_invalid`) |
| Task 1 results tables | `results/task1_accuracy.*`, `task1_ex_by_difficulty.*`, `task1_efficiency.*` (incl. cost from `config.PRICES`) |
| Task 1 figures | `results/task1_*.png/.pdf` (nb 04) |
| Qwen3 preflight report | `outputs/qwen3_32b_thinking/preflight.json`, `preflight_raw.jsonl`; `results/task1_preflight.*` (nb 04); gate cell in nb 03 |
| **Task 2: feature definitions** | `common/traces.py` (`FEATURES`); nb 05 feature cells |
| Task 2 trace length | `length_tokens` = number of tokens in the reasoning text under the `Qwen/Qwen3-32B` tokenizer (`tokenizers` 0.23.2, `common/tokenizer.py`), not the provider figure. `length_chars` (characters) is reported as a robustness check. Provider `reasoning_tokens` is kept as `provider_reasoning_tokens` only: DeepInfra's value is an estimate and exceeded `completion_tokens` on every LIMIT=10 row. Validation table: `results/task2_length_validation.*` |
| Unparseable predictions (EM) | Predictions the official parser cannot parse are scored EM=0 (official behaviour; EX unaffected). `unparseable_count`, `unparseable_rate`, `em_parseable`, `ex_unparseable` in `results/task1_invalid.*` and `results/task3_egsc_main.*` (`report.parse_stats`) |
| Task 2 logistic regression features | `traces.LOGIT_FEATURES` = length_tokens, markers_per_1k, sql_drafts, column_coverage + difficulty. table_coverage omitted (1.0 for all but one trace, quasi-separation); length_chars omitted (collinear with length_tokens). Fit by maximum likelihood |
| Task 2 provider token footnote | Task 1 efficiency table reports provider-reported `completion_tokens` and `reasoning_tokens`; footnote in the caption of `results/task1_efficiency.*` states the reasoning figure is a provider estimate |
| Task 2 exclusions | `results/task2_exclusions.*` |
| Task 2 descriptives | `results/task2_desc_by_correctness.*`, `task2_desc_by_difficulty.*` |
| Task 2 tests | `task2_mannwhitney.*` (by correctness, effect size), `task2_kruskal.*`, `task2_spearman.*` (difficulty); Holm-adjusted; `task2_logit.*` logistic regression |
| Task 2 figures | `results/task2_box_by_correctness.*`, `task2_box_by_difficulty.*` |
| **Task 3: method** | `common/egsc.py`; nb 06 (Qwen2.5-7B, retry on execution error/timeout, ≤2 retries, no gold SQL in the loop) |
| Task 3 main table | `results/task3_egsc_main.*` |
| Task 3 counts / cost / McNemar | `results/task3_egsc_counts.*` (triggered, fixed, broken, extra calls/tokens/time/cost, exact McNemar) |
| Task 3 examples | nb 06 "Examples: fixed and broken" cell output |
| Task 3 ablation (optional) | `results/task3_egsc_ablation_empty*.*` (only if `RUN_EMPTY_ABLATION=True`) |
| Wrapper vs CLI validation | `results/wrapper_cli_validation.{csv,md}` (nb 00, fixed first-50 mix) |
| AI usage declaration | `AI_USAGE.md` |
