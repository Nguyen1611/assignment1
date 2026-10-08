"""Tests for common/report.py on synthetic data (no network)."""
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from hypothesis import given, strategies as st

from common import config, report


def _pe(rows):
    return pd.DataFrame(rows)


SYN = _pe([
    {"difficulty": "easy", "ex": 1, "em": 1, "extraction_status": "ok", "exec_status": "ok",
     "prompt_tokens": 100, "completion_tokens": 20, "reasoning_tokens": None, "latency_s": 1.0},
    {"difficulty": "medium", "ex": 0, "em": 0, "extraction_status": "fallback", "exec_status": "error",
     "prompt_tokens": 120, "completion_tokens": 30, "reasoning_tokens": None, "latency_s": 2.0},
    {"difficulty": "hard", "ex": 1, "em": 0, "extraction_status": "missing", "exec_status": "ok",
     "prompt_tokens": 80, "completion_tokens": 10, "reasoning_tokens": None, "latency_s": 3.0},
])


def test_provider_report_flags_outside_pin():
    recs = [{"status": "ok", "provider": "Together", "run_config": {"provider_order": ["together"]}}] * 3
    recs += [{"status": "ok", "provider": "DeepInfra"}, {"status": "error", "provider": "X"}]
    eps = [{"provider_name": "Together", "quantization": "fp8"}]
    df = report.provider_report(recs, eps)
    t = df.set_index("provider")
    assert t.loc["Together", "calls"] == 3 and t.loc["Together", "quantization"] == "fp8"
    assert t.loc["Together", "in_pin"] and t.loc["Together", "flag"] == ""
    assert t.loc["DeepInfra", "flag"] == "outside pin"
    assert t.loc["DeepInfra", "quantization"] == "not reported"
    assert "X" not in t.index


def test_cost_usd(monkeypatch):
    assert math.isnan(report.cost_usd(10, 10, "qwen/qwen-2.5-7b-instruct"))  # unfilled
    assert math.isnan(report.cost_usd(10, 10, "nope/model"))
    monkeypatch.setitem(config.PRICES, "m", {"input_per_m": 1.0, "output_per_m": 2.0,
                                             "source": "s", "date": "d"})
    assert report.cost_usd(1_000_000, 500_000, "m") == pytest.approx(2.0)
    assert "s, d" in report.price_note(["m"])
    assert "not set" in report.price_note(["nope/model"])


def test_task1_tables_and_export(tmp_path, monkeypatch):
    acc = report.accuracy_table({"7B": SYN})
    assert acc.loc[0, "n_all"] == 3 and acc.loc[0, "ex_all"] == pytest.approx(2 / 3)
    assert acc.loc[0, "n_extra"] == 0
    monkeypatch.setitem(config.PRICES, "m", {"input_per_m": 1.0, "output_per_m": 1.0})
    eff = report.efficiency_table({"7B": SYN}, {"7B": "m"})
    assert eff.loc[0, "mean_latency_s"] == pytest.approx(2.0)
    assert eff.loc[0, "total_cost_usd"] == pytest.approx(360 / 1e6)
    assert math.isnan(eff.loc[0, "mean_reasoning_tokens"])
    inv = report.invalid_table({"7B": SYN})
    assert inv.loc[0, "missing_count"] == 1 and inv.loc[0, "non_exec_count"] == 1
    meta = report.metadata_table({"qwen/qwen3-32b": {"providers": {"A": {"calls": 5, "quantization": "bf16"}}}})
    assert meta.loc[0, "params"] == "32B" and meta.loc[0, "quantization"] == "bf16"
    env = report.environment_table({"machine": {"os": "Linux", "is_wsl": True},
                                    "packages": {"pandas": "2.0"}, "official_commit": None,
                                    "inference": {"access": "api", "service": "OpenRouter",
                                                  "local_role": "orchestration"},
                                    "runs": [{"name": "r", "started_at": "a", "finished_at": "b"}]})
    assert "Running under WSL" in set(env["item"])
    pf = report.preflight_table({"providers_served": ["A"], "example_ids": [0, 1], "passed": False,
                                 "checks": [{"name": "api_ok", "passed": False, "failing_ids": [1]}]})
    assert pf.set_index("item").loc["api_ok", "failing_ids"] == "1"

    paths = report.export_table(acc, "acc_tbl", "Accuracy_test & 50%", tmp_path)
    assert set(paths) == {"csv", "md", "tex"} and all(p.exists() for p in paths.values())
    assert paths["md"].read_text().startswith("*Accuracy_test & 50%*")
    tex = paths["tex"].read_text()
    assert r"\label{tab:acc_tbl}" in tex and r"\caption{Accuracy\_test \& 50\%}" in tex
    fig, ax = plt.subplots(); ax.bar([1, 2], [1, 2])
    fp = report.save_figure(fig, "fig", tmp_path)
    plt.close(fig)
    assert fp["png"].stat().st_size > 0 and fp["pdf"].stat().st_size > 0


def test_egsc_tables():
    final = SYN.assign(ex=[1, 1, 1])
    main = report.egsc_main_table(SYN, final)
    assert list(main["system"]) == ["original", "primary EGSC"]
    assert main.loc[1, "ex_all"] == pytest.approx(1.0)
    abl = report.egsc_ablation_table(SYN, final)
    assert "ablation: retry_on_empty=true" in abl.loc[1, "system"]
    m = report.mcnemar_exact([0, 0, 1, 1], [1, 1, 1, 1])
    assert (m["b"], m["c"]) == (2, 0) and m["p"] == pytest.approx(0.5)
    assert report.mcnemar_exact([1], [1])["p"] == 1.0


# Feature: spider-text-to-sql-benchmark, Property 29: EGSC counts are consistent
# **Validates: Requirements 17.2, 17.4**
@given(st.lists(st.tuples(st.integers(0, 1), st.integers(0, 1), st.integers(0, 2)), max_size=60))
def test_egsc_counts_property(trip):
    o = [a for a, _, _ in trip]; f = [b for _, b, _ in trip]; t = [c for _, _, c in trip]
    c = report.egsc_counts(o, f, t)
    assert c["fixed"] + c["broken"] + c["unchanged"] == len(trip)
    assert c["fixed"] - c["broken"] == sum(f) - sum(o)
    assert c["triggered"] == sum(x > 0 for x in t)
    m = report.mcnemar_exact(o, f)
    assert m["b"] == c["fixed"] and m["c"] == c["broken"] and 0.0 <= m["p"] <= 1.0


def test_parse_stats_in_invalid_and_egsc_tables():
    # parse_ok as read back from CSV (strings) and as bools
    for flags in ([True, False, False], ["True", "False", "False"]):
        df = SYN.assign(parse_ok=flags)
        s = report.parse_stats(df)
        assert s["unparseable_count"] == 2
        assert s["unparseable_rate"] == pytest.approx(2 / 3)
        assert s["em_parseable"] == pytest.approx(1.0)       # only row 0 parses, em=1
        assert s["ex_unparseable"] == pytest.approx(0.5)     # rows 1,2: ex 0,1
    df = SYN.assign(parse_ok=[True, False, False])
    inv = report.invalid_table({"7B": df})
    assert inv.loc[0, "unparseable_count"] == 2
    main = report.egsc_main_table(df, df.assign(ex=[1, 1, 1]))
    assert {"unparseable_count", "unparseable_rate", "em_parseable"} <= set(main.columns)
    # no parse_ok column -> NaN rates, no crash
    assert math.isnan(report.parse_stats(SYN)["em_parseable"])
