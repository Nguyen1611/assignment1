"""Tests for common/preflight.py (Req 21). Offline: mock client only."""
from __future__ import annotations

import dataclasses

import pytest
from hypothesis import given, settings, strategies as st

from common import config, preflight, runner
from common.config import RunConfig
from common.data import Example
from common.llm_client import LLMResult

PIN = ("DeepInfra",)


def make_cfg(**kw) -> RunConfig:
    base = dict(run_name="qwen_test", model_id="qwen/qwen3-32b", temperature=0.6,
                reasoning_enabled=True, provider_order=PIN, threads=2, preflight_n=5)
    base.update(kw)
    return RunConfig(**base)


def good_record(eid: int) -> dict:
    return {"example_id": eid, "status": "ok", "reasoning": "x" * 400,
            "reasoning_source": "details_text", "reasoning_detail_types": ["reasoning.text"],
            "reasoning_tokens": 100, "text": "```sql\nSELECT 1\n```",
            "finish_reason": "stop", "provider": "DeepInfra"}


MUTATIONS = {
    "api_ok": {"status": "error"},
    "reasoning_nonempty": {"reasoning": "   "},
    "reasoning_raw_not_summary": {"reasoning_detail_types": ["reasoning.summary"]},
    "reasoning_tokens_positive": {"reasoning_tokens": 0},
    "sql_extractable": {"text": "I cannot answer.", "finish_reason": "length"},
    "provider_matches_pin": {"provider": "OtherCo"},
}


class MockClient:
    def __init__(self, fn):
        self.fn, self.calls = fn, []

    def complete(self, messages, cfg):
        self.calls.append(messages)
        return self.fn(messages, cfg)


def ok_result(provider="DeepInfra", **kw):
    d = dict(status="ok", text="```sql\nSELECT name FROM singer\n```", reasoning="think " * 100,
             reasoning_source="details_text", reasoning_detail_types=["reasoning.text"],
             reasoning_tokens=120, provider=provider, finish_reason="stop")
    d.update(kw)
    return LLMResult(**d)


@pytest.fixture
def env(tmp_path, monkeypatch, mini_sqlite):
    monkeypatch.setattr(config, "out_root", lambda limit: tmp_path / "outputs")
    monkeypatch.setattr(runner, "db_path", lambda db_id: mini_sqlite)
    return [Example(i, f"q{i}", "SELECT 1", "mini") for i in range(12)]


# Feature: spider-text-to-sql-benchmark, Property 29
# **Validates: Requirements 21.2, 21.3, 21.4, 21.5, 21.6**
@settings(max_examples=60, deadline=None)
@given(n=st.integers(1, 10), data=st.data())
def test_property29_checks(n, data):
    cfg = make_cfg()
    recs = [good_record(i) for i in range(n)]
    assert all(c["passed"] for c in preflight.check_records(recs, cfg))
    name = data.draw(st.sampled_from(sorted(MUTATIONS)))
    idx = data.draw(st.integers(0, n - 1))
    recs[idx].update(MUTATIONS[name])
    res = {c["name"]: c for c in preflight.check_records(recs, cfg)}
    assert not res[name]["passed"] and res[name]["failing_ids"] == [idx]
    assert not all(c["passed"] for c in res.values())


def test_encrypted_only_fails_but_text_plus_summary_passes():
    cfg = make_cfg()
    r = good_record(0) | {"reasoning_detail_types": ["reasoning.encrypted"]}
    assert not {c["name"]: c for c in preflight.check_records([r], cfg)}["reasoning_raw_not_summary"]["passed"]
    r["reasoning_detail_types"] = ["reasoning.text", "reasoning.summary"]
    assert all(c["passed"] for c in preflight.check_records([r], cfg))


def test_no_pin_fails_provider_check():
    res = {c["name"]: c for c in preflight.check_records([good_record(0)], make_cfg(provider_order=None))}
    assert not res["provider_matches_pin"]["passed"]


# Feature: spider-text-to-sql-benchmark, Property 31
# **Validates: Requirements 21.1**
@given(pn=st.integers(5, 10), n=st.integers(1, 30))
def test_property31_sample_size(pn, n):
    exs = [Example(i, "q", "s", "d") for i in range(n)]
    got = preflight.sample_examples(make_cfg(preflight_n=pn), exs)
    assert [e.example_id for e in got] == list(range(min(pn, n)))


def test_report_roundtrip(tmp_path):
    r = preflight.PreflightReport("fp", {"a": 1}, ["X"], ["X"], [0, 1],
                                  [{"name": "api_ok", "passed": True, "failing_ids": [], "detail": ""}],
                                  True, "id", "ts")
    r.save(tmp_path / "p.json")
    assert preflight.PreflightReport.load(tmp_path / "p.json") == r


def test_pass_flow_and_gate(env):
    cfg = make_cfg()
    with pytest.raises(preflight.PreflightRequiredError, match="preflight.json"):
        runner.run(cfg, env, MockClient(lambda m, c: ok_result()), show_progress=False)
    client = MockClient(lambda m, c: ok_result())
    rep = preflight.run_preflight(cfg, env, client)
    assert rep.passed and rep.example_ids == [0, 1, 2, 3, 4] and len(client.calls) == 5
    assert rep.providers_served == ["DeepInfra"]
    assert preflight.raw_log_path(cfg).is_file() and not (cfg.output_dir() / "raw.jsonl").exists()
    assert preflight.require_passed(cfg).passed
    # Non-fingerprinted change keeps it current; fingerprinted change makes it stale.
    assert preflight.is_current(rep, dataclasses.replace(cfg, threads=7))
    assert not preflight.is_current(rep, dataclasses.replace(cfg, temperature=0.1))
    assert not preflight.is_current(rep, dataclasses.replace(cfg, provider_order=("Other",)))
    # Gate opens for the full run.
    stats = runner.run(cfg, env[:3], MockClient(lambda m, c: ok_result()), show_progress=False)
    assert stats.n_ok == 3


def test_fail_flow_then_fresh_pass(env, capsys):
    cfg = make_cfg()
    bad = MockClient(lambda m, c: ok_result(provider="OtherCo"))
    with pytest.raises(preflight.PreflightFailedError) as ei:
        preflight.run_preflight(cfg, env, bad)
    assert "provider_matches_pin" in str(ei.value) and "OtherCo" in capsys.readouterr().out
    rep = preflight.load_report(cfg)
    assert rep is not None and not rep.passed
    with pytest.raises(preflight.PreflightRequiredError):
        preflight.require_passed(cfg)
    stale_client = MockClient(lambda m, c: ok_result())
    with pytest.raises(preflight.PreflightRequiredError):
        runner.run(cfg, env, stale_client, show_progress=False)
    assert stale_client.calls == []
    # Earlier failed attempt stays in the log but does not affect a new run.
    rep2 = preflight.run_preflight(cfg, env, MockClient(lambda m, c: ok_result()))
    assert rep2.passed
