"""Tests for common/egsc.py (Req 16). No network: fake LLM client + mini.sqlite."""
from __future__ import annotations

import json

import pytest
from hypothesis import given, strategies as st

from common import config as config_mod
from common import egsc
from common.config import RunConfig
from common.data import Example
from common.evaluate import ExecResult
from common.llm_client import LLMResult
from common.runner import read_predictions

GOLD = "SELECT secret_gold_marker FROM singer"


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, messages, cfg):
        self.calls.append(messages)
        text = self.replies.pop(0)
        return LLMResult(status="ok", text=text, prompt_tokens=10, completion_tokens=5,
                         latency_s=0.1, finish_reason="stop")


def _cfg(run_name="qwen25_7b_egsc", empty=False, **kw):
    return RunConfig(run_name=run_name, model_id="qwen/qwen-2.5-7b-instruct", temperature=0.0,
                     retry_on_empty=empty, **kw)


@pytest.fixture
def out(tmp_path, monkeypatch):
    monkeypatch.setattr(config_mod, "out_root", lambda limit: tmp_path)
    return tmp_path


def _run(cfg, preds, client, mini_sqlite):
    exs = [Example(i, f"q{i}", GOLD, "mini") for i in range(len(preds))]
    return egsc.run_egsc(cfg, exs, preds, client, db_resolver=lambda _: mini_sqlite)


# ---------------------------------------------------------------- pure rules
results = st.one_of(
    st.just(ExecResult("error", None, "boom")),
    st.just(ExecResult("timeout", None, "t")),
    st.integers(0, 3).map(lambda n: ExecResult("ok", n, None)),
)


@given(results, st.booleans())
def test_needs_retry_property(res, flag):
    """**Validates: Requirements 16.4, 16.12**"""
    expected = res.status in {"error", "timeout"} or (flag and res.n_rows == 0)
    assert egsc.needs_retry(res, flag) == expected


@given(st.lists(results, min_size=1, max_size=3), st.booleans())
def test_select_final_property(rs, flag):
    """**Validates: Requirements 16.6, 16.12**"""
    attempts = [(f"s{i}", r) for i, r in enumerate(rs)]
    sql, idx = egsc.select_final("s0", attempts, flag)
    assert sql == attempts[idx][0]
    ok = [i for i, r in enumerate(rs) if r.status == "ok"]
    if not ok:
        assert idx == 0
    elif flag and any(rs[i].n_rows for i in ok):
        assert idx == max(i for i in ok if rs[i].n_rows) 
    else:
        assert idx == ok[-1]


def test_feedback_text():
    assert egsc.feedback_text(ExecResult("error", None, "no such table: x")) == \
        "Execution error: no such table: x"
    assert egsc.feedback_text(ExecResult("timeout", None, None)) == \
        "Execution timed out after 5 seconds."
    assert egsc.feedback_text(ExecResult("ok", 0, None)) == "The query executed but returned 0 rows."


def test_run_name_guard(out):
    with pytest.raises(ValueError):
        egsc.check_run_name(_cfg("qwen25_7b_egsc", empty=True))
    with pytest.raises(ValueError):
        egsc.check_run_name(_cfg("qwen25_7b_egsc_empty", empty=False))
    egsc.check_run_name(_cfg("qwen25_7b_egsc_empty", empty=True))


# ---------------------------------------------------------------- loop
def test_primary_loop_and_resume(out, mini_sqlite):
    preds = ["SELECT name FROM singer",              # ok -> no call
             "SELECT * FROM empty_table",             # 0 rows -> accepted in primary
             "SELECT nope FROM singer",               # error -> fixed on retry 1
             "SELECT bad FROM singer"]                # error twice -> original kept
    client = FakeClient(["```sql\nSELECT age FROM singer\n```",
                         "```sql\nSELECT x FROM y\n```", "```sql\nSELECT z FROM y\n```"])
    stats = _run(_cfg(), preds, client, mini_sqlite)
    assert stats.n_llm_calls == 3 and stats.n_triggered == 2
    finals = read_predictions(out / "qwen25_7b_egsc" / "preds.txt")
    assert finals == [preds[0], preds[1], "SELECT age FROM singer", preds[3]]
    for msgs in client.calls:
        assert "secret_gold_marker" not in json.dumps(msgs)
    recs = [json.loads(l) for l in open(out / "qwen25_7b_egsc" / "raw.jsonl")]
    assert len(recs) == 4 and len(recs[3]["attempts"]) == 3
    a = recs[2]["attempts"][1]
    assert a["feedback"].startswith("Execution error:") and a["prompt_tokens"] == 10
    # resume: no new calls, no duplicate records
    stats2 = _run(_cfg(), preds, FakeClient([]), mini_sqlite)
    assert stats2.n_skipped == 4 and stats2.n_llm_calls == 0
    assert len(open(out / "qwen25_7b_egsc" / "raw.jsonl").read().splitlines()) == 4


def test_empty_variant(out, mini_sqlite):
    preds = ["SELECT * FROM empty_table"]
    client = FakeClient(["```sql\nSELECT name FROM singer\n```"])
    _run(_cfg("qwen25_7b_egsc_empty", empty=True), preds, client, mini_sqlite)
    assert "returned 0 rows" in client.calls[0][-1]["content"]
    assert read_predictions(out / "qwen25_7b_egsc_empty" / "preds.txt") == ["SELECT name FROM singer"]
