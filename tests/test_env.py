"""Tests for common/env.py (Requirement 22)."""
from collections import Counter
from datetime import datetime

import pytest
from hypothesis import given, settings, strategies as st

from common import env


def test_collect_machine_fields():
    m = env.collect_machine()
    for k in ("os", "kernel", "is_wsl", "cpu_model", "logical_cpus", "ram_gib", "python"):
        assert k in m
    assert isinstance(m["is_wsl"], bool)


def test_collect_packages_missing_is_none():
    p = env.collect_packages(("pytest", "definitely-not-a-real-pkg-xyz"))
    assert p["pytest"] and p["definitely-not-a-real-pkg-xyz"] is None


def test_inference_info():
    i = env.inference_info()
    assert i["service"] == "OpenRouter" and i["remote_hardware_controlled"] is False


def test_track_run_success_and_failure(tmp_path):
    p = tmp_path / "environment.json"
    with env.track_run("ok", None, path=p):
        pass
    with pytest.raises(ValueError):
        with env.track_run("bad", 5, path=p):
            raise ValueError("boom")
    runs = {r["name"]: r for r in env.load_env(p)["runs"]}
    assert runs["ok"]["status"] == "completed"
    assert runs["bad"]["status"] == "failed"
    for r in runs.values():
        assert datetime.fromisoformat(r["started_at"]).tzinfo is not None
        assert datetime.fromisoformat(r["finished_at"]).tzinfo is not None


def test_record_models_merges(tmp_path):
    p = tmp_path / "environment.json"
    env.record_models("m", Counter({"A": 3}), [{"provider_name": "A", "quantization": "fp8"}], path=p)
    env.record_models("m", Counter({"B": 1}), [{"provider_name": "B", "quantization": "bf16"}], path=p)
    provs = env.load_env(p)["models"]["m"]["providers"]
    assert provs["A"] == {"calls": 3, "quantization": "fp8"}
    assert provs["B"]["quantization"] == "bf16"


def test_merge_env_keeps_earlier_runs_and_is_pure():
    old = {"machine": {"x": 1}, "runs": [{"name": "a", "started_at": "1"}]}
    new = {"machine": {"x": 2}, "runs": [{"name": "b", "started_at": "2"}]}
    old_copy = {"machine": {"x": 1}, "runs": [{"name": "a", "started_at": "1"}]}
    out = env.merge_env(old, new)
    assert old == old_copy
    assert out["machine"] == {"x": 2}
    assert [r["name"] for r in out["runs"]] == ["a", "b"]


json_leaf = st.none() | st.booleans() | st.integers() | st.text(max_size=10) | st.floats(allow_nan=False, allow_infinity=False)
json_val = st.recursive(json_leaf, lambda c: st.lists(c, max_size=4) | st.dictionaries(st.text(max_size=8), c, max_size=4), max_leaves=15)


@settings(max_examples=50, deadline=None)
@given(st.dictionaries(st.text(max_size=8), json_val, max_size=6))
def test_save_load_round_trip(tmp_path_factory, data):
    """**Validates: Requirements 22.9**"""
    p = tmp_path_factory.mktemp("env") / "environment.json"
    env.save_env(p, data)
    assert env.load_env(p) == data
