import json

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from common import config, traces
from common.data import Example


# ------------------------------------------------------------- markers
def test_count_markers_basic():
    t = "Wait, actually... Let me   check.\nHmm. double-check alternatively"
    assert traces.count_markers(t) == 6


def test_count_markers_word_boundary():
    assert traces.count_markers("awaited factually hmmm await_x") == 0


def test_multiword_any_whitespace():
    assert traces.count_markers("let\n\tme  check") == 1


@settings(max_examples=100)
@given(st.text(max_size=200), st.sampled_from(["wait", "actually", "hmm"]))
def test_marker_inside_word_unchanged(t, m):
    # Validates: Requirements 14.6
    assert traces.count_markers(t + " x" + m + "y ") == traces.count_markers(t + " ")


@settings(max_examples=100)
@given(st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), max_size=200))
def test_marker_case_insensitive(t):
    # Validates: Requirements 14.7
    assert traces.count_markers(t.upper()) == traces.count_markers(t.lower())


# ------------------------------------------------------------- drafts
def test_count_sql_drafts():
    t = ("I'll select the name from singer.\n```sql\nSELECT name FROM singer\n```\n"
         "Or SELECT x FROM y. WITH a AS (SELECT 1) ...\n```\nnot sql\n```")
    # 1 fenced + inline SELECT..FROM + WITH a AS( ; prose lowercase not counted
    assert traces.count_sql_drafts(t) == 3


# ------------------------------------------------------------- coverage
def test_coverage():
    assert traces.coverage("uses Singer and name_x", {"singer", "name"}) == 0.5
    assert traces.coverage("anything", set()) is None


@settings(max_examples=100)
@given(st.text(max_size=100), st.sets(st.text(min_size=1, max_size=8), max_size=5))
def test_coverage_in_range(t, items):
    # Validates: Requirements 14.8
    c = traces.coverage(t, items)
    assert c is None or 0.0 <= c <= 1.0


# ------------------------------------------------------------- exclusion
EX = Example(0, "q", "SELECT 1", "db")


@pytest.mark.parametrize("rec,reason", [
    ({"reasoning": "wait", "reasoning_missing": True, "reasoning_tokens": 5}, "reasoning_missing"),
    ({"reasoning": "  \n", "reasoning_tokens": 5}, "empty_trace"),
])
def test_exclusion(rec, reason):
    f = traces.extract_features(rec, EX, (set(), set()), token_counter=len)
    assert f.excluded and f.reason == reason and f.length_tokens is None


def test_extract_features_ok():
    rec = {"reasoning": "Wait. ```sql\nSELECT name FROM singer\n```", "reasoning_tokens": 500}
    f = traces.extract_features(rec, EX, ({"singer"}, {"name", "age"}), token_counter=lambda t: 500)
    assert not f.excluded
    # length comes from the text counter; the provider figure is kept separately
    assert f.length_tokens == 500 and f.provider_reasoning_tokens == 500
    assert f.length_chars == len(rec["reasoning"])
    assert (f.marker_count, f.markers_per_1k, f.sql_drafts) == (1, 2.0, 1)
    assert (f.table_coverage, f.column_coverage) == (1.0, 0.5)


def test_length_independent_of_provider_tokens():
    # Provider figure missing or zero must not exclude the trace; length is the text count.
    for ptok in (None, 0):
        rec = {"reasoning": "Wait. hmm", "reasoning_tokens": ptok}
        f = traces.extract_features(rec, EX, (set(), set()), token_counter=lambda t: 10)
        assert not f.excluded and f.length_tokens == 10
        assert f.provider_reasoning_tokens == ptok


# ------------------------------------------------------------- gold items (real data)
def _real_available():
    return (config.DATA_DIR / "dev.json").is_file() and (config.DATA_DIR / "tables.json").is_file()


@pytest.mark.skipif(not _real_available(), reason="Spider data not present")
def test_gold_schema_items_real():
    from common.data import load_examples
    from common.evaluate import import_official, official_cwd

    api = import_official()
    tj = json.loads((config.DATA_DIR / "tables.json").read_text(encoding="utf-8"))
    exs = load_examples()
    by_q = {e.gold_sql: e for e in exs}
    q1 = "SELECT count(*) FROM singer"
    with official_cwd():
        t, c = traces.gold_schema_items(q1, "concert_singer", api, tj)
    assert t == {"singer"} and c == set()
    q2 = ("SELECT T2.name ,  T2.capacity FROM concert AS T1 JOIN stadium AS T2 ON "
          "T1.stadium_id  =  T2.stadium_id WHERE T1.year  >=  2014 GROUP BY T2.stadium_id "
          "ORDER BY count(*) DESC LIMIT 1")
    with official_cwd():
        t, c = traces.gold_schema_items(q2, "concert_singer", api, tj)
    assert t == {"concert", "stadium"}
    assert c == {"Name", "Capacity", "Stadium_ID", "Year"}
    # nested subquery
    q3 = "SELECT name FROM stadium WHERE stadium_id NOT IN (SELECT stadium_id FROM concert)"
    with official_cwd():
        t, c = traces.gold_schema_items(q3, "concert_singer", api, tj)
    assert t == {"stadium", "concert"} and c == {"Name", "Stadium_ID"}
    assert by_q  # dev loaded


# ------------------------------------------------------------- statistics
@settings(max_examples=100, deadline=None)
@given(st.lists(st.floats(-1e3, 1e3), min_size=1, max_size=20),
       st.lists(st.floats(-1e3, 1e3), min_size=1, max_size=20))
def test_rank_biserial_range_and_sign(x, y):
    # Validates: Requirements 15.8
    r = traces.rank_biserial(x, y)
    assert -1 - 1e-12 <= r <= 1 + 1e-12
    assert r == pytest.approx(-traces.rank_biserial(y, x), abs=1e-9)


@settings(max_examples=100, deadline=None)
@given(st.lists(st.floats(0, 1), min_size=1, max_size=15))
def test_holm_valid(p):
    # Validates: Requirements 15.4
    adj = traces.holm(p)
    assert all(a >= r - 1e-12 and a <= 1 for a, r in zip(adj, p))


def test_rank_biserial_extremes():
    assert traces.rank_biserial([5, 6], [1, 2]) == 1.0
    assert traces.rank_biserial([1, 2], [5, 6]) == -1.0


def _synthetic(n=400, seed=0, separable=False):
    rng = np.random.default_rng(seed)
    diff = rng.choice(traces.DIFFICULTY_ORDER, n)
    rank = pd.Series(diff).map(traces.DIFFICULTY_RANK).to_numpy()
    length = rng.normal(1000 + 300 * rank, 200)
    markers = rng.poisson(3 + rank).astype(float)
    df = pd.DataFrame({
        "difficulty": diff, "length_tokens": length, "length_chars": length * 4, "marker_count": markers,
        "markers_per_1k": markers / length * 1000, "sql_drafts": rng.poisson(2, n),
        "table_coverage": rng.uniform(0, 1, n), "column_coverage": rng.uniform(0, 1, n),
    })
    if separable:
        df["ex"] = (df["length_tokens"] < df["length_tokens"].median()).astype(int)
    else:
        lin = 2.0 - 0.002 * (length - 1000) - 0.3 * (rank - 1)
        df["ex"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-lin))).astype(int)
    df.loc[:9, "column_coverage"] = np.nan  # dropped rows
    return df


def test_stats_tables_synthetic():
    df = _synthetic()
    feats = list(traces.FEATURES)
    d = traces.describe(df, feats, "ex")
    assert set(d.columns) >= {"count", "mean", "median", "q1", "q3", "iqr"}
    mw = traces.mann_whitney_table(df, feats)
    row = mw.set_index("feature").loc["length_tokens"]
    assert row.p_raw < 0.01 and row.rank_biserial < 0  # correct ones are shorter
    assert mw.set_index("feature").loc["column_coverage", "n_dropped"] == 10
    assert (mw.p_holm >= mw.p_raw).all()
    kw = traces.kruskal_table(df, feats)
    assert kw.set_index("feature").loc["length_tokens", "p_raw"] < 1e-6
    sp = traces.spearman_table(df, feats)
    assert sp.set_index("feature").loc["length_tokens", "rho"] > 0.3


def test_fit_logit_mle():
    res = traces.fit_logit(_synthetic())
    assert not res.regularized and res.n == 390
    t = res.table.set_index("term")
    assert t.loc["z_length_tokens", "coef"] < 0
    assert {"coef", "se", "z", "p", "odds_ratio", "or_ci_low", "or_ci_high", "n"} <= set(t.columns)
    assert any("difficulty" in i for i in t.index)


def test_fit_logit_regularized_fallback():
    res = traces.fit_logit(_synthetic(separable=True))
    assert res.regularized and "regularized" in res.label.lower()
