"""Unit tests for common/evaluate.py pure helpers and execute_sql."""
from __future__ import annotations

import pytest

from common.evaluate import (LEVELS, EvalRow, aggregate, compare_totals, execute_sql,
                             parse_cli_output)

CLI_SAMPLE = """\
                     easy                 medium               hard                 extra                all
count                4                    3                    2                    1                    10
=====================   EXECUTION ACCURACY     =====================
execution            1.000                0.667                0.500                0.000                0.700

====================== EXACT MATCHING ACCURACY =====================
exact match          0.750                0.333                0.000                0.000                0.400

---------------------PARTIAL MATCHING ACCURACY----------------------
select               1.000                1.000                1.000                1.000                1.000
"""


def _row(i, diff, ex, em):
    return EvalRow(i, "db", diff, ex, em, True, None, "ok", None, 1)


def test_parse_cli_output():
    out = parse_cli_output(CLI_SAMPLE)
    assert out["count"] == dict(zip(LEVELS, [4, 3, 2, 1, 10]))
    assert out["execution"]["all"] == 0.7
    assert out["exact match"]["easy"] == 0.75


def test_parse_cli_output_missing_rows():
    with pytest.raises(ValueError):
        parse_cli_output("nothing here")


def test_aggregate_and_compare():
    rows = ([_row(i, "easy", 1, int(i < 3)) for i in range(4)]
            + [_row(4, "medium", 1, 1), _row(5, "medium", 1, 0), _row(6, "medium", 0, 0)]
            + [_row(7, "hard", 1, 0), _row(8, "hard", 0, 0), _row(9, "extra", 0, 0)])
    t = aggregate(rows)
    assert t.loc["all", "count"] == 10 and t.loc["all", "ex"] == pytest.approx(0.7)
    assert compare_totals(t, parse_cli_output(CLI_SAMPLE)) == []
    cli = parse_cli_output(CLI_SAMPLE)
    cli["count"]["easy"] = 5
    cli["execution"]["all"] = 0.702
    mm = compare_totals(t, cli)
    assert {(m.metric, m.level) for m in mm} == {("count", "easy"), ("execution", "all")}


def test_aggregate_empty_level_is_zero():
    t = aggregate([_row(0, "easy", 1, 1)])
    assert t.loc["extra", "count"] == 0 and t.loc["extra", "ex"] == 0.0


def test_execute_sql_ok_error_timeout(mini_sqlite):
    r = execute_sql(mini_sqlite, "SELECT name FROM singer", 5)
    assert r.status == "ok" and r.n_rows == 3
    assert execute_sql(mini_sqlite, "SELECT nope FROM singer", 5).status == "error"
    slow = ("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) "
            "SELECT count(*) FROM c")
    assert execute_sql(mini_sqlite, slow, 0.5).status == "timeout"
    # read-only connection
    assert execute_sql(mini_sqlite, "DELETE FROM singer", 5).status == "error"
    assert execute_sql(mini_sqlite, "SELECT count(*) FROM singer", 5).n_rows == 1
