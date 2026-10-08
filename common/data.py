"""Spider dev data access: examples, database paths, gold files (Req 3, 18.3)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from common import config

EXPECTED_DEV_COUNT = 1034


@dataclass(frozen=True)
class Example:
    example_id: int
    question: str
    gold_sql: str
    db_id: str


class DatabaseNotFoundError(FileNotFoundError):
    """Raised when the .sqlite file for a db_id is missing (Req 3.4)."""

    def __init__(self, db_id: str, path: Path):
        self.db_id = db_id
        self.path = Path(path)
        super().__init__(f"Database '{db_id}' not found at {self.path}")


def _data_dir() -> Path:
    # Read at call time so tests can monkeypatch config.DATA_DIR.
    return Path(config.DATA_DIR)


def load_examples(limit: int | None = None) -> list[Example]:
    """Load dev.json in file order with ids 0..n-1; first `limit` if given."""
    with open(_data_dir() / "dev.json", encoding="utf-8") as f:
        raw = json.load(f)
    if limit is not None:
        raw = raw[:limit]
    return [
        Example(example_id=i, question=r["question"], gold_sql=r["query"], db_id=r["db_id"])
        for i, r in enumerate(raw)
    ]


def db_path(db_id: str) -> Path:
    p = _data_dir() / "database" / db_id / f"{db_id}.sqlite"
    if not p.is_file():
        raise DatabaseNotFoundError(db_id, p)
    return p


def _raw_gold_lines() -> list[str]:
    with open(_data_dir() / "dev_gold.sql", encoding="utf-8") as f:
        return [ln.rstrip("\r\n") for ln in f if ln.strip()]


def gold_lines(limit: int | None = None) -> list[tuple[str, str]]:
    """(sql, db_id) pairs from dev_gold.sql (tab-separated, db_id last)."""
    out = []
    for ln in _raw_gold_lines():
        sql, _, db = ln.rpartition("\t")
        out.append((sql.strip(), db.strip()))
    return out if limit is None else out[:limit]


def write_gold(path: Path, limit: int | None) -> Path:
    """Write the first `limit` lines (all if None) of dev_gold.sql to `path`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = gold_lines(limit)
    path.write_text("".join(f"{s}\t{d}\n" for s, d in lines), encoding="utf-8")
    return path


def verify_dataset() -> dict:
    """Check count, .sqlite presence and dev.json/dev_gold.sql db_id alignment."""
    examples = load_examples()
    gold = gold_lines()
    db_ids = sorted({e.db_id for e in examples})
    missing = []
    for d in db_ids:
        try:
            db_path(d)
        except DatabaseNotFoundError as e:
            missing.append(str(e.path))
    mismatches = [
        i for i, (e, (_, g)) in enumerate(zip(examples, gold)) if e.db_id != g
    ]
    problems = []
    if len(examples) != EXPECTED_DEV_COUNT:
        problems.append(f"dev.json has {len(examples)} examples, expected {EXPECTED_DEV_COUNT}")
    if len(gold) != len(examples):
        problems.append(f"dev_gold.sql has {len(gold)} lines, dev.json has {len(examples)}")
    if missing:
        problems.append(f"{len(missing)} missing databases: {missing[:5]}")
    if mismatches:
        problems.append(f"db_id mismatch at rows {mismatches[:10]}")
    if problems:
        raise AssertionError("; ".join(problems))
    return {"n_examples": len(examples), "n_gold": len(gold),
            "n_databases": len(db_ids), "missing": missing, "mismatches": mismatches}
