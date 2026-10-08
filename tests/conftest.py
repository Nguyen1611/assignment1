"""Shared pytest configuration: skip markers, network blocking, and the mini.sqlite fixture."""
from __future__ import annotations

import os
import socket
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
SPIDER_DIR = ROOT / "data" / "spider"

# Make `common` importable without installing the package.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _dataset_present() -> bool:
    return (SPIDER_DIR / "dev.json").is_file() and (SPIDER_DIR / "database").is_dir()


def _official_present() -> bool:
    eval_dir = os.environ.get("SPIDER_EVAL_DIR")
    if not eval_dir:
        try:
            from dotenv import dotenv_values

            eval_dir = dotenv_values(ROOT / ".env").get("SPIDER_EVAL_DIR")
        except Exception:
            eval_dir = None
    if not eval_dir:
        from common.config import OFFICIAL_DIR  # default path

        eval_dir = str(OFFICIAL_DIR)
    return bool(eval_dir) and (Path(eval_dir) / "evaluation.py").is_file()


requires_dataset = pytest.mark.skipif(
    not _dataset_present(), reason="Spider dev set not downloaded to data/spider/"
)
requires_official = pytest.mark.skipif(
    not _official_present(), reason="Official evaluation script (SPIDER_EVAL_DIR) not available"
)


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_dataset: needs the downloaded Spider dev set")
    config.addinivalue_line("markers", "requires_official: needs the official evaluation script")


def pytest_collection_modifyitems(config, items):
    """Allow `@pytest.mark.requires_dataset` / `requires_official` as plain marks too."""
    for item in items:
        if "requires_dataset" in item.keywords and not _dataset_present():
            item.add_marker(pytest.mark.skip(reason="Spider dev set not downloaded"))
        if "requires_official" in item.keywords and not _official_present():
            item.add_marker(pytest.mark.skip(reason="Official evaluation script not available"))


@pytest.fixture(autouse=True)
def _block_network(monkeypatch):
    """Fail any accidental network connection during tests."""

    def guard(*args, **kwargs):
        raise RuntimeError("Network access is disabled in tests")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket.socket, "connect_ex", guard)


def build_mini_sqlite(path: Path) -> Path:
    """Build a small SQLite DB with PK/FK, a spaced name, long text, an empty table,
    and a table suitable for recursive-CTE timeout tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            PRAGMA foreign_keys = ON;
            CREATE TABLE singer (
                singer_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                country TEXT,
                age INTEGER
            );
            CREATE TABLE concert (
                concert_id INTEGER PRIMARY KEY,
                "concert name" TEXT,
                singer_id INTEGER,
                year INTEGER,
                notes TEXT,
                FOREIGN KEY (singer_id) REFERENCES singer(singer_id)
            );
            CREATE TABLE empty_table (
                id INTEGER PRIMARY KEY,
                value TEXT
            );
            CREATE TABLE numbers (n INTEGER PRIMARY KEY);
            """
        )
        conn.executemany(
            "INSERT INTO singer VALUES (?, ?, ?, ?)",
            [(1, "Alice", "France", 30), (2, "Bob", "USA", 45), (3, "Chen", "China", 28)],
        )
        long_text = "lorem ipsum " * 50
        conn.executemany(
            'INSERT INTO concert VALUES (?, ?, ?, ?, ?)',
            [
                (1, "Spring Gala", 1, 2014, long_text),
                (2, "Summer Fest", 2, 2015, "short"),
                (3, "Autumn Nights", 1, 2015, None),
            ],
        )
        conn.executemany("INSERT INTO numbers VALUES (?)", [(i,) for i in range(1, 11)])
        conn.commit()
    finally:
        conn.close()
    return path


@pytest.fixture(scope="session")
def mini_sqlite() -> Path:
    """Path to tests/fixtures/mini.sqlite, rebuilt once per session."""
    return build_mini_sqlite(FIXTURES / "mini.sqlite")
