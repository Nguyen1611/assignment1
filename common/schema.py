"""Schema serialization: SQLite database -> prompt text (Requirement 4)."""
from __future__ import annotations

import sqlite3
from pathlib import Path

TRUNC_MARKER = "…[truncated]"


def _q(name: str) -> str:
    """Double-quote an SQL identifier, escaping embedded quotes."""
    return '"' + str(name).replace('"', '""') + '"'


def _fmt_value(value, value_trunc: int) -> str:
    if value is None:
        text = "NULL"
    elif isinstance(value, bytes):
        text = value.decode(errors="ignore")
    else:
        text = str(value)
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    if len(text) > value_trunc:
        text = text[:value_trunc] + TRUNC_MARKER
    return text


def _create_table(conn: sqlite3.Connection, table: str) -> str:
    cols = conn.execute(f"PRAGMA table_info({_q(table)})").fetchall()
    # cid, name, type, notnull, dflt_value, pk
    lines = [f"  {_q(c[1])} {c[2]}".rstrip() for c in cols]
    pk_cols = [c[1] for c in sorted((c for c in cols if c[5] > 0), key=lambda c: c[5])]
    if pk_cols:
        lines.append(f"  PRIMARY KEY ({', '.join(_q(c) for c in pk_cols)})")
    # id, seq, table, from, to, on_update, on_delete, match
    fks: dict[int, list] = {}
    for row in conn.execute(f"PRAGMA foreign_key_list({_q(table)})").fetchall():
        fks.setdefault(row[0], []).append(row)
    for fk_id in sorted(fks):
        rows = sorted(fks[fk_id], key=lambda r: r[1])
        src = ", ".join(_q(r[3]) for r in rows)
        ref = f"FOREIGN KEY ({src}) REFERENCES {_q(rows[0][2])}"
        if all(r[4] is not None for r in rows):
            ref += "(" + ", ".join(_q(r[4]) for r in rows) + ")"
        lines.append("  " + ref)
    return f"CREATE TABLE {_q(table)} (\n" + ",\n".join(lines) + "\n);"


def _sample_block(conn: sqlite3.Connection, table: str, sample_rows: int, value_trunc: int) -> str:
    cur = conn.execute(f"SELECT * FROM {_q(table)} LIMIT ?", (sample_rows,))
    header = "\t".join(_fmt_value(d[0], value_trunc) for d in cur.description)
    rows = ["\t".join(_fmt_value(v, value_trunc) for v in r) for r in cur.fetchall()]
    return "\n".join([f"/* {sample_rows} example rows from {_q(table)}: */", header, *rows])


def serialize_schema(db_file: Path, sample_rows: int = 3, value_trunc: int = 64) -> str:
    """Render CREATE TABLE statements (with PK/FK) plus sample rows for every user table.

    Opens the database read-only; output is deterministic for a given file.
    """
    uri = Path(db_file).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.text_factory = lambda b: b.decode(errors="ignore")
    try:
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite\\_%' ESCAPE '\\' ORDER BY rowid"
            ).fetchall()
        ]
        blocks = [
            _create_table(conn, t) + "\n" + _sample_block(conn, t, sample_rows, value_trunc)
            for t in tables
        ]
    finally:
        conn.close()
    return "\n\n".join(blocks) + "\n"
