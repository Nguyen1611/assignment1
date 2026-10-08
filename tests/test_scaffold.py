import socket
import sqlite3

import pytest


def test_mini_sqlite_schema(mini_sqlite):
    conn = sqlite3.connect(mini_sqlite)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"singer", "concert", "empty_table", "numbers"} <= tables
    assert conn.execute("SELECT COUNT(*) FROM empty_table").fetchone()[0] == 0
    assert conn.execute('SELECT "concert name" FROM concert WHERE concert_id=1').fetchone()[0] == "Spring Gala"
    conn.close()


def test_network_blocked():
    with pytest.raises(RuntimeError):
        socket.socket().connect(("127.0.0.1", 9))
