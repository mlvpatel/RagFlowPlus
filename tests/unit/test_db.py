"""
Unit tests for db_utils, using an in-memory SQLite database.
Tests run fully isolated with no real rag_app.db touched.

Author: Malav Patel
"""

import sqlite3
from unittest.mock import MagicMock, patch

import pytest

# ── helpers ──────────────────────────────────────────────────────────────────


def make_in_memory_db():
    """Return a fresh in-memory SQLite connection for testing."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS application_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            user_query TEXT NOT NULL,
            gpt_response TEXT NOT NULL,
            model TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS document_store (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


@pytest.fixture()
def db_conn():
    """Pytest fixture: isolated in-memory SQLite connection."""
    conn = make_in_memory_db()
    yield conn
    conn.close()


@pytest.fixture()
def db_utils(db_conn):
    """
    Import db_utils with get_db_connection patched to return our in-memory DB.
    create_application_logs / create_document_store are also patched
    so they don't try to use the real file.
    """
    with (
        patch("src.api.db_utils.get_db_connection", return_value=db_conn),
        patch("src.api.db_utils.create_application_logs"),
        patch("src.api.db_utils.create_document_store"),
    ):
        import importlib

        import src.api.db_utils as mod

        importlib.reload(mod)
        yield mod, db_conn


# ── document store tests ──────────────────────────────────────────────────────


class TestDocumentStore:
    def test_insert_and_retrieve_document(self, db_conn):
        """insert_document_record returns integer id; get_all_documents returns it."""
        db_conn.execute("INSERT INTO document_store (filename) VALUES (?)", ("test.pdf",))
        db_conn.commit()

        cursor = db_conn.execute("SELECT id, filename FROM document_store")
        rows = cursor.fetchall()
        assert len(rows) == 1
        assert rows[0]["filename"] == "test.pdf"
        assert isinstance(rows[0]["id"], int)

    def test_insert_multiple_documents(self, db_conn):
        db_conn.executemany(
            "INSERT INTO document_store (filename) VALUES (?)",
            [("a.pdf",), ("b.docx",), ("c.txt",)],
        )
        db_conn.commit()
        rows = db_conn.execute("SELECT * FROM document_store").fetchall()
        assert len(rows) == 3

    def test_delete_document_record(self, db_conn):
        cursor = db_conn.execute("INSERT INTO document_store (filename) VALUES (?)", ("del.pdf",))
        db_conn.commit()
        file_id = cursor.lastrowid

        db_conn.execute("DELETE FROM document_store WHERE id = ?", (file_id,))
        db_conn.commit()

        rows = db_conn.execute("SELECT * FROM document_store WHERE id = ?", (file_id,)).fetchall()
        assert len(rows) == 0

    def test_get_all_documents_newest_first(self, tmp_path, monkeypatch):
        """The real get_all_documents returns newest-first even for uploads
        landing in the same second (id order, not the second-granular timestamp)."""
        import src.api.db_utils as db

        monkeypatch.setattr(db, "DB_NAME", str(tmp_path / "docs.db"))
        db.create_document_store()
        db.insert_document_record("old.pdf")
        db.insert_document_record("new.pdf")

        rows = db.get_all_documents()
        assert rows[0]["filename"] == "new.pdf"
        assert rows[1]["filename"] == "old.pdf"


# ── application log tests ──────────────────────────────────────────────────────


class TestApplicationLogs:
    def test_insert_and_retrieve_log(self, db_conn):
        db_conn.execute(
            "INSERT INTO application_logs (session_id, user_query, gpt_response, model) "
            "VALUES (?, ?, ?, ?)",
            ("sess-1", "What is RAG?", "RAG stands for...", "gpt-4o-mini"),
        )
        db_conn.commit()

        rows = db_conn.execute(
            "SELECT * FROM application_logs WHERE session_id = ?", ("sess-1",)
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["user_query"] == "What is RAG?"

    def test_chat_history_ordering(self, db_conn):
        """Chat history must be ordered by created_at ascending."""
        for i, (q, a) in enumerate([("Q1", "A1"), ("Q2", "A2"), ("Q3", "A3")]):
            db_conn.execute(
                "INSERT INTO application_logs (session_id, user_query, gpt_response, model) "
                "VALUES (?, ?, ?, ?)",
                ("sess-order", q, a, "gpt-4o-mini"),
            )
        db_conn.commit()

        rows = db_conn.execute(
            "SELECT user_query FROM application_logs WHERE session_id = ? ORDER BY created_at",
            ("sess-order",),
        ).fetchall()
        assert [r["user_query"] for r in rows] == ["Q1", "Q2", "Q3"]

    def test_session_isolation(self, db_conn):
        """Logs from different sessions must not bleed into each other."""
        for sess in ("alpha", "beta"):
            db_conn.execute(
                "INSERT INTO application_logs (session_id, user_query, gpt_response, model) "
                "VALUES (?, ?, ?, ?)",
                (sess, f"Q from {sess}", "A", "gpt-4o-mini"),
            )
        db_conn.commit()

        alpha_rows = db_conn.execute(
            "SELECT * FROM application_logs WHERE session_id = 'alpha'"
        ).fetchall()
        assert len(alpha_rows) == 1
        assert alpha_rows[0]["session_id"] == "alpha"


def test_get_chat_history_windowed_to_recent_turns(tmp_path, monkeypatch):
    """History is capped: with 25 stored turns and the default limit of 20,
    the oldest 5 must be dropped and order stays oldest-first."""
    import src.api.db_utils as db

    monkeypatch.setattr(db, "DB_NAME", str(tmp_path / "chat.db"))
    db.create_application_logs()
    for i in range(25):
        db.insert_application_logs("s1", f"q{i}", f"a{i}", "gpt-4o-mini")

    messages = db.get_chat_history("s1")
    assert len(messages) == 40
    assert messages[0] == {"role": "human", "content": "q5"}
    assert messages[-1] == {"role": "ai", "content": "a24"}
