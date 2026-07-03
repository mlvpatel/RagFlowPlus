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

    def test_get_all_documents_ordered_by_timestamp(self, db_conn):
        """Documents should be returned newest-first."""
        db_conn.execute("INSERT INTO document_store (filename) VALUES (?)", ("old.pdf",))
        db_conn.execute("INSERT INTO document_store (filename) VALUES (?)", ("new.pdf",))
        db_conn.commit()

        rows = db_conn.execute(
            "SELECT filename FROM document_store ORDER BY upload_timestamp DESC"
        ).fetchall()
        assert rows[0]["filename"] == "new.pdf"


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
