"""
Database utilities for RAGFlow, SQLite with WAL mode.
Author: Malav Patel
"""

from dotenv import load_dotenv

load_dotenv()

import logging
import sqlite3
from datetime import datetime

logger = logging.getLogger(__name__)

DB_NAME = "rag_app.db"


def get_db_connection() -> sqlite3.Connection:
    """
    Open a SQLite connection with WAL journal mode and foreign keys enabled.

    WAL (Write-Ahead Logging) allows concurrent readers + one writer,
    eliminating lock contention that occurs with the default DELETE mode.
    """
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA synchronous=NORMAL;")  # Safe with WAL, faster than FULL
    return conn


def create_application_logs():
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS application_logs (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT    NOT NULL,
                user_query TEXT    NOT NULL,
                gpt_response TEXT  NOT NULL,
                model      TEXT    NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_session ON application_logs(session_id);")


def create_document_store():
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS document_store (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                filename         TEXT      NOT NULL,
                upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


def insert_application_logs(session_id: str, user_query: str, gpt_response: str, model: str):
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO application_logs (session_id, user_query, gpt_response, model) VALUES (?, ?, ?, ?)",
            (session_id, user_query, gpt_response, model),
        )


def get_chat_history(session_id: str) -> list:
    with get_db_connection() as conn:
        cursor = conn.execute(
            "SELECT user_query, gpt_response FROM application_logs "
            "WHERE session_id = ? ORDER BY created_at",
            (session_id,),
        )
        messages = []
        for row in cursor.fetchall():
            messages.extend(
                [
                    {"role": "human", "content": row["user_query"]},
                    {"role": "ai", "content": row["gpt_response"]},
                ]
            )
    return messages


def insert_document_record(filename: str) -> int:
    """Insert a document record and return the new integer file_id."""
    with get_db_connection() as conn:
        cursor = conn.execute("INSERT INTO document_store (filename) VALUES (?)", (filename,))
        return cursor.lastrowid


def delete_document_record(file_id: int) -> bool:
    try:
        with get_db_connection() as conn:
            conn.execute("DELETE FROM document_store WHERE id = ?", (file_id,))
        return True
    except Exception as e:
        logger.error(f"Failed to delete document record {file_id}: {e}")
        return False


def get_all_documents() -> list:
    with get_db_connection() as conn:
        cursor = conn.execute(
            "SELECT id, filename, upload_timestamp FROM document_store ORDER BY upload_timestamp DESC"
        )
        return [dict(row) for row in cursor.fetchall()]


# Initialize database tables on module load
create_application_logs()
create_document_store()
