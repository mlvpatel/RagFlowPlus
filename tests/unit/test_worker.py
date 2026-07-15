"""
Unit tests for the Celery ingestion task: record-before-index ordering,
temp-file cleanup, and the failure path.
"""

from unittest.mock import patch

import src.worker.tasks as tasks


def test_process_document_success_cleans_up(tmp_path):
    f = tmp_path / "up.txt"
    f.write_text("content")
    with (
        patch.object(tasks, "insert_document_record", return_value=42) as rec,
        patch.object(tasks, "index_document_to_chroma", return_value=True) as idx,
    ):
        result = tasks.process_document.apply(args=[str(f), "up.txt"]).get()

    assert result == {"status": "success", "file_id": 42, "filename": "up.txt"}
    rec.assert_called_once_with("up.txt")
    idx.assert_called_once_with(str(f), 42)
    assert not f.exists()


def test_process_document_reports_index_failure(tmp_path):
    f = tmp_path / "up.txt"
    f.write_text("content")
    with (
        patch.object(tasks, "insert_document_record", return_value=1),
        patch.object(tasks, "index_document_to_chroma", return_value=False),
    ):
        result = tasks.process_document.apply(args=[str(f), "up.txt"]).get()

    assert result["status"] == "failed"
    assert f.exists()
