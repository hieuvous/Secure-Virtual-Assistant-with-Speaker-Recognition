from pathlib import Path

import numpy as np
import pytest

from scripts.migrate_sqlite_to_supabase import (
    JsonPayloadValidationError,
    MigrationReport,
    PartialMigrationError,
    TABLE_ORDER,
    insert_payloads,
    payload_from_sqlite_row,
    profile_payload_from_sqlite_row,
    validate_json_payloads,
)
from src.speaker.embedding import embedding_to_numpy


def test_payload_remaps_user_foreign_key_and_preserves_primary_key():
    row = {"id": 12, "user_id": 4, "title": "NLP report", "due_date": None, "status": "pending"}

    payload = payload_from_sqlite_row("tasks", row, {4: 40})

    assert payload["id"] == 12
    assert payload["user_id"] == 40
    assert payload["title"] == "NLP report"


def test_profile_payload_loads_npy_and_serializes_pgvector(tmp_path):
    embedding_path = tmp_path / "data" / "users" / "4" / "speaker_embedding.npy"
    embedding_path.parent.mkdir(parents=True)
    vector = np.linspace(-1, 1, 192, dtype=np.float32)
    np.save(embedding_path, vector)
    row = {
        "id": 8,
        "user_id": 4,
        "embedding_path": "data/users/4/speaker_embedding.npy",
        "num_samples": 5,
        "model_version": "finetuned_epoch10",
        "enrollment_method": "mean",
        "created_at": "2026-01-01 00:00:00",
        "updated_at": "2026-01-01 00:00:00",
    }

    payload, warning = profile_payload_from_sqlite_row(row, {4: 4}, tmp_path)

    assert warning is None
    assert payload["id"] == 8
    assert payload["user_id"] == 4
    np.testing.assert_allclose(embedding_to_numpy(payload["embedding"]), vector)


def test_profile_payload_reports_missing_embedding(tmp_path):
    row = {
        "id": 8, "user_id": 4, "embedding_path": "data/users/4/missing.npy",
        "num_samples": 5, "model_version": "model", "enrollment_method": "mean",
        "created_at": None, "updated_at": None,
    }

    payload, warning = profile_payload_from_sqlite_row(row, {4: 4}, tmp_path)

    assert payload is None
    assert "missing embedding" in warning
    assert "user_id=4" in warning


def test_profile_payload_reports_invalid_embedding_dimension(tmp_path):
    embedding_path = tmp_path / "bad.npy"
    np.save(embedding_path, np.zeros(2, dtype=np.float32))
    row = {
        "id": 9, "user_id": 4, "embedding_path": "bad.npy",
        "num_samples": 5, "model_version": "model", "enrollment_method": "mean",
        "created_at": None, "updated_at": None,
    }

    payload, warning = profile_payload_from_sqlite_row(row, {4: 4}, tmp_path)

    assert payload is None
    assert "invalid embedding" in warning
    assert "192-D" in warning


def test_json_validation_normalizes_only_nullable_audit_scores():
    payloads = {
        "audit_logs": [{"id": 37, "similarity_score": float("nan"), "threshold": float("-inf")}],
    }
    report = MigrationReport()

    validate_json_payloads(payloads, report)

    assert payloads["audit_logs"][0]["similarity_score"] is None
    assert payloads["audit_logs"][0]["threshold"] is None
    assert "table=audit_logs id=37 field=similarity_score value=nan -> NULL" in report.normalized_nonfinite[0]
    assert "field=threshold value=-inf -> NULL" in report.normalized_nonfinite[1]


def test_json_validation_rejects_nonfinite_nonnullable_field():
    payloads = {"tasks": [{"id": 4, "title": float("inf")}]} 

    with pytest.raises(JsonPayloadValidationError, match="table=tasks") as exc:
        validate_json_payloads(payloads, MigrationReport())

    assert "id=4" in str(exc.value)
    assert "field=title" in str(exc.value)


def test_insert_failure_reports_table_batch_and_row_ids():
    class FailingQuery:
        def insert(self, rows):
            self.rows = rows
            return self

        def execute(self):
            raise RuntimeError("request failed")

    class FailingClient:
        def table(self, _table):
            return FailingQuery()

    report = MigrationReport()
    payloads = {table: [] for table in TABLE_ORDER}
    payloads["audit_logs"] = [{"id": 37}]

    with pytest.raises(PartialMigrationError, match="table=audit_logs"):
        insert_payloads(FailingClient(), payloads, report)

    assert report.failed_table == "audit_logs"
    assert report.failed_batch_start == 0
