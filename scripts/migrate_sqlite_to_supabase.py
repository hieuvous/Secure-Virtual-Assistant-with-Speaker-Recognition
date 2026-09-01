"""Safely migrate the project's SQLite data to the Supabase schema.

The script is deliberately opt-in: importing it does nothing, and a migration
aborts if any target table already contains rows unless --allow-existing is set.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import project_path
from src.database.supabase_client import get_supabase_client
from src.speaker.embedding import embedding_to_numpy, embedding_to_pgvector


# Determined from src/database/schema.sql and supabase/schema.sql. The order
# puts all parents before every table that references users.
TABLE_ORDER = (
    "users",
    "course_rooms",
    "enrollment_samples",
    "tasks",
    "schedules",
    "private_notes",
    "speaker_profiles",
    "audit_logs",
)
USER_FK_TABLES = {
    "enrollment_samples",
    "tasks",
    "schedules",
    "private_notes",
    "speaker_profiles",
    "audit_logs",
}
# Both SQLite and Supabase schemas declare these audit fields nullable. A
# non-finite similarity has no PostgreSQL JSON representation; NULL preserves
# the meaning "no usable numeric score" without inventing a score.
NULLABLE_NONFINITE_FIELDS = {
    "audit_logs": {"similarity_score", "threshold"},
}


class JsonPayloadValidationError(ValueError):
    def __init__(self, table: str, row_id: Any, field: str, value: Any, reason: str):
        self.table = table
        self.row_id = row_id
        self.field = field
        self.value = value
        super().__init__(
            "Invalid JSON numeric value:\n"
            f"table={table}\n"
            f"id={row_id}\n"
            f"field={field}\n"
            f"value={value!r}\n"
            f"reason={reason}"
        )


@dataclass
class MigrationReport:
    source_counts: Counter = field(default_factory=Counter)
    planned_counts: Counter = field(default_factory=Counter)
    target_before_counts: Counter = field(default_factory=Counter)
    target_counts: Counter = field(default_factory=Counter)
    warnings: list[str] = field(default_factory=list)
    embeddings_migrated: int = 0
    embeddings_missing: int = 0
    embeddings_invalid: int = 0
    normalized_nonfinite: list[str] = field(default_factory=list)
    inserted_counts: Counter = field(default_factory=Counter)
    failed_table: str | None = None
    failed_batch_start: int | None = None


class PartialMigrationError(RuntimeError):
    def __init__(self, table: str, batch_start: int, row_ids: list[Any], cause: Exception):
        self.table = table
        self.batch_start = batch_start
        self.row_ids = row_ids
        self.cause = cause
        super().__init__(
            f"Migration stopped while inserting table={table}, batch_start={batch_start}, "
            f"row_ids={row_ids}. Earlier acknowledged batches may already exist; "
            f"the failed batch status is unknown. Cause: {cause}"
        )


def open_sqlite_readonly(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise RuntimeError(f"SQLite database was not found: {path}")
    connection = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def sqlite_tables(connection: sqlite3.Connection) -> set[str]:
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return {row["name"] for row in rows}


def read_sqlite_rows(connection: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    return [dict(row) for row in connection.execute(f'SELECT * FROM "{table}" ORDER BY id')]


def resolve_embedding_path(embedding_path: str, project_root: Path = ROOT) -> Path:
    path = Path(embedding_path)
    return path if path.is_absolute() else project_root / path


def profile_payload_from_sqlite_row(
    row: dict[str, Any], user_id_map: dict[int, int], project_root: Path = ROOT
) -> tuple[dict[str, Any] | None, str | None]:
    """Convert a SQLite profile to a pgvector payload, retaining explicit IDs."""
    embedding_file = resolve_embedding_path(str(row["embedding_path"]), project_root)
    user_id = int(row["user_id"])
    if not embedding_file.is_file():
        return None, f"speaker_profiles id={row['id']} user_id={user_id}: missing embedding {embedding_file}"

    try:
        embedding = embedding_to_numpy(np.load(embedding_file, allow_pickle=False))
    except (OSError, ValueError, TypeError) as exc:
        return None, (
            f"speaker_profiles id={row['id']} user_id={user_id}: invalid embedding "
            f"{embedding_file} ({exc})"
        )

    return {
        "id": int(row["id"]),
        "user_id": user_id_map[user_id],
        "embedding": embedding_to_pgvector(embedding),
        "num_samples": int(row["num_samples"]),
        "model_version": row["model_version"],
        "enrollment_method": row["enrollment_method"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }, None


def payload_from_sqlite_row(
    table: str, row: dict[str, Any], user_id_map: dict[int, int]
) -> dict[str, Any]:
    """Map equal-shape tables and remap their user foreign key if required."""
    if table == "speaker_profiles":
        raise ValueError("Speaker profiles must be converted with their embedding file.")

    payload = dict(row)
    if table in USER_FK_TABLES and payload.get("user_id") is not None:
        old_user_id = int(payload["user_id"])
        if old_user_id not in user_id_map:
            raise ValueError(f"{table} id={row['id']} references missing user_id={old_user_id}")
        payload["user_id"] = user_id_map[old_user_id]
    return payload


def _json_safe_value(
    value: Any, *, table: str, row_id: Any, field: str, root_field: str,
    report: MigrationReport,
) -> Any:
    """Return JSON-safe data or raise with table/id/field/value context."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if root_field in NULLABLE_NONFINITE_FIELDS.get(table, set()):
            report.normalized_nonfinite.append(
                f"table={table} id={row_id} field={field} value={value!r} -> NULL "
                "(nullable audit numeric field)"
            )
            return None
        raise JsonPayloadValidationError(
            table, row_id, field, value, "non-finite values are not allowed for this field"
        )
    if isinstance(value, dict):
        return {
            str(key): _json_safe_value(
                item, table=table, row_id=row_id, field=f"{field}.{key}",
                root_field=root_field, report=report,
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _json_safe_value(
                item, table=table, row_id=row_id, field=f"{field}[{index}]",
                root_field=root_field, report=report,
            )
            for index, item in enumerate(value)
        ]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise JsonPayloadValidationError(
        table, row_id, field, value, f"unsupported JSON value type {type(value).__name__}"
    )


def validate_json_payloads(
    payloads: dict[str, list[dict[str, Any]]], report: MigrationReport
) -> None:
    """Validate every row before any Supabase request, including dry runs."""
    for table, rows in payloads.items():
        for index, row in enumerate(rows):
            row_id = row.get("id", f"row-index-{index}")
            normalized = {
                key: _json_safe_value(
                    value, table=table, row_id=row_id, field=key,
                    root_field=key, report=report,
                )
                for key, value in row.items()
            }
            try:
                json.dumps(normalized, allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise JsonPayloadValidationError(
                    table, row_id, "<payload>", normalized, str(exc)
                ) from exc
            rows[index] = normalized


def prepare_payloads(
    connection: sqlite3.Connection, project_root: Path = ROOT
) -> tuple[dict[str, list[dict[str, Any]]], MigrationReport, dict[int, int]]:
    actual_tables = sqlite_tables(connection)
    expected_tables = set(TABLE_ORDER)
    missing = expected_tables - actual_tables
    unexpected = actual_tables - expected_tables
    if missing or unexpected:
        raise RuntimeError(
            "SQLite schema does not match the migration schema. "
            f"Missing={sorted(missing)}, unexpected={sorted(unexpected)}."
        )

    rows_by_table = {table: read_sqlite_rows(connection, table) for table in TABLE_ORDER}
    report = MigrationReport(source_counts=Counter({
        table: len(rows) for table, rows in rows_by_table.items()
    }))

    # GENERATED BY DEFAULT AS IDENTITY accepts explicit IDs. Identity mapping is
    # retained as an explicit map so all FK payloads remain safe if this changes.
    user_id_map = {int(row["id"]): int(row["id"]) for row in rows_by_table["users"]}
    payloads: dict[str, list[dict[str, Any]]] = {table: [] for table in TABLE_ORDER}

    for table in TABLE_ORDER:
        for row in rows_by_table[table]:
            if table == "speaker_profiles":
                payload, warning = profile_payload_from_sqlite_row(row, user_id_map, project_root)
                if warning:
                    report.warnings.append(warning)
                    if "missing embedding" in warning:
                        report.embeddings_missing += 1
                    else:
                        report.embeddings_invalid += 1
                    continue
                report.embeddings_migrated += 1
            else:
                payload = payload_from_sqlite_row(table, row, user_id_map)
            payloads[table].append(payload)

    report.planned_counts = Counter({table: len(rows) for table, rows in payloads.items()})
    validate_json_payloads(payloads, report)
    return payloads, report, user_id_map


def target_count(client, table: str) -> int:
    response = client.table(table).select("id", count="exact", head=True).execute()
    return int(response.count or 0)


def verify_supabase_schema(client) -> dict[str, int]:
    """Check every expected target table is accessible before any insert."""
    counts = {}
    for table in TABLE_ORDER:
        try:
            counts[table] = target_count(client, table)
            if table == "speaker_profiles":
                client.table(table).select("id, user_id, embedding").limit(1).execute()
        except Exception as exc:
            raise RuntimeError(
                f"Supabase table/schema check failed for '{table}'. Apply supabase/schema.sql first."
            ) from exc
    return counts


def insert_payloads(client, payloads: dict[str, list[dict[str, Any]]], report: MigrationReport) -> None:
    for table in TABLE_ORDER:
        rows = payloads[table]
        for start in range(0, len(rows), 500):
            batch = rows[start:start + 500]
            try:
                client.table(table).insert(batch).execute()
            except Exception as exc:
                report.failed_table = table
                report.failed_batch_start = start
                raise PartialMigrationError(table, start, [row.get("id") for row in batch], exc) from exc
            report.inserted_counts[table] += len(batch)


def print_report(report: MigrationReport) -> None:
    print("\nSQLite source counts:")
    for table in TABLE_ORDER:
        print(f"  {table}: {report.source_counts[table]}")
    print("Supabase planned/actual counts:")
    for table in TABLE_ORDER:
        actual = report.target_counts.get(table)
        print(
            f"  {table}: before={report.target_before_counts[table]}, "
            f"planned={report.planned_counts[table]}, actual={actual}"
        )
    print(
        "Speaker embeddings: "
        f"migrated={report.embeddings_migrated}, "
        f"missing={report.embeddings_missing}, invalid_dimension_or_content={report.embeddings_invalid}"
    )
    if report.warnings:
        print("Warnings:")
        for warning in report.warnings:
            print(f"  WARNING: {warning}")
    if report.normalized_nonfinite:
        print("Non-finite values normalized to NULL:")
        for item in report.normalized_nonfinite:
            print(f"  WARNING: {item}")
    if report.inserted_counts or report.failed_table:
        print("Supabase insertion progress (acknowledged batches only):")
        for table in TABLE_ORDER:
            print(f"  {table}: {report.inserted_counts[table]}")
        if report.failed_table:
            print(
                f"  FAILED: table={report.failed_table}, "
                f"batch_start={report.failed_batch_start}"
            )


def run_migration(sqlite_path: Path, *, dry_run: bool, allow_existing: bool) -> MigrationReport:
    with open_sqlite_readonly(sqlite_path) as sqlite_connection:
        payloads, report, _ = prepare_payloads(sqlite_connection)

    client = get_supabase_client()
    existing_counts = verify_supabase_schema(client)
    report.target_before_counts = Counter(existing_counts)
    nonempty = {table: count for table, count in existing_counts.items() if count}

    if dry_run:
        report.target_counts = existing_counts
        print("DRY RUN: SQLite was read and embeddings/schema were validated. No rows were inserted.")
        if nonempty:
            print(
                "DRY RUN SAFETY: a real migration would abort because target tables "
                f"are non-empty: {nonempty}"
            )
        print_report(report)
        return report

    if nonempty and not allow_existing:
        raise RuntimeError(
            "Supabase target tables are not empty; aborting without changes. "
            f"Use --allow-existing only after reviewing: {nonempty}"
        )

    try:
        insert_payloads(client, payloads, report)
    except PartialMigrationError:
        try:
            report.target_counts = verify_supabase_schema(client)
        except Exception as count_error:
            report.warnings.append(f"Could not refresh Supabase counts after failure: {count_error}")
        print_report(report)
        raise
    report.target_counts = verify_supabase_schema(client)
    mismatches = {
        table: (
            report.target_before_counts[table] + report.planned_counts[table],
            report.target_counts[table],
        )
        for table in TABLE_ORDER
        if report.target_before_counts[table] + report.planned_counts[table]
        != report.target_counts[table]
    }
    print_report(report)
    if mismatches:
        raise RuntimeError(f"Migration verification failed; count mismatches: {mismatches}")
    print(
        "Migration completed. Run supabase/reset_identity_sequences.sql in the "
        "Supabase SQL Editor before creating new records."
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sqlite-path",
        type=Path,
        default=project_path("data/app.db"),
        help="Source SQLite database (default: data/app.db).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate only; never insert.")
    parser.add_argument(
        "--allow-existing",
        action="store_true",
        help="Allow inserts when one or more Supabase target tables are non-empty.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_migration(args.sqlite_path, dry_run=args.dry_run, allow_existing=args.allow_existing)


if __name__ == "__main__":
    main()
