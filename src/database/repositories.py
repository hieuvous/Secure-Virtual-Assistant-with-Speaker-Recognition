from __future__ import annotations

from src.database.db import connect


def create_user(name: str, student_code: str | None = None) -> int:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO users(name, student_code) VALUES (?, ?)",
            (name, student_code or None),
        )
        return int(cur.lastrowid)


def list_users() -> list[dict]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM users ORDER BY id").fetchall()
    return [dict(r) for r in rows]


def upsert_profile(
    user_id: int,
    embedding_path: str,
    num_samples: int,
    model_version: str,
    enrollment_method: str = "mean",
):
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO speaker_profiles(
                user_id, embedding_path, num_samples, model_version, enrollment_method
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                embedding_path=excluded.embedding_path,
                num_samples=excluded.num_samples,
                model_version=excluded.model_version,
                enrollment_method=excluded.enrollment_method,
                updated_at=CURRENT_TIMESTAMP
            """,
            (user_id, embedding_path, num_samples, model_version, enrollment_method),
        )


def list_profiles() -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT p.user_id, u.name, p.embedding_path, p.num_samples, p.model_version
            FROM speaker_profiles p
            JOIN users u ON u.id = p.user_id
            ORDER BY p.user_id
            """
        ).fetchall()
    return [dict(r) for r in rows]


def get_profile(user_id: int) -> dict | None:
    with connect() as conn:
        row = conn.execute(
            """
            SELECT p.*, u.name
            FROM speaker_profiles p
            JOIN users u ON u.id = p.user_id
            WHERE p.user_id=?
            """,
            (user_id,),
        ).fetchone()
    return dict(row) if row else None


def add_task(user_id: int, title: str, due_date: str | None = None):
    with connect() as conn:
        conn.execute(
            "INSERT INTO tasks(user_id, title, due_date) VALUES (?, ?, ?)",
            (user_id, title, due_date),
        )


def get_tasks(user_id: int) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM tasks WHERE user_id=? AND status='pending' ORDER BY due_date",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def add_private_note(user_id: int, title: str, content: str):
    with connect() as conn:
        conn.execute(
            "INSERT INTO private_notes(user_id, title, content) VALUES (?, ?, ?)",
            (user_id, title, content),
        )


def get_private_notes(user_id: int) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM private_notes WHERE user_id=? ORDER BY id",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def add_audit_log(
    user_id: int | None,
    intent: str,
    auth_method: str,
    similarity_score: float | None,
    threshold: float | None,
    result: str,
):
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO audit_logs(
                user_id, intent, auth_method, similarity_score, threshold, result
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_id, intent, auth_method, similarity_score, threshold, result),
        )
