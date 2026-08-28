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
            SELECT p.user_id, u.name, p.embedding_path, p.num_samples,
                   p.model_version, p.enrollment_method
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


def add_task(user_id: int, title: str, due_date: str | None = None) -> int:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO tasks(user_id, title, due_date) VALUES (?, ?, ?)",
            (user_id, title, due_date),
        )
        return int(cur.lastrowid)


def get_tasks(user_id: int) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM tasks WHERE user_id=? AND status='pending' ORDER BY due_date",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_task_by_title(user_id: int, title: str) -> int:
    with connect() as conn:
        cur = conn.execute(
            """
            DELETE FROM tasks
            WHERE user_id=? AND LOWER(title)=LOWER(?)
            """,
            (user_id, title.strip()),
        )
        return int(cur.rowcount)


def add_private_note(user_id: int, title: str, content: str) -> int:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO private_notes(user_id, title, content) VALUES (?, ?, ?)",
            (user_id, title, content),
        )
        return int(cur.lastrowid)


def get_private_notes(user_id: int) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM private_notes WHERE user_id=? ORDER BY id",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def add_schedule(
    user_id: int,
    subject: str,
    start_time: str,
    end_time: str | None = None,
    location: str | None = None,
) -> int:
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO schedules(user_id, subject, start_time, end_time, location)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, subject, start_time, end_time, location),
        )
        return int(cur.lastrowid)


def get_schedule(user_id: int, date_prefix: str | None = None) -> list[dict]:
    with connect() as conn:
        if date_prefix:
            rows = conn.execute(
                """
                SELECT * FROM schedules
                WHERE user_id=? AND start_time LIKE ?
                ORDER BY start_time
                """,
                (user_id, f"{date_prefix}%"),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM schedules WHERE user_id=? ORDER BY start_time",
                (user_id,),
            ).fetchall()
    return [dict(r) for r in rows]


def upsert_course_room(subject: str, location: str):
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO course_rooms(subject, location)
            VALUES (?, ?)
            ON CONFLICT(subject) DO UPDATE SET location=excluded.location
            """,
            (subject.strip(), location.strip()),
        )


def get_course_room(subject: str) -> dict | None:
    with connect() as conn:
        row = conn.execute(
            """
            SELECT * FROM course_rooms
            WHERE LOWER(subject)=LOWER(?)
            """,
            (subject.strip(),),
        ).fetchone()
    return dict(row) if row else None


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


def list_audit_logs(limit: int = 50) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM audit_logs
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    return [dict(r) for r in rows]
