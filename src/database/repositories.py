from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.config import ROOT
from src.database.db import connect, database_backend
from src.database.supabase_client import get_supabase_client
from src.speaker.embedding import embedding_to_numpy, embedding_to_pgvector


def _is_supabase() -> bool:
    return database_backend() == "supabase"


def _client():
    return get_supabase_client()


def _one(response) -> dict | None:
    data = response.data
    if not data:
        return None
    return dict(data[0]) if isinstance(data, list) else dict(data)


def _profile_with_embedding(profile: dict) -> dict:
    if profile.get("embedding") is not None:
        profile["embedding"] = embedding_to_numpy(profile["embedding"])
    return profile


def _sqlite_embedding_path(user_id: int, embedding) -> str:
    relative_path = f"data/users/{int(user_id)}/speaker_embedding.npy"
    destination = ROOT / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, embedding_to_numpy(embedding))
    return relative_path


def _sqlite_profile_with_embedding(profile: dict) -> dict:
    path = Path(profile["embedding_path"])
    if not path.is_absolute():
        path = ROOT / path
    if path.exists():
        profile["embedding"] = embedding_to_numpy(np.load(path))
    return profile


def create_user(name: str, student_code: str | None = None) -> int:
    if _is_supabase():
        row = _one(
            _client().table("users")
            .insert({"name": name, "student_code": student_code or None})
            .select("id")
            .execute()
        )
        return int(row["id"])

    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO users(name, student_code) VALUES (?, ?)",
            (name, student_code or None),
        )
        return int(cur.lastrowid)


def list_users() -> list[dict]:
    if _is_supabase():
        return list(_client().table("users").select("*").order("id").execute().data)
    with connect() as conn:
        rows = conn.execute("SELECT * FROM users ORDER BY id").fetchall()
    return [dict(r) for r in rows]


def upsert_profile(
    user_id: int,
    embedding,
    num_samples: int,
    model_version: str,
    enrollment_method: str = "mean",
):
    """Store a 192-D embedding in Supabase or in SQLite's local fallback."""
    normalized = embedding_to_numpy(embedding)
    if _is_supabase():
        _client().table("speaker_profiles").upsert(
            {
                "user_id": int(user_id),
                "embedding": embedding_to_pgvector(normalized),
                "num_samples": int(num_samples),
                "model_version": model_version,
                "enrollment_method": enrollment_method,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            on_conflict="user_id",
        ).execute()
        return

    embedding_path = _sqlite_embedding_path(user_id, normalized)
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO speaker_profiles(
                user_id, embedding_path, num_samples, model_version, enrollment_method
            ) VALUES (?, ?, ?, ?, ?)
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
    if _is_supabase():
        rows = _client().table("speaker_profiles").select(
            "user_id, embedding, num_samples, model_version, enrollment_method, users(name)"
        ).order("user_id").execute().data
        profiles = []
        for row in rows:
            profile = dict(row)
            user = profile.pop("users", None) or {}
            profile["name"] = user.get("name")
            profiles.append(_profile_with_embedding(profile))
        return profiles

    with connect() as conn:
        rows = conn.execute(
            """
            SELECT p.user_id, u.name, p.embedding_path, p.num_samples,
                   p.model_version, p.enrollment_method
            FROM speaker_profiles p JOIN users u ON u.id = p.user_id
            ORDER BY p.user_id
            """
        ).fetchall()
    return [_sqlite_profile_with_embedding(dict(r)) for r in rows]


def get_profile(user_id: int) -> dict | None:
    if _is_supabase():
        row = _one(
            _client().table("speaker_profiles").select("*, users(name)")
            .eq("user_id", int(user_id)).execute()
        )
        if row is None:
            return None
        user = row.pop("users", None) or {}
        row["name"] = user.get("name")
        return _profile_with_embedding(row)

    with connect() as conn:
        row = conn.execute(
            """SELECT p.*, u.name FROM speaker_profiles p JOIN users u ON u.id = p.user_id
               WHERE p.user_id=?""",
            (user_id,),
        ).fetchone()
    return _sqlite_profile_with_embedding(dict(row)) if row else None


def add_task(user_id: int, title: str, due_date: str | None = None) -> int:
    if _is_supabase():
        row = _one(_client().table("tasks").insert({
            "user_id": int(user_id), "title": title, "due_date": due_date,
        }).select("id").execute())
        return int(row["id"])
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO tasks(user_id, title, due_date) VALUES (?, ?, ?)",
            (user_id, title, due_date),
        )
        return int(cur.lastrowid)


def get_tasks(user_id: int) -> list[dict]:
    if _is_supabase():
        return list(_client().table("tasks").select("*").eq("user_id", int(user_id))
                    .eq("status", "pending").order("due_date").execute().data)
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM tasks WHERE user_id=? AND status='pending' ORDER BY due_date",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_task_by_title(user_id: int, title: str) -> int:
    if _is_supabase():
        matches = [
            task for task in get_tasks(user_id)
            if task["title"].lower() == title.strip().lower()
        ]
        for task in matches:
            _client().table("tasks").delete().eq("id", task["id"]).execute()
        return len(matches)
    with connect() as conn:
        cur = conn.execute(
            "DELETE FROM tasks WHERE user_id=? AND LOWER(title)=LOWER(?)",
            (user_id, title.strip()),
        )
        return int(cur.rowcount)


def add_private_note(user_id: int, title: str, content: str) -> int:
    if _is_supabase():
        row = _one(_client().table("private_notes").insert({
            "user_id": int(user_id), "title": title, "content": content,
        }).select("id").execute())
        return int(row["id"])
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO private_notes(user_id, title, content) VALUES (?, ?, ?)",
            (user_id, title, content),
        )
        return int(cur.lastrowid)


def get_private_notes(user_id: int) -> list[dict]:
    if _is_supabase():
        return list(_client().table("private_notes").select("*")
                    .eq("user_id", int(user_id)).order("id").execute().data)
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM private_notes WHERE user_id=? ORDER BY id", (user_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def add_schedule(user_id: int, subject: str, start_time: str,
                 end_time: str | None = None, location: str | None = None) -> int:
    if _is_supabase():
        row = _one(_client().table("schedules").insert({
            "user_id": int(user_id), "subject": subject, "start_time": start_time,
            "end_time": end_time, "location": location,
        }).select("id").execute())
        return int(row["id"])
    with connect() as conn:
        cur = conn.execute(
            """INSERT INTO schedules(user_id, subject, start_time, end_time, location)
               VALUES (?, ?, ?, ?, ?)""",
            (user_id, subject, start_time, end_time, location),
        )
        return int(cur.lastrowid)


def get_schedule(user_id: int, date_prefix: str | None = None) -> list[dict]:
    if _is_supabase():
        query = _client().table("schedules").select("*").eq("user_id", int(user_id))
        if date_prefix:
            query = query.like("start_time", f"{date_prefix}%")
        return list(query.order("start_time").execute().data)
    with connect() as conn:
        if date_prefix:
            rows = conn.execute(
                """SELECT * FROM schedules WHERE user_id=? AND start_time LIKE ?
                   ORDER BY start_time""",
                (user_id, f"{date_prefix}%"),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM schedules WHERE user_id=? ORDER BY start_time", (user_id,)
            ).fetchall()
    return [dict(r) for r in rows]


def upsert_course_room(subject: str, location: str):
    if _is_supabase():
        _client().table("course_rooms").upsert(
            {"subject": subject.strip(), "location": location.strip()},
            on_conflict="subject",
        ).execute()
        return
    with connect() as conn:
        conn.execute(
            """INSERT INTO course_rooms(subject, location) VALUES (?, ?)
               ON CONFLICT(subject) DO UPDATE SET location=excluded.location""",
            (subject.strip(), location.strip()),
        )


def get_course_room(subject: str) -> dict | None:
    if _is_supabase():
        rows = _client().table("course_rooms").select("*").execute().data
        wanted = subject.strip().lower()
        return next((dict(row) for row in rows if row["subject"].lower() == wanted), None)
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM course_rooms WHERE LOWER(subject)=LOWER(?)", (subject.strip(),)
        ).fetchone()
    return dict(row) if row else None


def add_audit_log(user_id: int | None, intent: str, auth_method: str,
                  similarity_score: float | None, threshold: float | None, result: str):
    payload = {
        "user_id": user_id, "intent": intent, "auth_method": auth_method,
        "similarity_score": similarity_score, "threshold": threshold, "result": result,
    }
    if _is_supabase():
        _client().table("audit_logs").insert(payload).execute()
        return
    with connect() as conn:
        conn.execute(
            """INSERT INTO audit_logs(user_id, intent, auth_method, similarity_score,
               threshold, result) VALUES (?, ?, ?, ?, ?, ?)""",
            (user_id, intent, auth_method, similarity_score, threshold, result),
        )


def list_audit_logs(limit: int = 50) -> list[dict]:
    if _is_supabase():
        return list(_client().table("audit_logs").select("*").order("id", desc=True)
                    .limit(int(limit)).execute().data)
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM audit_logs ORDER BY id DESC LIMIT ?", (int(limit),)
        ).fetchall()
    return [dict(r) for r in rows]
