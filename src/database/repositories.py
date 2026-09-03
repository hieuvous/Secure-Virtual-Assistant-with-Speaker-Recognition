from __future__ import annotations

from datetime import datetime, timezone

from src.database.supabase_client import get_supabase_client
from src.speaker.embedding import embedding_to_numpy, embedding_to_pgvector


def _client():
    """Return the only supported runtime persistence client: Supabase."""
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


def create_user(name: str, student_code: str | None = None) -> int:
    row = _one(
        _client().table("users")
        .insert({"name": name, "student_code": student_code or None})
        .select("id")
        .execute()
    )
    if row is None:
        raise RuntimeError("Supabase did not return an id for the created user.")
    return int(row["id"])


def delete_user(user_id: int) -> int:
    """Delete one user and return the number of rows deleted (zero or one).

    PostgreSQL foreign-key cascades remove child data. This never resets or
    renumbers the users identity sequence.
    """
    response = (
        _client().table("users").delete().eq("id", int(user_id)).select("id").execute()
    )
    return len(response.data or [])


def list_users() -> list[dict]:
    return list(_client().table("users").select("*").order("id").execute().data)


def upsert_profile(
    user_id: int,
    embedding,
    num_samples: int,
    model_version: str,
    enrollment_method: str = "mean",
):
    """Store the final normalized 192-D profile vector in Supabase only."""
    normalized = embedding_to_numpy(embedding)
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


def list_profiles() -> list[dict]:
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


def get_profile(user_id: int) -> dict | None:
    row = _one(
        _client().table("speaker_profiles").select("*, users(name)")
        .eq("user_id", int(user_id)).execute()
    )
    if row is None:
        return None
    user = row.pop("users", None) or {}
    row["name"] = user.get("name")
    return _profile_with_embedding(row)


def add_task(user_id: int, title: str, due_date: str | None = None) -> int:
    row = _one(_client().table("tasks").insert({
        "user_id": int(user_id), "title": title, "due_date": due_date,
    }).select("id").execute())
    if row is None:
        raise RuntimeError("Supabase did not return an id for the created task.")
    return int(row["id"])


def get_tasks(user_id: int) -> list[dict]:
    return list(_client().table("tasks").select("*").eq("user_id", int(user_id))
                .eq("status", "pending").order("due_date").execute().data)


def delete_task_by_title(user_id: int, title: str) -> int:
    matches = [
        task for task in get_tasks(user_id)
        if task["title"].lower() == title.strip().lower()
    ]
    for task in matches:
        _client().table("tasks").delete().eq("id", task["id"]).execute()
    return len(matches)


def delete_task_by_id(user_id: int, task_id: int) -> int:
    exists = any(int(task["id"]) == int(task_id) for task in get_tasks(user_id))
    if not exists:
        return 0
    _client().table("tasks").delete().eq("id", int(task_id)).eq(
        "user_id", int(user_id)
    ).execute()
    return 1


def add_private_note(user_id: int, title: str, content: str) -> int:
    row = _one(_client().table("private_notes").insert({
        "user_id": int(user_id), "title": title, "content": content,
    }).select("id").execute())
    if row is None:
        raise RuntimeError("Supabase did not return an id for the created private note.")
    return int(row["id"])


def get_private_notes(user_id: int) -> list[dict]:
    return list(_client().table("private_notes").select("*")
                .eq("user_id", int(user_id)).order("id").execute().data)


def add_schedule(user_id: int, subject: str, start_time: str,
                 end_time: str | None = None, location: str | None = None) -> int:
    row = _one(_client().table("schedules").insert({
        "user_id": int(user_id), "subject": subject, "start_time": start_time,
        "end_time": end_time, "location": location,
    }).select("id").execute())
    if row is None:
        raise RuntimeError("Supabase did not return an id for the created schedule.")
    return int(row["id"])


def get_schedule(user_id: int, date_prefix: str | None = None) -> list[dict]:
    query = _client().table("schedules").select("*").eq("user_id", int(user_id))
    if date_prefix:
        query = query.like("start_time", f"{date_prefix}%")
    return list(query.order("start_time").execute().data)


def upsert_course_room(subject: str, location: str):
    _client().table("course_rooms").upsert(
        {"subject": subject.strip(), "location": location.strip()},
        on_conflict="subject",
    ).execute()


def get_course_room(subject: str) -> dict | None:
    rows = _client().table("course_rooms").select("*").execute().data
    wanted = subject.strip().lower()
    return next((dict(row) for row in rows if row["subject"].lower() == wanted), None)


def add_audit_log(user_id: int | None, intent: str, auth_method: str,
                  similarity_score: float | None, threshold: float | None, result: str):
    _client().table("audit_logs").insert({
        "user_id": user_id, "intent": intent, "auth_method": auth_method,
        "similarity_score": similarity_score, "threshold": threshold, "result": result,
    }).execute()


def list_audit_logs(limit: int = 50) -> list[dict]:
    return list(_client().table("audit_logs").select("*").order("id", desc=True)
                .limit(int(limit)).execute().data)
