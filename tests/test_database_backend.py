from pathlib import Path

import numpy as np
import pytest

from src import config
from src.database import repositories, supabase_client
from src.database.db import database_backend, init_db
from src.speaker.embedding import embedding_to_numpy, embedding_to_pgvector


def _isolate_database_environment(monkeypatch, tmp_path):
    """Keep tests independent of a developer's real, ignored project .env."""
    for name in ("DATABASE_BACKEND", "SUPABASE_URL", "SUPABASE_SECRET_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "ENV_PATH", tmp_path / "missing.env")


def _set_fake_supabase_environment(monkeypatch, tmp_path):
    _isolate_database_environment(monkeypatch, tmp_path)
    monkeypatch.setenv("DATABASE_BACKEND", "supabase")
    monkeypatch.setenv("SUPABASE_URL", "https://abcdefgh.supabase.co")
    monkeypatch.setenv("SUPABASE_SECRET_KEY", "test-secret")


def test_default_env_path_is_resolved_from_repository_root():
    assert config.ROOT == Path(__file__).resolve().parents[1]
    assert config.ENV_PATH == config.ROOT / ".env"


def test_missing_database_backend_fails_fast(monkeypatch, tmp_path):
    _isolate_database_environment(monkeypatch, tmp_path)
    with pytest.raises(config.ConfigurationError, match="DATABASE_BACKEND is required"):
        database_backend()


def test_sqlite_runtime_backend_is_rejected(monkeypatch, tmp_path):
    _isolate_database_environment(monkeypatch, tmp_path)
    monkeypatch.setenv("DATABASE_BACKEND", "sqlite")
    with pytest.raises(config.ConfigurationError, match="SQLite runtime backend has been removed"):
        database_backend()


def test_explicit_process_environment_overrides_dotenv(monkeypatch, tmp_path):
    _isolate_database_environment(monkeypatch, tmp_path)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "DATABASE_BACKEND=supabase\nSUPABASE_URL=https://dotenv.supabase.co\n"
        "SUPABASE_SECRET_KEY=dotenv-secret\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "ENV_PATH", env_file)
    monkeypatch.setenv("DATABASE_BACKEND", "supabase")
    monkeypatch.setenv("SUPABASE_URL", "https://process.supabase.co")
    monkeypatch.setenv("SUPABASE_SECRET_KEY", "process-secret")

    effective = config.get_database_config()
    assert effective.backend == "supabase"
    assert effective.supabase_url == "https://process.supabase.co"
    assert effective.supabase_secret_key == "process-secret"


def test_supabase_config_is_validated_before_client_creation(monkeypatch, tmp_path):
    _set_fake_supabase_environment(monkeypatch, tmp_path)
    created = []
    monkeypatch.setattr(
        supabase_client, "_create_supabase_client",
        lambda url, key: created.append((url, key)) or object(),
    )

    assert supabase_client.get_supabase_client() is not None
    assert created == [("https://abcdefgh.supabase.co", "test-secret")]
    assert database_backend() == "supabase"


@pytest.mark.parametrize(
    ("missing_name", "expected_message"),
    [("SUPABASE_URL", "SUPABASE_URL"), ("SUPABASE_SECRET_KEY", "SUPABASE_SECRET_KEY")],
)
def test_supabase_missing_credentials_fails_fast(monkeypatch, tmp_path, missing_name, expected_message):
    _set_fake_supabase_environment(monkeypatch, tmp_path)
    monkeypatch.delenv(missing_name)
    with pytest.raises(config.ConfigurationError, match=expected_message):
        database_backend()


def test_env_path_is_not_affected_by_current_working_directory(monkeypatch, tmp_path):
    _isolate_database_environment(monkeypatch, tmp_path)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "DATABASE_BACKEND=supabase\nSUPABASE_URL=https://projectref.supabase.co\n"
        "SUPABASE_SECRET_KEY=test-secret\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "ENV_PATH", env_file)
    other_directory = tmp_path / "other-working-directory"
    other_directory.mkdir()
    monkeypatch.chdir(other_directory)

    assert database_backend() == "supabase"
    assert config.supabase_project_ref(config.get_database_config().supabase_url) == "projectref"


def test_pgvector_embedding_conversion_round_trip():
    values = np.linspace(-1.0, 1.0, 192, dtype=np.float32)
    stored = embedding_to_pgvector(values)
    restored = embedding_to_numpy(stored)
    assert stored.startswith("[") and stored.endswith("]")
    np.testing.assert_allclose(restored, values, rtol=0, atol=1e-7)


def test_embedding_conversion_rejects_wrong_dimension():
    with pytest.raises(ValueError, match="192-D"):
        embedding_to_numpy([0.0, 1.0])


class _FakeResponse:
    def __init__(self, data=None):
        self.data = [] if data is None else data


_FAKE_ROWS_BY_TABLE = {
    "users": [{"id": 1, "name": "Test", "student_code": None}],
    "speaker_profiles": [{
        "user_id": 1, "embedding": [1.0] + [0.0] * 191, "num_samples": 5,
        "model_version": "test", "enrollment_method": "mean", "users": {"name": "Test"},
    }],
    "tasks": [{"id": 1, "user_id": 1, "title": "Task", "due_date": None, "status": "pending"}],
    "schedules": [{
        "id": 1, "user_id": 1, "subject": "Class", "start_time": "2026-01-01 09:00",
        "end_time": None, "location": None,
    }],
    "private_notes": [{"id": 1, "user_id": 1, "title": "Title", "content": "Content"}],
    "course_rooms": [{"id": 1, "subject": "ML", "location": "A1"}],
    "audit_logs": [{
        "id": 1, "user_id": 1, "intent": "GET_TASKS", "auth_method": "GENERAL",
        "similarity_score": None, "threshold": None, "result": "ALLOWED",
    }],
}


class _FakeQuery:
    def __init__(self, table):
        self.table_name = table

    def __getattr__(self, _name):
        return lambda *args, **kwargs: self

    def execute(self):
        return _FakeResponse(_FAKE_ROWS_BY_TABLE[self.table_name])


class _FakeClient:
    def __init__(self):
        self.tables = []

    def table(self, table):
        self.tables.append(table)
        return _FakeQuery(table)


def test_repository_operations_use_fake_supabase_only(monkeypatch, tmp_path):
    _set_fake_supabase_environment(monkeypatch, tmp_path)
    client = _FakeClient()

    def real_client_must_not_be_created(*_args, **_kwargs):
        raise AssertionError("Real Supabase client must not be used in unit tests")

    monkeypatch.setattr(repositories, "get_supabase_client", lambda: client)
    monkeypatch.setattr(supabase_client, "_create_supabase_client", real_client_must_not_be_created)
    vector = np.ones(192, dtype=np.float32)

    assert repositories.create_user("Test") == 1
    repositories.list_users()
    repositories.upsert_profile(1, vector, 5, "test")
    assert repositories.get_profile(1)["embedding"].shape == (192,)
    assert repositories.list_profiles()[0]["embedding"].shape == (192,)
    repositories.add_task(1, "Task")
    repositories.get_tasks(1)
    repositories.delete_task_by_title(1, "Task")
    assert repositories.delete_task_by_id(1, 1) == 1
    repositories.add_schedule(1, "Class", "2026-01-01 09:00")
    repositories.get_schedule(1)
    repositories.add_private_note(1, "Title", "Content")
    repositories.get_private_notes(1)
    repositories.upsert_course_room("ML", "A1")
    repositories.get_course_room("ML")
    repositories.add_audit_log(1, "GET_TASKS", "GENERAL", None, None, "ALLOWED")
    repositories.list_audit_logs()
    assert repositories.delete_user(1) == 1

    assert database_backend() == "supabase"
    assert set(client.tables) == {
        "users", "speaker_profiles", "tasks", "schedules", "private_notes",
        "course_rooms", "audit_logs",
    }


def test_profile_repository_never_creates_local_npy(monkeypatch, tmp_path):
    _set_fake_supabase_environment(monkeypatch, tmp_path)
    client = _FakeClient()
    monkeypatch.setattr(repositories, "get_supabase_client", lambda: client)
    monkeypatch.setattr(
        "numpy.save",
        lambda *_args, **_kwargs: pytest.fail("Runtime profile persistence must not write .npy files"),
    )

    repositories.upsert_profile(1, np.ones(192, dtype=np.float32), 5, "test")
    assert client.tables == ["speaker_profiles"]


def test_delete_user_does_not_reset_or_renumber_identity(monkeypatch):
    calls = []

    class Query:
        def delete(self):
            calls.append("delete")
            return self

        def eq(self, field, value):
            calls.append(("eq", field, value))
            return self

        def select(self, fields):
            calls.append(("select", fields))
            return self

        def execute(self):
            return _FakeResponse([{"id": 41}])

    class Client:
        def table(self, table):
            assert table == "users"
            return Query()

    monkeypatch.setattr(repositories, "get_supabase_client", lambda: Client())
    assert repositories.delete_user(41) == 1
    assert calls == ["delete", ("eq", "id", 41), ("select", "id")]


def test_init_db_uses_supabase_client(monkeypatch, tmp_path):
    _set_fake_supabase_environment(monkeypatch, tmp_path)
    client = object()
    monkeypatch.setattr(supabase_client, "get_supabase_client", lambda: client)
    init_db()
