import numpy as np
import pytest
from pathlib import Path

from src import config
from src.database import db as db_module
from src.database import repositories
from src.database import supabase_client
from src.database.db import database_backend, init_db
from src.speaker.embedding import embedding_to_numpy, embedding_to_pgvector


def _isolate_database_environment(monkeypatch, tmp_path):
    """Keep tests independent of a developer's real, ignored project .env."""
    for name in ("DATABASE_BACKEND", "SUPABASE_URL", "SUPABASE_SECRET_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "ENV_PATH", tmp_path / "missing.env")


def test_default_env_path_is_resolved_from_repository_root():
    assert config.ROOT == Path(__file__).resolve().parents[1]
    assert config.ENV_PATH == config.ROOT / ".env"


def test_missing_database_backend_fails_fast(monkeypatch, tmp_path):
    _isolate_database_environment(monkeypatch, tmp_path)
    with pytest.raises(config.ConfigurationError, match="DATABASE_BACKEND is required"):
        database_backend()


def test_database_backend_explicit_sqlite(monkeypatch, tmp_path):
    _isolate_database_environment(monkeypatch, tmp_path)
    monkeypatch.setenv("DATABASE_BACKEND", "sqlite")
    assert database_backend() == "sqlite"


def test_explicit_process_environment_overrides_dotenv(monkeypatch, tmp_path):
    _isolate_database_environment(monkeypatch, tmp_path)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "DATABASE_BACKEND=supabase\nSUPABASE_URL=https://dotenv.supabase.co\n"
        "SUPABASE_SECRET_KEY=dotenv-secret\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "ENV_PATH", env_file)
    monkeypatch.setenv("DATABASE_BACKEND", "sqlite")

    assert database_backend() == "sqlite"


def test_supabase_config_is_validated_before_client_creation(monkeypatch, tmp_path):
    _isolate_database_environment(monkeypatch, tmp_path)
    monkeypatch.setenv("DATABASE_BACKEND", "supabase")
    monkeypatch.setenv("SUPABASE_URL", "https://abcdefgh.supabase.co")
    monkeypatch.setenv("SUPABASE_SECRET_KEY", "test-secret")
    created = []
    monkeypatch.setattr(
        supabase_client, "_create_supabase_client", lambda url, key: created.append((url, key)) or object()
    )

    assert supabase_client.get_supabase_client() is not None
    assert created == [("https://abcdefgh.supabase.co", "test-secret")]
    assert database_backend() == "supabase"


@pytest.mark.parametrize(
    ("missing_name", "expected_message"),
    [("SUPABASE_URL", "SUPABASE_URL"), ("SUPABASE_SECRET_KEY", "SUPABASE_SECRET_KEY")],
)
def test_supabase_missing_credentials_fails_fast(monkeypatch, tmp_path, missing_name, expected_message):
    _isolate_database_environment(monkeypatch, tmp_path)
    monkeypatch.setenv("DATABASE_BACKEND", "supabase")
    monkeypatch.setenv("SUPABASE_URL", "https://abcdefgh.supabase.co")
    monkeypatch.setenv("SUPABASE_SECRET_KEY", "test-secret")
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


def test_sqlite_profile_fallback_returns_embedding(tmp_path, monkeypatch):
    _isolate_database_environment(monkeypatch, tmp_path)
    monkeypatch.setenv("DATABASE_BACKEND", "sqlite")
    monkeypatch.setattr(db_module, "db_path", lambda: tmp_path / "test.db")
    monkeypatch.setattr(repositories, "ROOT", tmp_path)
    init_db()
    user_id = repositories.create_user("Embedding test")
    vector = np.ones(192, dtype=np.float32)
    repositories.upsert_profile(user_id, vector, 5, "test-model")
    profile = repositories.get_profile(user_id)
    assert profile is not None
    np.testing.assert_allclose(profile["embedding"], vector)


class _FakeResponse:
    def __init__(self, data=None):
        self.data = [] if data is None else data


_FAKE_ROWS_BY_TABLE = {
    "users": [{"id": 1, "name": "Test", "student_code": None}],
    "tasks": [{
        "id": 1, "user_id": 1, "title": "Task", "due_date": None,
        "status": "pending",
    }],
    "schedules": [{
        "id": 1, "user_id": 1, "subject": "Class",
        "start_time": "2026-01-01 09:00", "end_time": None, "location": None,
    }],
    "private_notes": [{"id": 1, "user_id": 1, "title": "Title", "content": "Content"}],
    "course_rooms": [{"id": 1, "subject": "ML", "location": "A1"}],
    "audit_logs": [{
        "id": 1, "user_id": 1, "intent": "GET_TASKS", "auth_method": "GENERAL",
        "similarity_score": None, "threshold": None, "result": "ALLOWED",
    }],
    # The mock does not persist the preceding upsert, so profile reads model a
    # valid empty result rather than an invented malformed speaker profile.
    "speaker_profiles": [],
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


def test_repository_operations_share_the_central_supabase_backend(monkeypatch, tmp_path):
    _isolate_database_environment(monkeypatch, tmp_path)
    monkeypatch.setenv("DATABASE_BACKEND", "supabase")
    monkeypatch.setenv("SUPABASE_URL", "https://abcdefgh.supabase.co")
    monkeypatch.setenv("SUPABASE_SECRET_KEY", "test-secret")
    client = _FakeClient()

    def real_client_must_not_be_created(*_args, **_kwargs):
        raise AssertionError("Real Supabase client must not be used in unit tests")

    # repositories._client() resolves this module-local imported symbol.
    monkeypatch.setattr(repositories, "get_supabase_client", lambda: client)
    monkeypatch.setattr(supabase_client, "_create_supabase_client", real_client_must_not_be_created)
    vector = np.ones(192, dtype=np.float32)

    repositories.create_user("Test")
    repositories.list_users()
    repositories.upsert_profile(1, vector, 5, "test")
    repositories.list_profiles()
    repositories.get_profile(1)
    repositories.add_task(1, "Task")
    repositories.get_tasks(1)
    repositories.delete_task_by_title(1, "Task")
    repositories.add_schedule(1, "Class", "2026-01-01 09:00")
    repositories.get_schedule(1)
    repositories.add_private_note(1, "Title", "Content")
    repositories.get_private_notes(1)
    repositories.upsert_course_room("ML", "A1")
    repositories.get_course_room("ML")
    repositories.add_audit_log(1, "GET_TASKS", "GENERAL", None, None, "ALLOWED")
    repositories.list_audit_logs()

    assert database_backend() == "supabase"
    assert set(client.tables) == {
        "users", "speaker_profiles", "tasks", "schedules", "private_notes",
        "course_rooms", "audit_logs",
    }
