import numpy as np
import pytest

from src.database import db as db_module
from src.database import repositories
from src.database.db import database_backend, init_db
from src.speaker.embedding import embedding_to_numpy, embedding_to_pgvector


def test_database_backend_defaults_to_sqlite(monkeypatch):
    monkeypatch.delenv("DATABASE_BACKEND", raising=False)
    assert database_backend() == "sqlite"


def test_database_backend_accepts_supabase_and_rejects_unknown(monkeypatch):
    monkeypatch.setenv("DATABASE_BACKEND", "supabase")
    assert database_backend() == "supabase"

    monkeypatch.setenv("DATABASE_BACKEND", "unsupported")
    with pytest.raises(RuntimeError, match="DATABASE_BACKEND"):
        database_backend()


def test_pgvector_embedding_conversion_round_trip():
    values = np.linspace(-1.0, 1.0, 192, dtype=np.float32)
    stored = embedding_to_pgvector(values)
    restored = embedding_to_numpy(stored)

    assert stored.startswith("[") and stored.endswith("]")
    assert restored.shape == (192,)
    np.testing.assert_allclose(restored, values, rtol=0, atol=1e-7)


def test_embedding_conversion_rejects_wrong_dimension():
    with pytest.raises(ValueError, match="192-D"):
        embedding_to_numpy([0.0, 1.0])


def test_sqlite_profile_fallback_returns_embedding(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_BACKEND", raising=False)
    monkeypatch.setattr(db_module, "db_path", lambda: tmp_path / "test.db")
    monkeypatch.setattr(repositories, "ROOT", tmp_path)
    init_db()

    user_id = repositories.create_user("Embedding test")
    vector = np.ones(192, dtype=np.float32)
    repositories.upsert_profile(user_id, vector, 5, "test-model")

    profile = repositories.get_profile(user_id)
    assert profile is not None
    np.testing.assert_allclose(profile["embedding"], vector)
