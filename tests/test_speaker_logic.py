import numpy as np
import pytest
from src.speaker.profile import aggregate_embeddings
from src.speaker.verification import _reference_to_numpy, verify_embedding
from src.speaker import identification
from src.speaker.identification import identify_embedding

def test_profile_norm():
    p=aggregate_embeddings([np.array([1.,0.]),np.array([.8,.2])])
    assert abs(np.linalg.norm(p)-1)<1e-6

def test_sv_logic():
    assert verify_embedding(np.array([1.,0.]),np.array([1.,0.]),.8)["accepted"]
    assert not verify_embedding(np.array([0.,1.]),np.array([1.,0.]),.8)["accepted"]

def test_sid_logic():
    ps=[{"user_id":1,"name":"A","embedding":np.array([1.,0.])},
        {"user_id":2,"name":"B","embedding":np.array([0.,1.])}]
    assert identify_embedding(np.array([.9,.1]),ps,.8)["user_id"]==1
    assert identify_embedding(np.array([-1.,-1.]),ps,.5)["is_unknown"]


def test_sid_runtime_requires_calibrated_sid_threshold(monkeypatch):
    monkeypatch.setattr(identification, "load_thresholds", lambda: {"sid_threshold": None})
    with pytest.raises(RuntimeError, match="SID threshold has not been calibrated"):
        identification.identify_speaker("unused.wav", [])


def test_sid_runtime_does_not_fallback_to_local_embedding_path(monkeypatch):
    monkeypatch.setattr(identification, "extract_embedding", lambda *_args, **_kwargs: np.array([1.0, 0.0]))
    result = identification.identify_speaker(
        "unused.wav",
        [{"user_id": 1, "name": "Legacy", "embedding_path": "data/users/1/speaker_embedding.npy"}],
        threshold=0.0,
    )
    assert result["is_unknown"]


def test_sv_runtime_rejects_local_embedding_path():
    with pytest.raises(RuntimeError, match="Local speaker embedding paths are no longer supported"):
        _reference_to_numpy("data/users/1/speaker_embedding.npy")
