import numpy as np
from src.speaker.profile import aggregate_embeddings
from src.speaker.verification import verify_embedding
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
