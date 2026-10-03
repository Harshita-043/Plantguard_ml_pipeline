import numpy as np
from plantguard_ml.models.severity import make_features

def test_features():
 f=make_features(np.ones((10,10),bool),np.eye(10,dtype=np.uint8),np.array([.8]),1,.9)
 assert f.shape==(7,)
 assert 0 < f[0] < 100
