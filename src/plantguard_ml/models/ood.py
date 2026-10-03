from __future__ import annotations
import json
from pathlib import Path
import numpy as np

class MahalanobisOOD:
    def __init__(self, means, inv_cov, threshold):
        self.means = np.asarray(means); self.inv_cov=np.asarray(inv_cov); self.threshold=float(threshold)
    def score(self, x):
        x=np.asarray(x)
        d=[]
        for mu in self.means:
            z=x-mu; d.append(float(z @ self.inv_cov @ z.T))
        best=min(d); return {'distance':best,'ood':best>self.threshold,'threshold':self.threshold}
    def save(self,path):
        np.savez(path, means=self.means, inv_cov=self.inv_cov, threshold=self.threshold)
    @classmethod
    def load(cls,path):
        z=np.load(path); return cls(z['means'],z['inv_cov'],z['threshold'])
