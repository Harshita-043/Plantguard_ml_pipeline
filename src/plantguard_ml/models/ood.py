
from __future__ import annotations

from pathlib import Path

import numpy as np


class MahalanobisOOD:
    def __init__(self, means, inv_cov, threshold):
        self.means = np.asarray(means, dtype=np.float64)
        self.inv_cov = np.asarray(inv_cov, dtype=np.float64)
        self.threshold = float(np.asarray(threshold).item())

        if self.means.ndim != 2:
            raise ValueError(
                f"Expected class means with shape (classes, features); "
                f"received {self.means.shape}"
            )

        if self.inv_cov.shape != (
            self.means.shape[1],
            self.means.shape[1],
        ):
            raise ValueError(
                "Inverse covariance dimensions do not match class means."
            )

    def score(self, x, class_index: int | None = None):
        x = np.asarray(x, dtype=np.float64).reshape(-1)

        if x.shape[0] != self.means.shape[1]:
            raise ValueError(
                f"Expected {self.means.shape[1]} features, "
                f"received {x.shape[0]}"
            )

        if class_index is None:
            # Backward-compatible behavior: nearest class mean.
            distances = self._distances(x)
            predicted_mean_index = int(np.argmin(distances))
            distance = distances[predicted_mean_index]
        else:
            if not 0 <= class_index < len(self.means):
                raise ValueError(
                    f"class_index {class_index} is outside the valid range."
                )

            distance = self._distance(x, self.means[class_index])
            predicted_mean_index = class_index

        return {
            "distance": float(distance),
            "ood": bool(distance > self.threshold),
            "threshold": self.threshold,
            "class_mean_index": predicted_mean_index,
        }

    def _distance(self, x, mean):
        delta = x - mean
        distance = float(delta @ self.inv_cov @ delta.T)

        # Small numerical errors can occasionally produce a negative value.
        return max(0.0, distance)

    def _distances(self, x):
        return [
            self._distance(x, mean)
            for mean in self.means
        ]

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            path,
            means=self.means,
            inv_cov=self.inv_cov,
            threshold=self.threshold,
        )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(
                means=data["means"],
                inv_cov=data["inv_cov"],
                threshold=data["threshold"],
            )
