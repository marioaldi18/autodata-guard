"""Unsupervised anomaly detectors sharing a fit/score interface (higher score = more anomalous)."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


class RobustZScoreDetector:
    """Baseline: maximum absolute modified z-score (median/MAD) across features."""

    name = "robust_zscore_baseline"

    def fit(self, X: np.ndarray) -> "RobustZScoreDetector":
        X = np.asarray(X, dtype=float)
        self.median_ = np.median(X, axis=0)
        mad = np.median(np.abs(X - self.median_), axis=0)
        self.mad_ = np.where(mad < 1e-9, 1.0, mad)  # constant column: avoid division by zero
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        z = 0.6745 * (np.asarray(X, dtype=float) - self.median_) / self.mad_
        return np.abs(z).max(axis=1)


class IsolationForestDetector:
    name = "isolation_forest"

    def __init__(self, seed: int, n_estimators: int = 200):
        self.model = IsolationForest(n_estimators=n_estimators, random_state=seed, contamination="auto")

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        self.model.fit(np.asarray(X, dtype=float))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self.model.score_samples(np.asarray(X, dtype=float))


class LOFDetector:
    name = "local_outlier_factor"

    def __init__(self, n_neighbors: int = 35):
        self.model = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)

    def fit(self, X: np.ndarray) -> "LOFDetector":
        self.model.fit(np.asarray(X, dtype=float))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self.model.score_samples(np.asarray(X, dtype=float))


def build_detectors(cfg: Dict[str, Any]) -> List[Any]:
    seed = cfg["seed"]
    return [
        RobustZScoreDetector(),
        IsolationForestDetector(seed, **cfg["models"]["isolation_forest"]),
        LOFDetector(**cfg["models"]["lof"]),
    ]
