"""Thin wrapper around sklearn's IsolationForest -- the ML baseline the brief
calls for (simpler and more defensible than an LSTM autoencoder for imbalanced/
rare anomaly data, which is a stretch goal only if time remains later).
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from absproj.config import IsolationForestConfig


class IsolationForestDetector:
    def __init__(self, config: IsolationForestConfig):
        self.config = config
        self.model = IsolationForest(
            n_estimators=config.n_estimators,
            contamination=config.contamination,
            random_state=config.random_state,
        )

    def fit(self, X: np.ndarray) -> None:
        self.model.fit(X)

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Higher = more anomalous (flipped from sklearn's decision_function,
        where higher means more normal, to match the NIS convention used
        elsewhere in this project)."""
        return -self.model.decision_function(X)

    def predict_is_anomaly(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X) == -1

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: Path) -> None:
        self.model = joblib.load(path)
