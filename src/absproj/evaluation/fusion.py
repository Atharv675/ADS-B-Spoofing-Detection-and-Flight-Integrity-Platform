"""Evidence-fusion score: combines NIS, the ML anomaly score, MLAT/radar
agreement, and identity consistency into a single 0-100 integrity score per
update. This is a scoring function living in the evaluation code, not a
standalone service, per the brief.

The five inputs are on wildly different natural scales (NIS is unbounded and
right-skewed, MLAT/radar disagreement is in meters, the ML score is an
sklearn decision-function value with no physical unit, identity is a bare
0/1) and are not equally informative for every attack class (MLAT/radar
carry zero signal for ICAO collision, by design -- see
attacks/icao_collision.py -- which is exactly why identity consistency was
added: it's the one input that does carry signal there). Rather than
hand-pick weights, fusion is a logistic regression (with standardization)
*fit on the train split only* -- this is the concrete place the brief's
"tune on train, evaluate on test" methodology happens, since the individual
detectors' thresholds were already calibrated against clean traffic in
Phases 2-5 and aren't retuned here.

Where MLAT/radar report no_corroboration (no physical target at all), their
disagreement is capped at a large finite value rather than passed through as
infinity, so it can be standardized like everything else.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NO_CORROBORATION_CAP_M = 50_000.0  # stands in for "infinite" disagreement, kept finite for scaling


@dataclass
class FusionRow:
    nis: float
    ml_score: float
    mlat_disagreement_m: float
    mlat_no_corroboration: bool
    radar_disagreement_m: float
    radar_no_corroboration: bool
    identity_mismatch: bool = False


def _feature_vector(row: FusionRow) -> list[float]:
    mlat = NO_CORROBORATION_CAP_M if row.mlat_no_corroboration else row.mlat_disagreement_m
    radar = NO_CORROBORATION_CAP_M if row.radar_no_corroboration else row.radar_disagreement_m
    return [row.nis, row.ml_score, mlat, radar, 1.0 if row.identity_mismatch else 0.0]


class EvidenceFusion:
    def __init__(self, random_state: int = 42, decision_threshold: float = 0.5):
        self.pipeline = Pipeline([
            ("scale", StandardScaler()),
            ("logreg", LogisticRegression(random_state=random_state, class_weight="balanced")),
        ])
        self.decision_threshold = decision_threshold
        self._fitted = False

    def fit(self, rows: list[FusionRow], labels: list[bool]) -> None:
        X = np.array([_feature_vector(r) for r in rows])
        y = np.array(labels, dtype=int)
        self.pipeline.fit(X, y)
        self._fitted = True

    def suspicion(self, rows: list[FusionRow]) -> np.ndarray:
        """P(attacked), one per row, in [0, 1]."""
        if not self._fitted:
            raise RuntimeError("EvidenceFusion.fit() must be called before scoring")
        X = np.array([_feature_vector(r) for r in rows])
        return self.pipeline.predict_proba(X)[:, 1]

    def integrity_score(self, rows: list[FusionRow]) -> np.ndarray:
        """0-100, 100 = fully trustworthy, 0 = maximally suspicious."""
        return 100.0 * (1.0 - self.suspicion(rows))

    def is_anomalous(self, rows: list[FusionRow]) -> np.ndarray:
        return self.suspicion(rows) > self.decision_threshold
