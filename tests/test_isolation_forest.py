import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from absproj.config import IsolationForestConfig  # noqa: E402
from absproj.ml.isolation_forest import IsolationForestDetector  # noqa: E402


def _synthetic_dataset(rng):
    normal = rng.normal(loc=0.0, scale=1.0, size=(300, 4))
    outliers = rng.normal(loc=25.0, scale=1.0, size=(15, 4))
    X = np.vstack([normal, outliers])
    is_outlier = np.array([False] * 300 + [True] * 15)
    return X, is_outlier


def test_isolation_forest_flags_synthetic_outliers():
    rng = np.random.default_rng(0)
    X, is_outlier = _synthetic_dataset(rng)

    config = IsolationForestConfig(n_estimators=100, contamination=15 / 315, random_state=0)
    detector = IsolationForestDetector(config)
    detector.fit(X)

    predicted = detector.predict_is_anomaly(X)
    # Nearly all injected outliers should be caught; a small miss margin is fine.
    recall = predicted[is_outlier].mean()
    assert recall > 0.8

    # False positive rate among the normal cluster should be low.
    fpr = predicted[~is_outlier].mean()
    assert fpr < 0.1


def test_isolation_forest_scores_outliers_higher_than_normal():
    rng = np.random.default_rng(1)
    X, is_outlier = _synthetic_dataset(rng)

    config = IsolationForestConfig(n_estimators=100, contamination=15 / 315, random_state=0)
    detector = IsolationForestDetector(config)
    detector.fit(X)

    scores = detector.anomaly_score(X)
    assert scores[is_outlier].mean() > scores[~is_outlier].mean()


def test_isolation_forest_save_and_load_roundtrip(tmp_path):
    rng = np.random.default_rng(2)
    X, _ = _synthetic_dataset(rng)

    config = IsolationForestConfig(n_estimators=50, contamination=0.05, random_state=0)
    detector = IsolationForestDetector(config)
    detector.fit(X)
    scores_before = detector.anomaly_score(X)

    path = tmp_path / "model.joblib"
    detector.save(path)

    reloaded = IsolationForestDetector(config)
    reloaded.load(path)
    scores_after = reloaded.anomaly_score(X)

    assert np.allclose(scores_before, scores_after)
