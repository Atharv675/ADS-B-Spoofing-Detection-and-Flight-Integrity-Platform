"""Trains the Isolation Forest baseline on Kalman-derived features from all
accumulated (clean, no injected attacks) traffic, scores that same data, writes
results to detections (method='ml'), and compares against the NIS baseline on
the same clean data -- per the brief, this same-data comparison happens now,
before any attacks are introduced in later phases.

This is an in-sample fit/score for this sanity-check baseline; the held-out
train/test split methodology (and the true precision/recall evaluation)
belongs to the adversarial testbed in Phase 6/7, once there are labeled
attacks to evaluate against. Unsupervised Isolation Forest has no labels here
either way -- what we can sanity-check now is calibration (does its flag rate
match its own contamination prior) and agreement/disagreement with NIS.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from absproj.config import get_config  # noqa: E402
from absproj.logging_setup import configure_logging  # noqa: E402
from absproj.ml.features import build_feature_frame, feature_matrix  # noqa: E402
from absproj.ml.isolation_forest import IsolationForestDetector  # noqa: E402
from absproj.storage import repository  # noqa: E402
from absproj.storage.db import get_connection  # noqa: E402

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "isolation_forest.joblib"


def main() -> None:
    config = get_config()
    configure_logging(config.logging.level, config.logging.format)

    with get_connection(config.database) as conn:
        rows = repository.fetch_kalman_updates_for_ml(conn)
        if not rows:
            logger.error("no_kalman_updates_found_run_phase2_first")
            sys.exit(1)

        df = build_feature_frame(rows, config.ml)
        X = feature_matrix(df)

        detector = IsolationForestDetector(config.ml.isolation_forest)
        detector.fit(X)
        detector.save(MODEL_PATH)

        scores = detector.anomaly_score(X)
        is_anomaly = detector.predict_is_anomaly(X)

        df["ml_score"] = scores
        df["ml_is_anomaly"] = is_anomaly

        repository.truncate_ml_detections(conn)
        ml_rows = list(zip(df["time"], df["icao24"], df["ml_score"], df["ml_is_anomaly"]))
        repository.insert_ml_detections(conn, ml_rows)

    n = len(df)
    ml_flagged = int(df["ml_is_anomaly"].sum())
    nis_flagged = int(df["nis_is_anomalous"].sum())
    both_flagged = int((df["ml_is_anomaly"] & df["nis_is_anomalous"]).sum())
    either_flagged = int((df["ml_is_anomaly"] | df["nis_is_anomalous"]).sum())
    jaccard = both_flagged / either_flagged if either_flagged else 0.0
    # Spearman (rank), not Pearson: NIS is extremely right-skewed (a few
    # updates in the hundreds of thousands next to a median under 1), which
    # would make a linear correlation dominated by a handful of points and
    # understate genuine rank agreement between the two detectors.
    rank_correlation = float(spearmanr(df["ml_score"], df["nis"]).statistic)

    logger.info(
        "isolation_forest_trained",
        extra={
            "model_path": str(MODEL_PATH),
            "n_updates": n,
            "n_features": X.shape[1],
            "contamination_config": config.ml.isolation_forest.contamination,
        },
    )
    logger.info(
        "ml_vs_nis_comparison_clean_traffic",
        extra={
            "n": n,
            "ml_flagged": ml_flagged,
            "ml_flag_rate": ml_flagged / n,
            "nis_flagged": nis_flagged,
            "nis_flag_rate": nis_flagged / n,
            "both_flagged": both_flagged,
            "jaccard_overlap": jaccard,
            "ml_score_nis_rank_correlation": rank_correlation,
        },
    )


if __name__ == "__main__":
    main()
