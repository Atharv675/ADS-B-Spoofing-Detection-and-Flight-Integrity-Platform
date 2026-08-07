import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from absproj.evaluation.fusion import EvidenceFusion, FusionRow  # noqa: E402


def _synthetic_rows(rng, n_clean=150, n_attacked=50):
    rows = []
    labels = []
    for _ in range(n_clean):
        rows.append(FusionRow(
            nis=rng.uniform(0.0, 3.0), ml_score=rng.uniform(-0.2, 0.0),
            mlat_disagreement_m=rng.uniform(0.0, 50.0), mlat_no_corroboration=False,
            radar_disagreement_m=rng.uniform(0.0, 500.0), radar_no_corroboration=False,
        ))
        labels.append(False)
    for _ in range(n_attacked):
        rows.append(FusionRow(
            nis=rng.uniform(50.0, 500.0), ml_score=rng.uniform(0.1, 0.4),
            mlat_disagreement_m=rng.uniform(1000.0, 5000.0), mlat_no_corroboration=False,
            radar_disagreement_m=rng.uniform(5000.0, 20000.0), radar_no_corroboration=False,
        ))
        labels.append(True)
    return rows, labels


def test_fusion_separates_synthetic_clean_and_attacked():
    rng = np.random.default_rng(0)
    rows, labels = _synthetic_rows(rng)

    fusion = EvidenceFusion(random_state=1)
    fusion.fit(rows, labels)

    predicted = fusion.is_anomalous(rows)
    labels_arr = np.array(labels)
    recall = predicted[labels_arr].mean()
    fpr = predicted[~labels_arr].mean()
    assert recall > 0.9
    assert fpr < 0.1


def test_integrity_score_bounds_and_direction():
    rng = np.random.default_rng(2)
    rows, labels = _synthetic_rows(rng)
    fusion = EvidenceFusion(random_state=1)
    fusion.fit(rows, labels)

    scores = fusion.integrity_score(rows)
    assert np.all(scores >= 0.0) and np.all(scores <= 100.0)

    labels_arr = np.array(labels)
    # Attacked rows should have lower integrity (more suspicious) on average.
    assert scores[labels_arr].mean() < scores[~labels_arr].mean()


def test_no_corroboration_does_not_crash_and_is_treated_as_suspicious():
    rng = np.random.default_rng(3)
    rows, labels = _synthetic_rows(rng)
    fusion = EvidenceFusion(random_state=1)
    fusion.fit(rows, labels)

    no_corrob_row = FusionRow(
        nis=1.0, ml_score=-0.1,
        mlat_disagreement_m=0.0, mlat_no_corroboration=True,
        radar_disagreement_m=0.0, radar_no_corroboration=True,
    )
    score = fusion.integrity_score([no_corrob_row])[0]
    assert 0.0 <= score <= 100.0
    assert score < 50.0  # capped disagreement should read as suspicious despite low nis/ml


def test_scoring_before_fit_raises():
    fusion = EvidenceFusion()
    row = FusionRow(nis=1.0, ml_score=0.0, mlat_disagreement_m=0.0,
                     mlat_no_corroboration=False, radar_disagreement_m=0.0, radar_no_corroboration=False)
    try:
        fusion.suspicion([row])
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
