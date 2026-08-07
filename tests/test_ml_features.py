import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from absproj.config import MLConfig, IsolationForestConfig  # noqa: E402
from absproj.ml.features import (  # noqa: E402
    CATEGORY_DUMMY_COLUMNS,
    FEATURE_COLUMNS,
    FlightPhase,
    PHASE_DUMMY_COLUMNS,
    build_feature_frame,
    classify_flight_phase,
    feature_matrix,
)

ML_CONFIG = MLConfig(
    rolling_window=3,
    level_flight_vz_threshold_mps=1.5,
    isolation_forest=IsolationForestConfig(n_estimators=50, contamination=0.05, random_state=0),
)

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _row(icao24, t_offset_s, nis, ix=1.0, iy=1.0, iz=0.0, vz=0.0, category="transport"):
    return {
        "time": T0 + timedelta(seconds=t_offset_s),
        "icao24": icao24,
        "category": category,
        "dt_seconds": 15.0,
        "innovation_x": ix,
        "innovation_y": iy,
        "innovation_z": iz,
        "nis": nis,
        "vx": 100.0,
        "vy": 0.0,
        "vz": vz,
        "nis_is_anomalous": nis > 11.34,
    }


def test_classify_flight_phase():
    assert classify_flight_phase(5.0, 1.5) == FlightPhase.CLIMB
    assert classify_flight_phase(-5.0, 1.5) == FlightPhase.DESCENT
    assert classify_flight_phase(0.5, 1.5) == FlightPhase.LEVEL


def test_build_feature_frame_empty():
    df = build_feature_frame([], ML_CONFIG)
    assert len(df) == 0


def test_build_feature_frame_has_all_feature_columns():
    rows = [_row("abc123", i * 15, nis=1.0) for i in range(5)]
    df = build_feature_frame(rows, ML_CONFIG)
    for col in FEATURE_COLUMNS:
        assert col in df.columns


def test_category_one_hot_is_mutually_exclusive():
    rows = [_row("abc123", 0, nis=1.0, category="light")]
    df = build_feature_frame(rows, ML_CONFIG)
    row = df.iloc[0]
    assert row["category_light"] == 1.0
    assert row["category_transport"] == 0.0
    assert row["category_high_performance"] == 0.0
    assert sum(row[c] for c in CATEGORY_DUMMY_COLUMNS) == 1.0


def test_phase_one_hot_is_mutually_exclusive():
    rows = [_row("abc123", 0, nis=1.0, vz=10.0)]
    df = build_feature_frame(rows, ML_CONFIG)
    row = df.iloc[0]
    assert row["phase_climb"] == 1.0
    assert sum(row[c] for c in PHASE_DUMMY_COLUMNS) == 1.0


def test_innovation_magnitude_computed_correctly():
    rows = [_row("abc123", 0, nis=1.0, ix=3.0, iy=4.0, iz=0.0)]
    df = build_feature_frame(rows, ML_CONFIG)
    assert df.iloc[0]["innovation_magnitude"] == 5.0


def test_rolling_mean_nis_is_causal_and_correct():
    # NIS sequence 1,2,3 for a single track; window=3 -> rolling means: 1, 1.5, 2
    rows = [_row("abc123", i * 15, nis=float(v)) for i, v in enumerate([1, 2, 3])]
    df = build_feature_frame(rows, ML_CONFIG)
    df = df.sort_values("time").reset_index(drop=True)
    assert df["nis_roll_mean"].tolist() == [1.0, 1.5, 2.0]


def test_rolling_features_are_per_track_isolated():
    # Two tracks interleaved in input order; rolling stats must not leak across tracks.
    rows = [
        _row("trackA", 0, nis=10.0),
        _row("trackB", 0, nis=1.0),
        _row("trackA", 15, nis=10.0),
        _row("trackB", 15, nis=1.0),
    ]
    df = build_feature_frame(rows, ML_CONFIG)
    a_means = df[df.icao24 == "trackA"].sort_values("time")["nis_roll_mean"].tolist()
    b_means = df[df.icao24 == "trackB"].sort_values("time")["nis_roll_mean"].tolist()
    assert a_means == [10.0, 10.0]
    assert b_means == [1.0, 1.0]


def test_feature_matrix_shape_matches_feature_columns():
    rows = [_row("abc123", i * 15, nis=1.0) for i in range(4)]
    df = build_feature_frame(rows, ML_CONFIG)
    X = feature_matrix(df)
    assert X.shape == (4, len(FEATURE_COLUMNS))
