"""Builds the ML feature matrix from Kalman-derived signals only.

Per the brief, this must never touch raw ADS-B fields (reported lat/lon,
velocity, heading, etc.) directly -- every column here is either the Kalman
filter's own output (innovation, NIS, its posterior velocity estimate) or a
transform of that output (rolling trend, category, flight phase derived from
the filter's vz). This is what makes it a genuinely independent second opinion
on top of the physics-based NIS test, rather than a restatement of it.
"""
from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd

from absproj.config import MLConfig

CATEGORY_VALUES = ["light", "transport", "high_performance"]


class FlightPhase(str, Enum):
    CLIMB = "climb"
    DESCENT = "descent"
    LEVEL = "level"


def classify_flight_phase(vz: float, threshold: float) -> FlightPhase:
    if vz > threshold:
        return FlightPhase.CLIMB
    if vz < -threshold:
        return FlightPhase.DESCENT
    return FlightPhase.LEVEL


# The final numeric feature columns fed to the model, in a fixed order so
# training and scoring always agree on column meaning.
NUMERIC_FEATURE_COLUMNS = [
    "nis",
    "innovation_magnitude",
    "innovation_horizontal_mag",
    "innovation_z",
    "dt_seconds",
    "nis_roll_mean",
    "nis_roll_std",
    "innovation_mag_roll_mean",
]
CATEGORY_DUMMY_COLUMNS = [f"category_{c}" for c in CATEGORY_VALUES]
PHASE_DUMMY_COLUMNS = [f"phase_{p.value}" for p in FlightPhase]
FEATURE_COLUMNS = NUMERIC_FEATURE_COLUMNS + CATEGORY_DUMMY_COLUMNS + PHASE_DUMMY_COLUMNS


def build_feature_frame(rows: list[dict], ml_config: MLConfig) -> pd.DataFrame:
    """rows: dicts with keys time, icao24, category, dt_seconds, innovation_x/y/z,
    nis, vz (all straight from kalman_updates). Returns a DataFrame with one row
    per input row (same order not guaranteed -- sorted by icao24,time) and all
    of FEATURE_COLUMNS populated, plus the original identifying/context columns
    (time, icao24, nis, is_anomalous if present) passed through for joining
    results back later.
    """
    if not rows:
        return pd.DataFrame(columns=["time", "icao24"] + FEATURE_COLUMNS)

    df = pd.DataFrame(rows).sort_values(["icao24", "time"]).reset_index(drop=True)

    df["innovation_magnitude"] = np.sqrt(
        df["innovation_x"] ** 2 + df["innovation_y"] ** 2 + df["innovation_z"] ** 2
    )
    df["innovation_horizontal_mag"] = np.sqrt(df["innovation_x"] ** 2 + df["innovation_y"] ** 2)

    df["flight_phase"] = df["vz"].apply(
        lambda vz: classify_flight_phase(vz, ml_config.level_flight_vz_threshold_mps).value
    )

    grp = df.groupby("icao24", sort=False)
    w = ml_config.rolling_window
    df["nis_roll_mean"] = grp["nis"].transform(lambda s: s.rolling(window=w, min_periods=1).mean())
    df["nis_roll_std"] = grp["nis"].transform(lambda s: s.rolling(window=w, min_periods=1).std()).fillna(0.0)
    df["innovation_mag_roll_mean"] = grp["innovation_magnitude"].transform(
        lambda s: s.rolling(window=w, min_periods=1).mean()
    )

    for cat in CATEGORY_VALUES:
        df[f"category_{cat}"] = (df["category"] == cat).astype(float)
    for phase in FlightPhase:
        df[f"phase_{phase.value}"] = (df["flight_phase"] == phase.value).astype(float)

    return df


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[FEATURE_COLUMNS].to_numpy(dtype=float)
