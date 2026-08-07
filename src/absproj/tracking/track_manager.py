"""Maintains one Kalman filter instance per active ICAO24 track and turns a
stream of normalized StateVectors into per-update innovation/NIS records.

This is the component that ties together geo.py (ENU projection), categories.py
(dynamics bucket -> process noise), kalman.py (the filter itself), and nis.py
(the statistical test) into the "per-track Kalman filter" the brief asks for.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np

from absproj.config import KalmanConfig
from absproj.geo import latlon_to_enu, velocity_to_enu
from absproj.ingestion.normalize import StateVector
from absproj.tracking.categories import CategoryBucket, categorize
from absproj.tracking.kalman import KalmanFilterCV
from absproj.tracking.nis import chi_square_threshold, compute_nis

logger = logging.getLogger(__name__)


@dataclass
class KalmanUpdateRecord:
    time: datetime
    icao24: str
    category: CategoryBucket
    dt_seconds: float
    innovation: tuple[float, float, float]
    innovation_cov: list[list[float]]
    nis: float
    chi2_threshold: float
    is_anomalous: bool
    vx: float
    vy: float
    vz: float


class _KalmanTrack:
    def __init__(self, sv: StateVector, kalman_config: KalmanConfig, R: np.ndarray):
        self.origin = (sv.latitude, sv.longitude, sv.preferred_altitude())
        self.last_time = sv.observed_at

        if sv.velocity is not None and sv.true_track is not None:
            ve, vn, vu = velocity_to_enu(sv.velocity, sv.true_track, sv.vertical_rate or 0.0)
        else:
            ve = vn = vu = 0.0

        x0 = np.array([0.0, 0.0, 0.0, ve, vn, vu])
        P0 = np.diag([
            R[0, 0], R[1, 1], R[2, 2],
            kalman_config.initial_velocity_std_mps ** 2,
            kalman_config.initial_velocity_std_mps ** 2,
            kalman_config.initial_velocity_std_mps ** 2,
        ])
        category = categorize(sv.category, sv.velocity, sv.preferred_altitude(), sv.vertical_rate)
        sigma_a = kalman_config.process_noise_sigma_a[category.value]
        self.filter = KalmanFilterCV(x0=x0, P0=P0, R=R, sigma_a=sigma_a)


class KalmanTrackManager:
    def __init__(self, kalman_config: KalmanConfig):
        self.kalman_config = kalman_config
        self.R = np.diag([
            kalman_config.sigma_horizontal_m ** 2,
            kalman_config.sigma_horizontal_m ** 2,
            kalman_config.sigma_vertical_m ** 2,
        ])
        self._tracks: dict[str, _KalmanTrack] = {}
        self.reset_count = 0
        self.skip_count = 0
        self.init_count = 0

    def process(self, sv: StateVector) -> Optional[KalmanUpdateRecord]:
        """Feeds one observation into its track's filter. Returns an update
        record, or None if this observation only (re)initialized the track, or
        was skipped (out-of-order/duplicate timestamp)."""
        track = self._tracks.get(sv.icao24)

        if track is None:
            self._tracks[sv.icao24] = _KalmanTrack(sv, self.kalman_config, self.R)
            self.init_count += 1
            return None

        dt = (sv.observed_at - track.last_time).total_seconds()

        if dt <= 0:
            logger.warning("kalman_skip_nonpositive_dt", extra={"icao24": sv.icao24, "dt": dt})
            self.skip_count += 1
            return None

        if dt > self.kalman_config.track_gap_reset_seconds:
            logger.info("kalman_track_reset_gap", extra={"icao24": sv.icao24, "dt": dt})
            self._tracks[sv.icao24] = _KalmanTrack(sv, self.kalman_config, self.R)
            self.reset_count += 1
            return None

        category = categorize(sv.category, sv.velocity, sv.preferred_altitude(), sv.vertical_rate)
        track.filter.sigma_a = self.kalman_config.process_noise_sigma_a[category.value]

        track.filter.predict(dt)

        z = np.array(latlon_to_enu(sv.latitude, sv.longitude, sv.preferred_altitude(), *track.origin))
        result = track.filter.update(z)

        nis = compute_nis(result.innovation, result.innovation_cov)
        threshold = chi_square_threshold(alpha=self.kalman_config.chi_square_alpha)

        track.last_time = sv.observed_at

        vx, vy, vz = track.filter.x[3], track.filter.x[4], track.filter.x[5]

        return KalmanUpdateRecord(
            time=sv.observed_at,
            icao24=sv.icao24,
            category=category,
            dt_seconds=dt,
            innovation=(float(result.innovation[0]), float(result.innovation[1]), float(result.innovation[2])),
            innovation_cov=result.innovation_cov.tolist(),
            nis=nis,
            chi2_threshold=threshold,
            is_anomalous=nis > threshold,
            vx=float(vx), vy=float(vy), vz=float(vz),
        )
