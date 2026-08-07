"""SIMULATED primary (skin-paint) radar verification.

*** This is a software simulation, not a real sensor. *** There is no real
radar site, no real RF returns. Everything here -- the site location, range/
azimuth measurements, noise -- is synthetic, generated for this project. It
must never be presented as, or mistaken for, a live independent sensor.

Real primary radar reports position + velocity only -- no identity, no
transponder data, unlike ADS-B or even MLAT (which at least resolves an
ICAO24-keyed TDOA solve). This module's output mirrors that: RadarResult
carries no callsign/category/etc. It keys results by icao24 purely as
simulation bookkeeping, to compare a synthetic plot back against "the same"
aircraft's broadcast -- a simplification real radar-to-ADS-B fusion doesn't
get for free (real systems need a separate plot-to-track association/gating
algorithm, which is out of scope here).

Deliberately small and deliberately different from MLAT: one fixed site
(not a network), measuring range + azimuth (polar, from that one site) with
independent Gaussian noise on each, rather than MLAT's multi-receiver TDOA.
Velocity is derived by finite-differencing two synthetic plots a few seconds
apart (extrapolating the earlier one from the aircraft's own reported
velocity), the same way real scan-to-scan primary radar tracking works --
radar doesn't measure velocity directly either. Because error here is driven
by *angular* precision from a single site, it grows with range from the site,
unlike MLAT's networked geometry -- a genuinely different error signature,
which is the point of having two independent simulated sources rather than
one.

check() (Phases 4/5) simulates from and compares against the same broadcast
position/velocity -- appropriate for clean-traffic verification where there's
no other ground truth. check_with_ground_truth() (Phase 6+) simulates the
radar return from the aircraft's real synthetic trajectory and compares
against the (possibly falsified) broadcast, the same true-vs-broadcast split
used in mlat.py -- see that module's docstring for why, and for what
true_sv=None (no real physical aircraft at all) means.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np

from absproj.config import RadarConfig
from absproj.geo import enu_to_latlon, latlon_to_enu, velocity_to_enu
from absproj.ingestion.normalize import StateVector


@dataclass
class RadarResult:
    time: datetime
    icao24: str
    radar_latitude: float
    radar_longitude: float
    radar_vx: float  # east, m/s, finite-differenced
    radar_vy: float  # north, m/s, finite-differenced
    disagreement_m: float  # horizontal, radar plot vs. broadcast position
    velocity_disagreement_mps: float | None
    is_anomalous: bool
    no_corroboration: bool = False  # True when there was no physical target to return a plot from at all


def _enu_to_range_azimuth(xy: np.ndarray) -> tuple[float, float]:
    rng = float(np.linalg.norm(xy))
    azimuth = math.atan2(xy[0], xy[1])  # 0 = north, increases clockwise (east positive)
    return rng, azimuth


def _range_azimuth_to_enu(rng: float, azimuth: float) -> np.ndarray:
    return np.array([rng * math.sin(azimuth), rng * math.cos(azimuth)])


class RadarSimulator:
    def __init__(self, config: RadarConfig):
        """The radar site is the ENU origin for this module -- range is then
        just the vector norm, no separate site-offset bookkeeping needed."""
        self.config = config
        self.site = (config.site.lat, config.site.lon, 0.0)
        self.rng = np.random.default_rng(config.random_seed)

    def _noisy_plot(self, true_xy: np.ndarray) -> np.ndarray:
        rng_m, azimuth = _enu_to_range_azimuth(true_xy)
        rng_m += self.rng.normal(0.0, self.config.range_noise_std_m)
        azimuth += self.rng.normal(0.0, math.radians(self.config.azimuth_noise_std_deg))
        return _range_azimuth_to_enu(rng_m, azimuth)

    def check(self, sv: StateVector) -> RadarResult:
        """Phase 4/5 behavior, unchanged: simulates from and compares against
        the same position/velocity. Equivalent to
        check_with_ground_truth(sv, true_sv=sv)."""
        return self.check_with_ground_truth(broadcast_sv=sv, true_sv=sv)

    @staticmethod
    def _horizontal_velocity(sv: StateVector) -> Optional[np.ndarray]:
        if sv.velocity is None or sv.true_track is None:
            return None
        ve, vn, _ = velocity_to_enu(sv.velocity, sv.true_track, 0.0)
        return np.array([ve, vn])

    def check_with_ground_truth(self, broadcast_sv: StateVector, true_sv: Optional[StateVector]) -> RadarResult:
        """Simulates the radar return from the *true* physical trajectory,
        then compares the resulting plot (position and finite-differenced
        velocity) against the (possibly falsified) *broadcast* state --- see
        module docstring. true_sv=None means no real physical aircraft, i.e.
        no return for a real radar to have picked up at all.
        """
        if true_sv is None:
            return RadarResult(
                time=broadcast_sv.observed_at,
                icao24=broadcast_sv.icao24,
                radar_latitude=float("nan"),
                radar_longitude=float("nan"),
                radar_vx=float("nan"),
                radar_vy=float("nan"),
                disagreement_m=float("inf"),
                velocity_disagreement_mps=None,
                is_anomalous=True,
                no_corroboration=True,
            )

        pos_now_true = np.array(latlon_to_enu(true_sv.latitude, true_sv.longitude, 0.0, *self.site)[:2])
        true_velocity = self._horizontal_velocity(true_sv)
        pos_prev_true = pos_now_true - (true_velocity if true_velocity is not None else np.zeros(2)) * self.config.plot_interval_s

        plot_prev = self._noisy_plot(pos_prev_true)
        plot_now = self._noisy_plot(pos_now_true)
        radar_velocity = (plot_now - plot_prev) / self.config.plot_interval_s

        broadcast_pos = np.array(latlon_to_enu(broadcast_sv.latitude, broadcast_sv.longitude, 0.0, *self.site)[:2])
        disagreement_m = float(np.linalg.norm(plot_now - broadcast_pos))

        broadcast_velocity = self._horizontal_velocity(broadcast_sv)
        velocity_disagreement = (
            float(np.linalg.norm(radar_velocity - broadcast_velocity)) if broadcast_velocity is not None else None
        )

        radar_lat, radar_lon, _ = enu_to_latlon(plot_now[0], plot_now[1], 0.0, *self.site)

        return RadarResult(
            time=broadcast_sv.observed_at,
            icao24=broadcast_sv.icao24,
            radar_latitude=radar_lat,
            radar_longitude=radar_lon,
            radar_vx=float(radar_velocity[0]),
            radar_vy=float(radar_velocity[1]),
            disagreement_m=disagreement_m,
            velocity_disagreement_mps=velocity_disagreement,
            is_anomalous=disagreement_m > self.config.disagreement_threshold_m,
        )
