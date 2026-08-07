"""SIMULATED multilateration (MLAT) verification.

*** This is a software simulation, not a real sensor network. *** There are no
real ground receivers, no real RF timing measurements. Everything in this
module -- receiver coordinates, time-difference-of-arrival, timing noise -- is
synthetic, generated for this project. It must never be presented as, or
mistaken for, live independent sensor corroboration.

What it actually does: takes an aircraft's position (currently, in Phase 4,
its own ADS-B broadcast position -- there is no other ground truth available
yet), simulates what a small fixed network of ground receivers would have
measured (time-difference-of-arrival, with injected timing noise standing in
for real receiver clock/multipath error), solves the resulting hyperbolic
positioning equations for an independent *horizontal* position estimate, and
compares that estimate back against the broadcast horizontal position.

Horizontal only, altitude fixed from the aircraft's own report: a network of
ground-level receivers has poor vertical GDOP looking up at a target tens of
thousands of feet overhead -- confirmed empirically here (an earlier 3D solve
attempt showed ~400m of noise-driven disagreement dominated by altitude error
alone, even for a well-placed aircraft near the network centroid). Real
terrestrial MLAT systems have this same limitation and handle it the same
way: fix altitude from the aircraft's own barometric/geometric report (which
this module still gets from ADS-B, same as in reality) and solve TDOA for
horizontal position only, which is what ground-based multilateration can
actually resolve well.

check() (used by Phases 4/5's clean-traffic verification) simulates from and
compares against the same broadcast position -- on clean traffic the
"independent" estimate is expected to agree closely almost by construction,
since both trace back to the same input. That's not a limitation of the
method, it's what validates that the TDOA generation + hyperbolic solve is
*geometrically and numerically correct* before anything is asked of it.

check_with_ground_truth() (Phase 6+) is where this becomes a genuine
discriminator: the adversarial testbed knows each attack's real synthetic
"true" trajectory separately from what gets broadcast, so TDOA is simulated
from the true position and compared against the (possibly falsified)
broadcast position -- at that point disagreement is actually informative, not
tautological.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
from scipy.optimize import least_squares

from absproj.config import MLATConfig
from absproj.geo import enu_to_latlon, latlon_to_enu
from absproj.ingestion.normalize import StateVector


@dataclass
class Receiver:
    name: str
    enu: np.ndarray  # (3,), relative to the shared MLAT origin


@dataclass
class MLATResult:
    time: datetime
    icao24: str
    mlat_latitude: float
    mlat_longitude: float
    mlat_altitude: float  # pass-through of the input altitude, not independently solved -- see module docstring
    disagreement_m: float  # horizontal only
    residual: float
    is_anomalous: bool
    no_corroboration: bool = False  # True when there was no physical target to simulate a return from at all


class MLATSimulator:
    def __init__(self, config: MLATConfig, origin_lat: float, origin_lon: float):
        """origin_lat/lon: the shared ENU frame origin for the whole simulated
        receiver network -- fixed (e.g. the polling bbox center), not per-track,
        since multiple receivers and many aircraft all need a common frame.
        """
        self.config = config
        self.origin = (origin_lat, origin_lon, 0.0)
        self.receivers = [
            Receiver(name=r.name, enu=np.array(latlon_to_enu(r.lat, r.lon, r.alt, *self.origin)))
            for r in config.receivers
        ]
        self.ref_idx = config.reference_receiver_index
        self._other_idx = [i for i in range(len(self.receivers)) if i != self.ref_idx]
        self.rng = np.random.default_rng(config.random_seed)

    def _simulate_range_diffs(self, true_pos: np.ndarray) -> np.ndarray:
        """true_pos: full 3D (x,y,z) -- propagation range genuinely depends on
        altitude even though we only solve for horizontal position."""
        c = self.config.speed_of_light_mps
        ranges = np.array([np.linalg.norm(true_pos - r.enu) for r in self.receivers])
        ref_range = ranges[self.ref_idx]

        diffs = []
        for i in self._other_idx:
            noise_s = self.rng.normal(0.0, self.config.timing_noise_std_ns * 1e-9)
            tdoa_s = (ranges[i] - ref_range) / c + noise_s
            diffs.append(tdoa_s * c)
        return np.array(diffs)

    def _residuals(self, xy: np.ndarray, z: float, range_diffs: np.ndarray) -> np.ndarray:
        p = np.array([xy[0], xy[1], z])
        ref_range = np.linalg.norm(p - self.receivers[self.ref_idx].enu)
        out = np.empty(len(self._other_idx))
        for k, i in enumerate(self._other_idx):
            ri = np.linalg.norm(p - self.receivers[i].enu)
            out[k] = (ri - ref_range) - range_diffs[k]
        return out

    def _solve(self, range_diffs: np.ndarray, z: float) -> tuple[np.ndarray, float]:
        """Multi-start Levenberg-Marquardt over horizontal position only (z
        fixed -- see module docstring). Two seeds (network centroid + the
        reference receiver's own position, on opposite sides of the likely
        solution) guard against the hyperbolic system's known susceptibility
        to spurious local minima -- the full 7-seed sweep used while first
        debugging the (now-replaced) 3D solve was needed for that harder,
        poorly-conditioned problem; this 2D one converges reliably from far
        fewer starts, which matters at this project's batch sizes (one solve
        per track_state row).
        """
        centroid_xy = np.mean([r.enu[:2] for r in self.receivers], axis=0)
        horizontal_seeds = [centroid_xy, self.receivers[self.ref_idx].enu[:2]]

        best_x, best_residual = None, np.inf
        for x0 in horizontal_seeds:
            result = least_squares(self._residuals, x0, args=(z, range_diffs), method="lm", max_nfev=200)
            residual = float(np.sum(result.fun ** 2))
            if residual < best_residual:
                best_x, best_residual = result.x, residual

        return best_x, best_residual

    def check(self, sv: StateVector) -> MLATResult:
        """Phase 4 behavior, unchanged: simulates from and compares against the
        same position (broadcast treated as truth, since no other ground truth
        exists yet). Equivalent to check_with_ground_truth(sv, true_sv=sv)."""
        return self.check_with_ground_truth(broadcast_sv=sv, true_sv=sv)

    def check_with_ground_truth(self, broadcast_sv: StateVector, true_sv: Optional[StateVector]) -> MLATResult:
        """The Phase 6+ form: simulates the TDOA/solve from the *true* physical
        position, then compares the resulting independent estimate against
        the (possibly falsified) *broadcast* position -- this is what makes
        MLAT a genuine discriminator against position-spoofing attacks rather
        than a tautology.

        true_sv=None means there is no real physical aircraft at all (a ghost
        injection, or a hijacked/replayed segment with no corresponding real
        target) -- a real MLAT network would simply get no corroborating TDOA
        return to solve in the first place, which we represent directly as
        "no corroboration", not as an attempted (meaningless) simulation.
        """
        if true_sv is None:
            return MLATResult(
                time=broadcast_sv.observed_at,
                icao24=broadcast_sv.icao24,
                mlat_latitude=float("nan"),
                mlat_longitude=float("nan"),
                mlat_altitude=float("nan"),
                disagreement_m=float("inf"),
                residual=float("nan"),
                is_anomalous=True,
                no_corroboration=True,
            )

        true_altitude = true_sv.preferred_altitude()
        true_pos = np.array(latlon_to_enu(true_sv.latitude, true_sv.longitude, true_altitude, *self.origin))

        range_diffs = self._simulate_range_diffs(true_pos)
        solved_xy, residual = self._solve(range_diffs, true_altitude)

        broadcast_pos_xy = np.array(
            latlon_to_enu(broadcast_sv.latitude, broadcast_sv.longitude, broadcast_sv.preferred_altitude(), *self.origin)
        )[:2]
        disagreement_m = float(np.linalg.norm(solved_xy - broadcast_pos_xy))
        mlat_lat, mlat_lon, mlat_alt = enu_to_latlon(solved_xy[0], solved_xy[1], true_altitude, *self.origin)

        return MLATResult(
            time=broadcast_sv.observed_at,
            icao24=broadcast_sv.icao24,
            mlat_latitude=mlat_lat,
            mlat_longitude=mlat_lon,
            mlat_altitude=mlat_alt,
            disagreement_m=disagreement_m,
            residual=residual,
            is_anomalous=disagreement_m > self.config.disagreement_threshold_m,
        )
