"""Naive, non-adaptive baseline detector: no Kalman filter, no per-category
tuning, no learned anything -- just a fixed physical sanity bound on the
speed/turn-rate implied by two consecutive raw broadcasts. This is the "did
we even need the rest of this project" row the benchmark compares everything
else against; if a fusion score can't clearly beat this, that's a real
finding, not something to paper over.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from absproj.config import RuleBasedConfig
from absproj.geo import latlon_to_enu
from absproj.ingestion.normalize import StateVector


def check_rule_based(prev: Optional[StateVector], curr: StateVector, config: RuleBasedConfig) -> bool:
    """True (anomalous) if the position implied by prev -> curr requires an
    implausible speed or turn rate. The first update in a track (prev=None,
    nothing to compare against yet) is never flagged."""
    if prev is None:
        return False

    dt = (curr.observed_at - prev.observed_at).total_seconds()
    if dt <= 0:
        return False

    dx, dy, dz = latlon_to_enu(
        curr.latitude, curr.longitude, curr.preferred_altitude(),
        prev.latitude, prev.longitude, prev.preferred_altitude(),
    )
    implied_speed = float(np.sqrt(dx * dx + dy * dy + dz * dz)) / dt
    if implied_speed > config.max_speed_mps:
        return True

    if prev.true_track is not None and curr.true_track is not None:
        delta_heading = abs(((curr.true_track - prev.true_track + 180.0) % 360.0) - 180.0)
        turn_rate = delta_heading / dt
        if turn_rate > config.max_turn_rate_deg_s:
            return True

    return False
