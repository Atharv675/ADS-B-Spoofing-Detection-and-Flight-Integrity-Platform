"""Aircraft dynamics category buckets used to set the Kalman filter's process
noise (i.e. how much unmodeled acceleration -- turning, climbing -- the filter
should tolerate before treating it as an anomaly rather than normal flight).

A single fixed threshold across all aircraft is wrong: a widebody's plausible
turn/climb rate is nowhere near a light aircraft's or a high-performance
aircraft's. We use OpenSky's reported ADS-B emitter category when it's
populated (real feeds; it is often 0/unknown in practice), and fall back to a
velocity/altitude/vertical-rate heuristic otherwise.

OpenSky category codes (ADS-B emitter category, DO-260B):
  0/1 = no info, 2 = light, 3 = small, 4 = large, 5 = high vortex large,
  6 = heavy, 7 = high performance, 8 = rotorcraft, 9 = glider/sailplane,
  10 = lighter-than-air, 11-19 = parachutist, ultralight, UAV, space vehicle,
  surface vehicles, obstacles.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

_LIGHT_RAW_CATEGORIES = frozenset({2, 8, 9, 10, 11, 12})
_HIGH_PERF_RAW_CATEGORIES = frozenset({7})
_TRANSPORT_RAW_CATEGORIES = frozenset({3, 4, 5, 6})

# Fallback heuristic thresholds (used when raw category is 0/1/None -- i.e.
# unknown, which is the common case on real OpenSky traffic).
_LIGHT_MAX_VELOCITY_MPS = 80.0
_LIGHT_MAX_ALTITUDE_M = 4000.0
_HIGH_PERF_MIN_VELOCITY_MPS = 260.0
_HIGH_PERF_MIN_ABS_VERTICAL_RATE_MPS = 20.0


class CategoryBucket(str, Enum):
    LIGHT = "light"
    TRANSPORT = "transport"
    HIGH_PERFORMANCE = "high_performance"


def categorize(
    raw_category: Optional[int],
    velocity: Optional[float],
    altitude: Optional[float],
    vertical_rate: Optional[float],
) -> CategoryBucket:
    if raw_category is not None:
        if raw_category in _LIGHT_RAW_CATEGORIES:
            return CategoryBucket.LIGHT
        if raw_category in _HIGH_PERF_RAW_CATEGORIES:
            return CategoryBucket.HIGH_PERFORMANCE
        if raw_category in _TRANSPORT_RAW_CATEGORIES:
            return CategoryBucket.TRANSPORT
        # raw_category in {0, 1, or any other unlisted code}: fall through to heuristic.

    if velocity is not None and velocity > _HIGH_PERF_MIN_VELOCITY_MPS:
        return CategoryBucket.HIGH_PERFORMANCE
    if vertical_rate is not None and abs(vertical_rate) > _HIGH_PERF_MIN_ABS_VERTICAL_RATE_MPS:
        return CategoryBucket.HIGH_PERFORMANCE

    if velocity is not None and velocity < _LIGHT_MAX_VELOCITY_MPS:
        if altitude is None or altitude < _LIGHT_MAX_ALTITUDE_M:
            return CategoryBucket.LIGHT

    return CategoryBucket.TRANSPORT
