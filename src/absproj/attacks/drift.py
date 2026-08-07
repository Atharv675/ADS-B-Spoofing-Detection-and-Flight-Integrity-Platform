"""Position spoofing / drift: a real track's broadcast position is falsified
starting at some point, either suddenly (a jump) or gradually (a ramp), while
every other broadcast field (velocity, heading, altitude, squawk, callsign)
stays exactly what the real aircraft actually reported -- "other fields stay
plausible" per the brief. The real aircraft's true trajectory is unchanged and
kept as true_state_vectors, which is what makes this the clearest case for
MLAT/radar: they simulate from the real position and should catch the
broadcast diverging from it.
"""
from __future__ import annotations

import math

from absproj.attacks.types import AttackClass, AttackedTrack
from absproj.attacks.util import clone_sv
from absproj.geo import enu_to_latlon
from absproj.ingestion.normalize import StateVector

RAMP_UPDATES = 5


def generate_position_drift(
    base_track: list[StateVector],
    rng,
    severity: float,
    variant_id: str,
    mode: str = "sudden",
) -> AttackedTrack:
    """severity: final broadcast offset magnitude in meters. mode: "sudden"
    (offset applied at full magnitude from the first attacked row) or
    "gradual" (linearly ramped up over RAMP_UPDATES rows)."""
    if mode not in ("sudden", "gradual"):
        raise ValueError(f"unknown drift mode: {mode}")

    n = len(base_track)
    lo, hi = int(n * 0.3), max(int(n * 0.3) + 1, int(n * 0.7))
    attack_start = int(rng.integers(lo, hi))

    angle = rng.uniform(0.0, 2 * math.pi)
    direction = (math.sin(angle), math.cos(angle))  # unit vector: (east, north)

    broadcast: list[StateVector] = []
    true_svs: list[StateVector] = []
    labels: list[bool] = []

    for i, sv in enumerate(base_track):
        if i < attack_start:
            broadcast.append(sv)
            true_svs.append(sv)
            labels.append(False)
            continue

        if mode == "sudden":
            frac = 1.0
        else:
            frac = min(1.0, (i - attack_start + 1) / RAMP_UPDATES)
        offset_m = severity * frac

        new_lat, new_lon, _ = enu_to_latlon(
            offset_m * direction[0], offset_m * direction[1], 0.0, sv.latitude, sv.longitude, 0.0
        )
        broadcast.append(clone_sv(sv, latitude=new_lat, longitude=new_lon))
        true_svs.append(sv)  # real aircraft's actual position, unchanged
        labels.append(True)

    return AttackedTrack(
        attack_class=AttackClass.POSITION_DRIFT,
        variant_id=variant_id,
        severity=severity,
        icao24=base_track[0].icao24,
        broadcast_state_vectors=broadcast,
        true_state_vectors=true_svs,
        is_attacked=labels,
        params={
            "mode": mode,
            "attack_start_index": attack_start,
            "direction_east": direction[0],
            "direction_north": direction[1],
            "donor_icao24": base_track[0].icao24,
        },
    )
