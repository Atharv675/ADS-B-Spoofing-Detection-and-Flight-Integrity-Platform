"""ICAO identity collision: two real, independent aircraft trajectories
broadcast under the same ICAO24 identifier, overlapping in time.

Unlike position drift, neither stream's *position* is falsified -- the
intruder aircraft is really wherever it says it is. The lie is purely about
identity. That makes this the one attack class MLAT/radar structurally can't
help with (they corroborate physical position, not claimed identity); it's
primarily a Kalman/NIS signature, since a single per-ICAO24 filter switching
between two spatially separated real trajectories produces large, genuine
innovations at the collision boundaries.
"""
from __future__ import annotations

from datetime import timedelta

from absproj.attacks.types import AttackClass, AttackedTrack
from absproj.attacks.util import clone_sv
from absproj.ingestion.normalize import StateVector


def generate_icao_collision(
    victim_track: list[StateVector],
    intruder_track: list[StateVector],
    rng,
    severity: float,
    variant_id: str,
) -> AttackedTrack:
    """severity: fraction (0-1) of the intruder's own broadcast duration that
    ends up overlapping the victim's time window -- higher severity places
    the intruder's stream more squarely inside the victim's, producing more
    directly simultaneous conflicting reports."""
    victim_icao24 = victim_track[0].icao24
    intruder_icao24 = intruder_track[0].icao24

    victim_end = victim_track[-1].observed_at
    intruder_start = intruder_track[0].observed_at
    intruder_end = intruder_track[-1].observed_at
    intruder_duration = (intruder_end - intruder_start).total_seconds()

    shifted_start = victim_end - timedelta(seconds=severity * intruder_duration)
    time_shift = shifted_start - intruder_start

    broadcast: list[StateVector] = []
    true_svs: list[StateVector] = []
    labels: list[bool] = []

    for sv in victim_track:
        broadcast.append(sv)
        true_svs.append(sv)
        labels.append(False)

    for sv in intruder_track:
        shifted_time = sv.observed_at + time_shift
        relabeled = clone_sv(
            sv,
            icao24=victim_icao24,
            observed_at=shifted_time,
            last_contact=int(shifted_time.timestamp()),
            time_position=int(shifted_time.timestamp()) if sv.time_position is not None else None,
        )
        broadcast.append(relabeled)
        # The intruder is a real aircraft genuinely at that position -- only
        # its claimed identity is fraudulent, so true == broadcast here.
        true_svs.append(relabeled)
        labels.append(True)

    order = sorted(range(len(broadcast)), key=lambda i: broadcast[i].observed_at)
    broadcast = [broadcast[i] for i in order]
    true_svs = [true_svs[i] for i in order]
    labels = [labels[i] for i in order]

    return AttackedTrack(
        attack_class=AttackClass.ICAO_COLLISION,
        variant_id=variant_id,
        severity=severity,
        icao24=victim_icao24,
        broadcast_state_vectors=broadcast,
        true_state_vectors=true_svs,
        is_attacked=labels,
        params={
            "victim_icao24": victim_icao24,
            "intruder_icao24": intruder_icao24,
            "overlap_fraction": severity,
            "time_shift_seconds": time_shift.total_seconds(),
        },
    )
