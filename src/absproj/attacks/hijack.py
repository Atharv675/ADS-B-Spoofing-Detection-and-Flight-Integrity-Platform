"""Track hijacking: mid-flight, a real track's continuation is swapped for a
different real aircraft's motion pattern (transplanted, not a fresh
fabrication -- so it's still smooth and physically plausible in isolation,
just not a continuation of *this* aircraft's actual flight).

Severity controls an additional deliberate position discontinuity injected at
the splice point, on top of whatever velocity/heading discontinuity comes
naturally from suddenly adopting a different aircraft's motion. After the
splice there is no real physical aircraft actually flying that transplanted
path under this identity, so true_sv=None from the splice point on -- MLAT/
radar find no corroboration for the entire post-hijack segment, same as a
ghost, while NIS/ML see the kinematic discontinuity at the splice itself and
however unusual the transplanted dynamics look against this track's own
established category-based envelope.
"""
from __future__ import annotations

from absproj.attacks.types import AttackClass, AttackedTrack
from absproj.attacks.util import graft_segment
from absproj.ingestion.normalize import StateVector


def generate_track_hijack(
    base_track: list[StateVector],
    donor_track: list[StateVector],
    rng,
    severity: float,
    variant_id: str,
) -> AttackedTrack:
    """severity: extra position discontinuity (meters) injected at the splice,
    beyond whatever jump the donor-transplant naturally introduces."""
    n = len(base_track)
    lo, hi = int(n * 0.3), max(int(n * 0.3) + 1, int(n * 0.7))
    hijack_index = int(rng.integers(lo, hi))

    remaining_len = min(n - hijack_index, len(donor_track))
    donor_offset = int(rng.integers(0, max(1, len(donor_track) - remaining_len + 1)))
    donor_segment = donor_track[donor_offset:donor_offset + remaining_len]

    icao24 = base_track[0].icao24
    grafted = graft_segment(
        donor_segment=donor_segment,
        anchor_sv=base_track[hijack_index - 1],
        timing_rows=base_track[hijack_index:hijack_index + remaining_len],
        discontinuity_m=severity,
        rng=rng,
        icao24=icao24,
    )

    broadcast = list(base_track[:hijack_index]) + grafted
    true_svs: list = list(base_track[:hijack_index]) + [None] * len(grafted)
    labels = [False] * hijack_index + [True] * len(grafted)

    return AttackedTrack(
        attack_class=AttackClass.TRACK_HIJACK,
        variant_id=variant_id,
        severity=severity,
        icao24=icao24,
        broadcast_state_vectors=broadcast,
        true_state_vectors=true_svs,
        is_attacked=labels,
        params={
            "hijack_index": hijack_index,
            "donor_icao24": donor_track[0].icao24,
            "extra_discontinuity_m": severity,
        },
    )
