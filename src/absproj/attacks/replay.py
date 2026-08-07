"""Replay attack: a previously-valid recording of a real (but different,
already-completed) aircraft's track is rebroadcast as if it were live,
spliced into an ongoing track's stream.

Severity is a direct position discontinuity (meters) injected at the splice
point -- deliberately the same physical quantity and scale as
track_hijack.severity, not a derived "staleness x speed" estimate. An earlier
version tried to model staleness in seconds and convert it to an implied
jump via the replayed segment's mean speed, but that formula pushed every
variant's implied jump into the tens-to-hundreds of kilometers even at the
lowest configured severity -- comfortably past both MLAT's and radar's
thresholds regardless of severity, so severity stopped being a meaningful
knob and the whole class scored close to 100% by construction rather than
by genuine detection margin. A direct meters-based severity, spanning
sub-threshold to super-threshold, is both simpler to reason about and
actually exercises the detectors across their real operating range.

As with hijack, there's no real physical aircraft actually generating the
replayed positions under this identity right now, so true_sv=None for the
whole replayed segment -- MLAT/radar find no corroboration for it regardless
of the injected discontinuity's size.
"""
from __future__ import annotations

from absproj.attacks.types import AttackClass, AttackedTrack
from absproj.attacks.util import graft_segment
from absproj.ingestion.normalize import StateVector


def generate_replay(
    live_track: list[StateVector],
    replay_source_track: list[StateVector],
    rng,
    severity: float,
    variant_id: str,
) -> AttackedTrack:
    """severity: extra position discontinuity (meters) injected at the splice."""
    n = len(live_track)
    lo, hi = int(n * 0.3), max(int(n * 0.3) + 1, int(n * 0.7))
    splice_index = int(rng.integers(lo, hi))

    remaining_len = min(n - splice_index, len(replay_source_track))
    offset = int(rng.integers(0, max(1, len(replay_source_track) - remaining_len + 1)))
    replay_segment = replay_source_track[offset:offset + remaining_len]

    icao24 = live_track[0].icao24
    grafted = graft_segment(
        donor_segment=replay_segment,
        anchor_sv=live_track[splice_index - 1],
        timing_rows=live_track[splice_index:splice_index + remaining_len],
        discontinuity_m=severity,
        rng=rng,
        icao24=icao24,
    )

    broadcast = list(live_track[:splice_index]) + grafted
    true_svs: list = list(live_track[:splice_index]) + [None] * len(grafted)
    labels = [False] * splice_index + [True] * len(grafted)

    return AttackedTrack(
        attack_class=AttackClass.REPLAY,
        variant_id=variant_id,
        severity=severity,
        icao24=icao24,
        broadcast_state_vectors=broadcast,
        true_state_vectors=true_svs,
        is_attacked=labels,
        params={
            "splice_index": splice_index,
            "replay_source_icao24": replay_source_track[0].icao24,
            "extra_discontinuity_m": severity,
        },
    )
