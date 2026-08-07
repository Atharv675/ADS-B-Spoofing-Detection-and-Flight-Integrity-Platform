"""Identity consistency check: flags broadcasts whose callsign differs from
the *established* callsign for that track, not just the single row where a
change first happens.

Targets the one attack class every other detector in this project is largely
blind to: ICAO identity collision (see attacks/icao_collision.py). MLAT/radar
check whether a broadcast position matches an independent physical
measurement -- in a collision attack the position is real (the intruder
aircraft genuinely is where it says it is), only the *identity* is
fraudulent, so position-truth checks structurally can't see it. NIS/ML check
kinematic self-consistency and pick up some signal from the resulting
Kalman-filter confusion, but neither looks at the one field directly being
lied about: identity. This does.

Why a *tracker* (established baseline) and not a pairwise previous-vs-current
comparison: a pairwise version was tried first and measured at ~10% recall
on real ICAO-collision test data despite correctly catching every collision's
onset. The reason: it only fires at the exact row where callsign changes --
once the intruder's own run of broadcasts becomes internally consistent
(intruder-row compared to the previous intruder-row, same callsign each
time), a pairwise check has nothing left to flag, even though every one of
those rows is still fraudulent. Comparing against the track's established
(first-seen) callsign instead flags the entire intruder segment, matching
how the attack is actually labeled (the whole segment, not just its first
row).

Requires callsign to actually be present in track_state -- see migration
006_track_state_callsign.sql. Earlier phases silently discarded callsign
after ingestion (track_state never had the column; only tracks.last_callsign,
a latest-value-only field that can't show a mid-track change). Existing
accumulated data predates that fix and has NULL callsign for those rows,
which this check treats as "no information," not "no mismatch."
"""
from __future__ import annotations

from typing import Optional

from absproj.ingestion.normalize import StateVector


class IdentityTracker:
    """Stateful, one instance per track (mirrors KalmanTrackManager's
    per-track state, just simpler): records the first non-blank callsign
    seen as the track's established identity, then flags every later
    broadcast whose non-blank callsign differs from it.

    A real, honest limitation: a genuine mid-flight callsign reassignment
    (rare, but does happen operationally) would be flagged as a persistent
    mismatch from that point on, the same as an actual identity collision --
    this check cannot tell the two apart, and it will over-flag if the
    former occurs. That tradeoff is accepted here because the former is rare
    and the latter (identity fraud) is exactly what this check exists to
    catch persistently, not just at its first appearance.
    """

    def __init__(self) -> None:
        self._established: Optional[str] = None

    def check(self, sv: StateVector) -> bool:
        cs = (sv.callsign or "").strip()
        if not cs:
            return False
        if self._established is None:
            self._established = cs
            return False
        return cs != self._established
