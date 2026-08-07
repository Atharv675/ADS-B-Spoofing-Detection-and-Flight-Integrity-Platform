"""Loads real clean track segments to use as substrate ("donor") material for
attacks that modify or recombine real trajectories (everything except ghost,
which has no substrate by construction).

Deliberately reads from the same already-ingested, already-verified-clean
track_state history the earlier phases used -- these are real flights, so any
physical realism the attacks inherit (plausible speed, altitude, smooth
turns) comes for free from actually having been a real aircraft, right up
until the point an attack starts modifying them.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

from absproj.ingestion.normalize import StateVector


@dataclass
class SubstratePool:
    tracks: dict[str, list[StateVector]]  # icao24 -> chronological StateVectors

    def icao24s(self) -> list[str]:
        return list(self.tracks.keys())

    def sample_track(self, rng) -> list[StateVector]:
        icao24 = rng.choice(self.icao24s())
        return self.tracks[icao24]

    def sample_two_distinct_tracks(self, rng) -> tuple[list[StateVector], list[StateVector]]:
        icao24s = self.icao24s()
        a, b = rng.choice(icao24s, size=2, replace=False)
        return self.tracks[a], self.tracks[b]


_MAX_GAP_SECONDS = 60.0  # splits a track into separate contiguous runs across bigger gaps than this


def _longest_contiguous_run(rows: list[StateVector]) -> list[StateVector]:
    runs: list[list[StateVector]] = [[rows[0]]]
    for prev, cur in zip(rows, rows[1:]):
        gap = (cur.observed_at - prev.observed_at).total_seconds()
        if gap <= _MAX_GAP_SECONDS:
            runs[-1].append(cur)
        else:
            runs.append([cur])
    return max(runs, key=len)


def build_substrate_pool(state_vectors: list[StateVector], min_length: int) -> SubstratePool:
    """state_vectors: ordered by (icao24, time), e.g. straight from
    repository.fetch_track_state_history_for_kalman(). Real accumulated
    history has gaps (rate limiting, etc.) -- only the longest gap-free run
    per aircraft is kept, so substrate tracks are physically continuous
    flight segments, not a real track's rows with an artificial jump baked
    in from an ingestion hiccup.
    """
    tracks: dict[str, list[StateVector]] = {}
    for icao24, group in itertools.groupby(state_vectors, key=lambda sv: sv.icao24):
        run = _longest_contiguous_run(list(group))
        if len(run) >= min_length:
            tracks[icao24] = run
    return SubstratePool(tracks=tracks)
