"""Small shared helpers for the attack generators."""
from __future__ import annotations

import dataclasses
import math

from absproj.geo import enu_to_latlon, latlon_to_enu
from absproj.ingestion.normalize import StateVector


def clone_sv(sv: StateVector, **overrides) -> StateVector:
    return dataclasses.replace(sv, **overrides)


def random_synthetic_icao24(rng) -> str:
    """A random 24-bit hex address, clearly not drawn from real ICAO24 space
    in any structured sense -- fine here since synthetic tracks are always
    kept in their own attack-scenario data, never mixed into real track_state."""
    return f"{rng.integers(0, 0xFFFFFF):06x}"


def graft_segment(
    donor_segment: list[StateVector],
    anchor_sv: StateVector,
    timing_rows: list[StateVector],
    discontinuity_m: float,
    rng,
    icao24: str,
) -> list[StateVector]:
    """Transplants donor_segment's own shape of motion (preserving its
    relative position/altitude changes) so it starts near anchor_sv, offset
    by an extra `discontinuity_m` in a random direction, relabeled to
    icao24, with timestamps taken from timing_rows (same length as
    donor_segment) so the result keeps whatever polling cadence the splice
    site is being inserted into. Used by both hijack.py (donor = a different
    concurrent aircraft) and replay.py (donor = an earlier, already-completed
    recording) -- the two attacks differ in what they pick as the donor and
    how they compute discontinuity_m, not in the splice mechanics itself.
    """
    donor_origin = donor_segment[0]
    donor_origin_alt = donor_origin.preferred_altitude()

    angle = rng.uniform(0.0, 2 * math.pi)
    direction = (math.sin(angle), math.cos(angle))

    grafted = []
    for donor_row, time_row in zip(donor_segment, timing_rows):
        dx, dy, dz = latlon_to_enu(
            donor_row.latitude, donor_row.longitude, donor_row.preferred_altitude(),
            donor_origin.latitude, donor_origin.longitude, donor_origin_alt,
        )
        new_lat, new_lon, new_alt = enu_to_latlon(
            dx + discontinuity_m * direction[0], dy + discontinuity_m * direction[1], dz,
            anchor_sv.latitude, anchor_sv.longitude, anchor_sv.preferred_altitude(),
        )
        grafted.append(clone_sv(
            donor_row,
            icao24=icao24,
            latitude=new_lat,
            longitude=new_lon,
            baro_altitude=new_alt,
            geo_altitude=new_alt,
            observed_at=time_row.observed_at,
            last_contact=int(time_row.observed_at.timestamp()),
            time_position=int(time_row.observed_at.timestamp()) if donor_row.time_position is not None else None,
        ))
    return grafted
