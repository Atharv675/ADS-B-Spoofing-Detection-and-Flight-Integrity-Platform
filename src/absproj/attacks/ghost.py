"""Ghost aircraft injection: a fully synthetic track with no corresponding
real aircraft anywhere in it.

Severity controls how much the fabricated motion deviates from smooth,
physically unremarkable flight: at severity=0 the ghost flies a nearly
straight line, which is *not* a weakness of this attack class to fix -- it's
an honest, important finding. NIS only checks self-consistency between a
track's own successive broadcasts; a smooth fabricated trajectory is
perfectly self-consistent, so NIS/ML structurally cannot catch it. Only
MLAT/radar can, because there is genuinely no physical aircraft for either
simulated sensor to return a corroborating plot for (true_sv=None on every
row here) -- which is exactly why this project has independent verification
sources and not just a better temporal model.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from absproj.attacks.types import AttackClass, AttackedTrack
from absproj.attacks.util import random_synthetic_icao24
from absproj.config import BBox
from absproj.geo import enu_to_latlon, velocity_to_enu
from absproj.ingestion.normalize import StateVector

_REPRESENTATIVE_RAW_CATEGORY = 4  # "large" -- an unremarkable typical-airliner emitter category

_SMOOTH_TURN_STD_DEG = 0.5
_ERRATIC_TURN_STD_DEG = 15.0
_SMOOTH_SPEED_STD_MPS = 1.0
_ERRATIC_SPEED_STD_MPS = 15.0


def generate_ghost(
    rng,
    severity: float,
    variant_id: str,
    bbox: BBox,
    start_time: datetime,
    n_steps: int = 20,
    poll_interval_s: float = 15.0,
) -> AttackedTrack:
    icao24 = random_synthetic_icao24(rng)

    lat0 = rng.uniform(bbox.lamin, bbox.lamax)
    lon0 = rng.uniform(bbox.lomin, bbox.lomax)
    altitude = rng.uniform(8000.0, 12000.0)
    speed = rng.uniform(180.0, 260.0)
    heading = rng.uniform(0.0, 360.0)

    turn_std = _SMOOTH_TURN_STD_DEG + severity * (_ERRATIC_TURN_STD_DEG - _SMOOTH_TURN_STD_DEG)
    speed_std = _SMOOTH_SPEED_STD_MPS + severity * (_ERRATIC_SPEED_STD_MPS - _SMOOTH_SPEED_STD_MPS)

    x, y = 0.0, 0.0
    broadcast: list[StateVector] = []
    true_svs: list[None] = []
    labels: list[bool] = []

    for i in range(n_steps):
        heading = (heading + rng.normal(0.0, turn_std)) % 360.0
        speed = max(50.0, speed + rng.normal(0.0, speed_std))
        ve, vn, _ = velocity_to_enu(speed, heading, 0.0)
        x += ve * poll_interval_s
        y += vn * poll_interval_s
        lat, lon, _ = enu_to_latlon(x, y, 0.0, lat0, lon0, 0.0)
        t = start_time + timedelta(seconds=i * poll_interval_s)

        sv = StateVector(
            icao24=icao24, callsign=f"GHOST{icao24[:3].upper()}", origin_country=None,
            time_position=int(t.timestamp()), last_contact=int(t.timestamp()),
            longitude=lon, latitude=lat, baro_altitude=altitude, on_ground=False,
            velocity=speed, true_track=heading, vertical_rate=0.0,
            geo_altitude=altitude, squawk=None, spi=False, position_source=0,
            category=_REPRESENTATIVE_RAW_CATEGORY, observed_at=t,
        )
        broadcast.append(sv)
        true_svs.append(None)
        labels.append(True)

    return AttackedTrack(
        attack_class=AttackClass.GHOST,
        variant_id=variant_id,
        severity=severity,
        icao24=icao24,
        broadcast_state_vectors=broadcast,
        true_state_vectors=true_svs,
        is_attacked=labels,
        params={"start_lat": lat0, "start_lon": lon0, "start_speed_mps": speed, "start_heading_deg": heading},
    )
