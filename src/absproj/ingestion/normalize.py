"""Maps raw OpenSky /states/all rows into a consistent internal schema.

Raw row layout (index -> field), per OpenSky's documented state vector format
with extended=1: https://openskynetwork.github.io/opensky-api/rest.html
  0 icao24, 1 callsign, 2 origin_country, 3 time_position, 4 last_contact,
  5 longitude, 6 latitude, 7 baro_altitude, 8 on_ground, 9 velocity,
  10 true_track, 11 vertical_rate, 12 sensors, 13 geo_altitude, 14 squawk,
  15 spi, 16 position_source, 17 category
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

_MIN_FIELDS = 17  # everything through position_source; category (17) is optional


@dataclass
class StateVector:
    icao24: str
    callsign: Optional[str]
    origin_country: Optional[str]
    time_position: Optional[int]
    last_contact: int
    longitude: float
    latitude: float
    baro_altitude: Optional[float]
    on_ground: bool
    velocity: Optional[float]
    true_track: Optional[float]
    vertical_rate: Optional[float]
    geo_altitude: Optional[float]
    squawk: Optional[str]
    spi: bool
    position_source: Optional[int]
    category: Optional[int]
    observed_at: datetime  # server-reported batch time, UTC

    def preferred_altitude(self) -> float:
        """Geo (GNSS) altitude over barometric where both exist -- barometric
        altitude is a pressure reading, not a position measurement, and can
        carry a systematic offset unrelated to the aircraft's actual position.
        Shared by every downstream consumer that needs a single altitude value
        (Kalman filter, MLAT simulator, ...)."""
        if self.geo_altitude is not None:
            return self.geo_altitude
        if self.baro_altitude is not None:
            return self.baro_altitude
        return 0.0


def normalize_state_vector(row: list[Any], batch_time: int) -> Optional[StateVector]:
    """Returns a StateVector, or None if the row is malformed/unusable.

    A row is unusable for our purposes if it lacks a position fix (lon/lat are
    None for aircraft OpenSky hasn't positioned yet) -- everything downstream
    (Kalman filter, storage) needs a position, so we drop those here rather than
    push nulls further into the pipeline.
    """
    if not isinstance(row, (list, tuple)) or len(row) < _MIN_FIELDS:
        logger.warning("normalize_row_too_short", extra={"row_len": len(row) if hasattr(row, "__len__") else None})
        return None

    icao24 = row[0]
    longitude = row[5]
    latitude = row[6]
    last_contact = row[4]

    if not icao24 or longitude is None or latitude is None or last_contact is None:
        return None

    try:
        longitude = float(longitude)
        latitude = float(latitude)
        last_contact = int(last_contact)
    except (TypeError, ValueError):
        logger.warning("normalize_row_bad_types", extra={"icao24": icao24})
        return None

    if not (-180.0 <= longitude <= 180.0) or not (-90.0 <= latitude <= 90.0):
        logger.warning("normalize_row_position_out_of_range", extra={"icao24": icao24, "lon": longitude, "lat": latitude})
        return None

    def _opt_float(v: Any) -> Optional[float]:
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    category = None
    if len(row) > 17:
        try:
            category = int(row[17]) if row[17] is not None else None
        except (TypeError, ValueError):
            category = None

    return StateVector(
        icao24=str(icao24).strip().lower(),
        callsign=(row[1].strip() if isinstance(row[1], str) and row[1].strip() else None),
        origin_country=row[2] if isinstance(row[2], str) else None,
        time_position=int(row[3]) if row[3] is not None else None,
        last_contact=last_contact,
        longitude=longitude,
        latitude=latitude,
        baro_altitude=_opt_float(row[7]),
        on_ground=bool(row[8]),
        velocity=_opt_float(row[9]),
        true_track=_opt_float(row[10]),
        vertical_rate=_opt_float(row[11]),
        geo_altitude=_opt_float(row[13]),
        squawk=row[14] if isinstance(row[14], str) else None,
        spi=bool(row[15]),
        position_source=int(row[16]) if row[16] is not None else None,
        category=category,
        observed_at=datetime.fromtimestamp(batch_time, tz=timezone.utc),
    )


def normalize_batch(body: dict[str, Any]) -> list[StateVector]:
    """Normalizes an entire /states/all response body. Skips malformed rows,
    never raises -- a batch with some bad rows should still yield the good ones.
    """
    batch_time = body.get("time")
    states = body.get("states") or []
    if batch_time is None:
        logger.error("normalize_batch_missing_time")
        return []

    out: list[StateVector] = []
    skipped = 0
    for row in states:
        sv = normalize_state_vector(row, batch_time)
        if sv is None:
            skipped += 1
            continue
        out.append(sv)

    logger.info(
        "normalize_batch_done",
        extra={"total_rows": len(states), "normalized": len(out), "skipped": skipped},
    )
    return out
