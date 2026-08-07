"""Local tangent-plane (East-North-Up) conversions used by the Kalman filter and,
later, the MLAT/radar solvers -- all of which need to do linear-ish math on
positions that lat/lon degrees don't support directly.

Approximation: flat-earth equirectangular projection anchored at a per-track
origin (that track's first observed position). This is deliberately not
geodesy-grade -- error grows with distance from the origin, but each track's
own excursion within the polled bbox (~1100km across) stays well within the
range where this approximation's distortion (well under 1% at these scales) is
negligible next to our detection thresholds (tens to hundreds of meters). A
real multi-sensor system would use a proper ellipsoidal projection; this is
adequate for consistency-checking one aircraft's short-term motion against
itself.
"""
from __future__ import annotations

import math

WGS84_A = 6378137.0  # WGS84 semi-major axis, meters


def latlon_to_enu(
    lat: float, lon: float, alt: float,
    lat0: float, lon0: float, alt0: float,
) -> tuple[float, float, float]:
    """Converts (lat, lon, alt) to local (east, north, up) meters relative to
    the origin (lat0, lon0, alt0)."""
    lat0_rad = math.radians(lat0)
    dlat = math.radians(lat - lat0)
    dlon = math.radians(lon - lon0)
    east = WGS84_A * math.cos(lat0_rad) * dlon
    north = WGS84_A * dlat
    up = alt - alt0
    return east, north, up


def enu_to_latlon(
    east: float, north: float, up: float,
    lat0: float, lon0: float, alt0: float,
) -> tuple[float, float, float]:
    """Inverse of latlon_to_enu."""
    lat0_rad = math.radians(lat0)
    lat = lat0 + math.degrees(north / WGS84_A)
    lon = lon0 + math.degrees(east / (WGS84_A * math.cos(lat0_rad)))
    alt = alt0 + up
    return lat, lon, alt


def velocity_to_enu(speed: float, true_track_deg: float, vertical_rate: float) -> tuple[float, float, float]:
    """Converts ADS-B reported ground speed (m/s) + true track (deg, clockwise
    from north) + vertical rate (m/s, positive up) into (ve, vn, vu) m/s."""
    heading_rad = math.radians(true_track_deg)
    ve = speed * math.sin(heading_rad)
    vn = speed * math.cos(heading_rad)
    vu = vertical_rate
    return ve, vn, vu
