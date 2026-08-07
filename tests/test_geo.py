import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from absproj.geo import enu_to_latlon, latlon_to_enu, velocity_to_enu  # noqa: E402


def test_enu_round_trip():
    lat0, lon0, alt0 = 50.0, 10.0, 1000.0
    lat, lon, alt = 50.05, 10.08, 1200.0

    e, n, u = latlon_to_enu(lat, lon, alt, lat0, lon0, alt0)
    lat2, lon2, alt2 = enu_to_latlon(e, n, u, lat0, lon0, alt0)

    assert math.isclose(lat, lat2, abs_tol=1e-9)
    assert math.isclose(lon, lon2, abs_tol=1e-9)
    assert math.isclose(alt, alt2, abs_tol=1e-9)


def test_enu_origin_is_zero():
    lat0, lon0, alt0 = 50.0, 10.0, 1000.0
    e, n, u = latlon_to_enu(lat0, lon0, alt0, lat0, lon0, alt0)
    assert (e, n, u) == (0.0, 0.0, 0.0)


def test_enu_north_offset_is_positive_north_only():
    lat0, lon0, alt0 = 50.0, 10.0, 0.0
    e, n, u = latlon_to_enu(50.01, 10.0, 0.0, lat0, lon0, alt0)
    assert n > 0
    assert math.isclose(e, 0.0, abs_tol=1e-6)


def test_enu_east_offset_is_positive_east_only():
    lat0, lon0, alt0 = 50.0, 10.0, 0.0
    e, n, u = latlon_to_enu(50.0, 10.01, 0.0, lat0, lon0, alt0)
    assert e > 0
    assert math.isclose(n, 0.0, abs_tol=1e-6)


def test_velocity_to_enu_north_heading():
    ve, vn, vu = velocity_to_enu(speed=100.0, true_track_deg=0.0, vertical_rate=5.0)
    assert math.isclose(ve, 0.0, abs_tol=1e-9)
    assert math.isclose(vn, 100.0, abs_tol=1e-9)
    assert vu == 5.0


def test_velocity_to_enu_east_heading():
    ve, vn, vu = velocity_to_enu(speed=100.0, true_track_deg=90.0, vertical_rate=0.0)
    assert math.isclose(ve, 100.0, abs_tol=1e-6)
    assert math.isclose(vn, 0.0, abs_tol=1e-6)


def test_velocity_to_enu_south_heading():
    ve, vn, vu = velocity_to_enu(speed=50.0, true_track_deg=180.0, vertical_rate=0.0)
    assert math.isclose(ve, 0.0, abs_tol=1e-6)
    assert math.isclose(vn, -50.0, abs_tol=1e-6)
