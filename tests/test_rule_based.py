import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from absproj.config import RuleBasedConfig  # noqa: E402
from absproj.evaluation.rule_based import check_rule_based  # noqa: E402
from absproj.ingestion.normalize import StateVector  # noqa: E402

CONFIG = RuleBasedConfig(max_speed_mps=350.0, max_turn_rate_deg_s=10.0)
T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _sv(lat, lon, true_track, t_offset_s, velocity=200.0):
    t = T0 + timedelta(seconds=t_offset_s)
    return StateVector(
        icao24="abc123", callsign="TEST", origin_country="Germany",
        time_position=None, last_contact=int(t.timestamp()),
        longitude=lon, latitude=lat, baro_altitude=9000.0, on_ground=False,
        velocity=velocity, true_track=true_track, vertical_rate=0.0,
        geo_altitude=9000.0, squawk=None, spi=False, position_source=0,
        category=4, observed_at=t,
    )


def test_first_update_never_flagged():
    curr = _sv(50.0, 10.0, 90.0, 0)
    assert check_rule_based(None, curr, CONFIG) is False


def test_normal_motion_not_flagged():
    prev = _sv(50.0, 10.0, 90.0, 0)
    curr = _sv(50.0, 10.03, 90.0, 15)  # ~2.2km in 15s -> ~146 m/s, plausible
    assert check_rule_based(prev, curr, CONFIG) is False


def test_implausible_speed_flagged():
    prev = _sv(50.0, 10.0, 90.0, 0)
    curr = _sv(51.5, 10.0, 90.0, 15)  # ~167km in 15s, absurd
    assert check_rule_based(prev, curr, CONFIG) is True


def test_implausible_turn_rate_flagged():
    prev = _sv(50.0, 10.0, 10.0, 0)
    curr = _sv(50.0001, 10.0001, 200.0, 15)  # heading swings 170 deg in 15s -> ~11.3 deg/s
    assert check_rule_based(prev, curr, CONFIG) is True


def test_nonpositive_dt_not_flagged():
    prev = _sv(50.0, 10.0, 90.0, 15)
    curr = _sv(51.0, 11.0, 90.0, 15)  # same timestamp, huge position diff
    assert check_rule_based(prev, curr, CONFIG) is False
