import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from absproj.evaluation.identity_check import IdentityTracker  # noqa: E402
from absproj.ingestion.normalize import StateVector  # noqa: E402

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _sv(callsign):
    return StateVector(
        icao24="abc123", callsign=callsign, origin_country="Germany",
        time_position=None, last_contact=int(T0.timestamp()),
        longitude=10.0, latitude=50.0, baro_altitude=9000.0, on_ground=False,
        velocity=200.0, true_track=90.0, vertical_rate=0.0,
        geo_altitude=9000.0, squawk=None, spi=False, position_source=0,
        category=4, observed_at=T0,
    )


def test_first_update_establishes_baseline_never_flagged():
    tracker = IdentityTracker()
    assert tracker.check(_sv("DLH9LF")) is False


def test_same_callsign_never_flagged():
    tracker = IdentityTracker()
    tracker.check(_sv("DLH9LF"))
    assert tracker.check(_sv("DLH9LF")) is False
    assert tracker.check(_sv("DLH9LF")) is False


def test_changed_callsign_flagged_and_stays_flagged():
    """The key behavior a pairwise comparison got wrong: every row after the
    change should stay flagged, not just the first one."""
    tracker = IdentityTracker()
    tracker.check(_sv("DLH9LF"))  # establishes baseline
    assert tracker.check(_sv("BAW123")) is True
    assert tracker.check(_sv("BAW123")) is True  # still flagged, even though internally consistent now
    assert tracker.check(_sv("BAW123")) is True


def test_blank_callsign_rows_never_flagged_and_dont_reset_baseline():
    tracker = IdentityTracker()
    tracker.check(_sv("DLH9LF"))
    assert tracker.check(_sv(None)) is False
    assert tracker.check(_sv("   ")) is False
    # baseline should still be DLH9LF, unaffected by the blank rows in between
    assert tracker.check(_sv("BAW123")) is True


def test_all_blank_never_flags_and_never_establishes_baseline():
    tracker = IdentityTracker()
    assert tracker.check(_sv(None)) is False
    assert tracker.check(_sv("   ")) is False
    # first real callsign establishes baseline rather than being compared
    assert tracker.check(_sv("DLH9LF")) is False
    assert tracker.check(_sv("DLH9LF")) is False


def test_whitespace_normalized_not_flagged():
    tracker = IdentityTracker()
    tracker.check(_sv("DLH9LF"))
    assert tracker.check(_sv("DLH9LF  ")) is False
    assert tracker.check(_sv("  DLH9LF")) is False


def test_independent_trackers_do_not_share_state():
    t1 = IdentityTracker()
    t2 = IdentityTracker()
    t1.check(_sv("DLH9LF"))
    t2.check(_sv("BAW123"))
    assert t1.check(_sv("DLH9LF")) is False
    assert t2.check(_sv("BAW123")) is False
