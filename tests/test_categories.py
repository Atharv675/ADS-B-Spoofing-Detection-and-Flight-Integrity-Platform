import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from absproj.tracking.categories import CategoryBucket, categorize  # noqa: E402


def test_raw_category_light():
    assert categorize(raw_category=2, velocity=200.0, altitude=9000.0, vertical_rate=0.0) == CategoryBucket.LIGHT


def test_raw_category_rotorcraft_is_light():
    assert categorize(raw_category=8, velocity=60.0, altitude=500.0, vertical_rate=0.0) == CategoryBucket.LIGHT


def test_raw_category_high_performance():
    assert categorize(raw_category=7, velocity=100.0, altitude=1000.0, vertical_rate=0.0) == CategoryBucket.HIGH_PERFORMANCE


def test_raw_category_heavy_is_transport():
    assert categorize(raw_category=6, velocity=250.0, altitude=11000.0, vertical_rate=0.0) == CategoryBucket.TRANSPORT


def test_unknown_category_falls_back_to_velocity_heuristic_light():
    # raw_category=0 (no info) + slow, low altitude -> light
    assert categorize(raw_category=0, velocity=40.0, altitude=1000.0, vertical_rate=0.0) == CategoryBucket.LIGHT


def test_unknown_category_falls_back_to_velocity_heuristic_high_performance():
    assert categorize(raw_category=1, velocity=300.0, altitude=10000.0, vertical_rate=0.0) == CategoryBucket.HIGH_PERFORMANCE


def test_unknown_category_high_vertical_rate_is_high_performance():
    assert categorize(raw_category=None, velocity=150.0, altitude=5000.0, vertical_rate=30.0) == CategoryBucket.HIGH_PERFORMANCE


def test_unknown_category_typical_airliner_defaults_transport():
    assert categorize(raw_category=None, velocity=230.0, altitude=11000.0, vertical_rate=2.0) == CategoryBucket.TRANSPORT


def test_all_none_defaults_transport():
    assert categorize(raw_category=None, velocity=None, altitude=None, vertical_rate=None) == CategoryBucket.TRANSPORT
