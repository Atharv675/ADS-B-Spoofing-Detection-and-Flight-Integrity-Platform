import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from absproj.ingestion.normalize import normalize_batch, normalize_state_vector  # noqa: E402

BATCH_TIME = 1_700_000_000

# A realistic complete row: icao24, callsign, origin_country, time_position,
# last_contact, lon, lat, baro_alt, on_ground, velocity, true_track,
# vertical_rate, sensors, geo_alt, squawk, spi, position_source, category
VALID_ROW = [
    "3c6444", "DLH9LF  ", "Germany", BATCH_TIME, BATCH_TIME,
    8.5622, 50.0379, 10972.8, False, 230.5, 91.4,
    0.0, None, 11277.6, "1000", False, 0, 3,
]


def test_normalize_valid_row():
    sv = normalize_state_vector(VALID_ROW, BATCH_TIME)
    assert sv is not None
    assert sv.icao24 == "3c6444"
    assert sv.callsign == "DLH9LF"
    assert sv.longitude == 8.5622
    assert sv.latitude == 50.0379
    assert sv.on_ground is False
    assert sv.category == 3


def test_normalize_row_missing_position_is_dropped():
    row = list(VALID_ROW)
    row[5] = None
    row[6] = None
    assert normalize_state_vector(row, BATCH_TIME) is None


def test_normalize_row_missing_icao24_is_dropped():
    row = list(VALID_ROW)
    row[0] = None
    assert normalize_state_vector(row, BATCH_TIME) is None


def test_normalize_row_out_of_range_position_is_dropped():
    row = list(VALID_ROW)
    row[5] = 500.0  # invalid longitude
    assert normalize_state_vector(row, BATCH_TIME) is None


def test_normalize_row_too_short_is_dropped():
    assert normalize_state_vector(["3c6444"], BATCH_TIME) is None


def test_normalize_row_without_category_still_works():
    row = VALID_ROW[:17]  # drop category
    sv = normalize_state_vector(row, BATCH_TIME)
    assert sv is not None
    assert sv.category is None


def test_normalize_row_bad_types_is_dropped():
    row = list(VALID_ROW)
    row[5] = "not-a-number"
    assert normalize_state_vector(row, BATCH_TIME) is None


def test_normalize_row_blank_callsign_becomes_none():
    row = list(VALID_ROW)
    row[1] = "        "
    sv = normalize_state_vector(row, BATCH_TIME)
    assert sv is not None
    assert sv.callsign is None


def test_normalize_batch_mixed_good_and_bad_rows():
    bad_row = list(VALID_ROW)
    bad_row[0] = "abcdef"
    bad_row[5] = None
    body = {"time": BATCH_TIME, "states": [VALID_ROW, bad_row, None, []]}
    result = normalize_batch(body)
    assert len(result) == 1
    assert result[0].icao24 == "3c6444"


def test_normalize_batch_missing_time_returns_empty():
    assert normalize_batch({"states": [VALID_ROW]}) == []


def test_normalize_batch_empty_states():
    assert normalize_batch({"time": BATCH_TIME, "states": None}) == []
