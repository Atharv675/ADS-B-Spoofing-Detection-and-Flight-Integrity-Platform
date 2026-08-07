import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from absproj.config import MLATConfig, ReceiverConfig  # noqa: E402
from absproj.ingestion.normalize import StateVector  # noqa: E402
from absproj.verification.mlat import MLATSimulator  # noqa: E402

RECEIVERS = [
    ReceiverConfig(name="a", lat=48.78, lon=9.18, alt=250),
    ReceiverConfig(name="b", lat=48.14, lon=11.58, alt=520),
    ReceiverConfig(name="c", lat=47.45, lon=8.55, alt=430),
    ReceiverConfig(name="d", lat=52.37, lon=9.73, alt=55),
    ReceiverConfig(name="e", lat=52.52, lon=13.40, alt=35),
    ReceiverConfig(name="f", lat=45.46, lon=9.19, alt=120),
]

ORIGIN_LAT, ORIGIN_LON = 50.0, 10.0


def _config(timing_noise_std_ns=0.0, disagreement_threshold_m=400.0, seed=123):
    return MLATConfig(
        receivers=RECEIVERS,
        reference_receiver_index=0,
        speed_of_light_mps=299792458.0,
        timing_noise_std_ns=timing_noise_std_ns,
        disagreement_threshold_m=disagreement_threshold_m,
        random_seed=seed,
    )


def _state_vector(lat, lon, alt, icao24="abc123"):
    return StateVector(
        icao24=icao24, callsign="TEST", origin_country="Germany",
        time_position=None, last_contact=0,
        longitude=lon, latitude=lat, baro_altitude=alt, on_ground=False,
        velocity=200.0, true_track=90.0, vertical_rate=0.0,
        geo_altitude=alt, squawk=None, spi=False, position_source=0,
        category=3, observed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def test_zero_noise_recovers_true_position_almost_exactly():
    sim = MLATSimulator(_config(timing_noise_std_ns=0.0), ORIGIN_LAT, ORIGIN_LON)
    sv = _state_vector(lat=50.3, lon=10.4, alt=9000.0)

    result = sim.check(sv)

    # With zero timing noise, the hyperbolic solve should recover the true
    # position to well within a meter (limited only by solver tolerance).
    assert result.disagreement_m < 1.0
    assert result.residual < 1e-6
    assert result.is_anomalous is False


def test_zero_noise_recovers_position_at_multiple_locations():
    sim = MLATSimulator(_config(timing_noise_std_ns=0.0), ORIGIN_LAT, ORIGIN_LON)
    for lat, lon, alt in [(49.0, 8.0, 3000.0), (51.5, 12.0, 11500.0), (46.5, 11.0, 6000.0)]:
        sv = _state_vector(lat=lat, lon=lon, alt=alt)
        result = sim.check(sv)
        assert result.disagreement_m < 1.0


def test_realistic_noise_keeps_disagreement_small_relative_to_threshold():
    # 50ns timing noise -> ~15m range-diff noise; with 6 receivers spread over
    # a wide baseline, position error should stay well under the 400m gate.
    sim = MLATSimulator(_config(timing_noise_std_ns=50.0, disagreement_threshold_m=400.0), ORIGIN_LAT, ORIGIN_LON)
    sv = _state_vector(lat=50.3, lon=10.4, alt=9000.0)

    result = sim.check(sv)

    assert result.disagreement_m < 400.0
    assert result.is_anomalous is False


def test_large_timing_noise_can_trigger_anomalous_flag():
    # Deliberately unrealistic noise to exercise the flagging path itself.
    sim = MLATSimulator(_config(timing_noise_std_ns=200_000.0, disagreement_threshold_m=400.0), ORIGIN_LAT, ORIGIN_LON)
    sv = _state_vector(lat=50.3, lon=10.4, alt=9000.0)

    result = sim.check(sv)

    assert result.disagreement_m > 400.0
    assert result.is_anomalous is True


def test_deterministic_given_fixed_seed():
    sv = _state_vector(lat=50.3, lon=10.4, alt=9000.0)
    sim1 = MLATSimulator(_config(timing_noise_std_ns=50.0, seed=7), ORIGIN_LAT, ORIGIN_LON)
    sim2 = MLATSimulator(_config(timing_noise_std_ns=50.0, seed=7), ORIGIN_LAT, ORIGIN_LON)

    r1 = sim1.check(sv)
    r2 = sim2.check(sv)

    assert r1.disagreement_m == r2.disagreement_m
    assert r1.mlat_latitude == r2.mlat_latitude


def test_result_position_round_trips_near_broadcast_lat_lon():
    sim = MLATSimulator(_config(timing_noise_std_ns=0.0), ORIGIN_LAT, ORIGIN_LON)
    sv = _state_vector(lat=50.3, lon=10.4, alt=9000.0)

    result = sim.check(sv)

    assert abs(result.mlat_latitude - sv.latitude) < 0.001
    assert abs(result.mlat_longitude - sv.longitude) < 0.001


def test_no_corroboration_when_true_sv_is_none():
    sim = MLATSimulator(_config(timing_noise_std_ns=0.0), ORIGIN_LAT, ORIGIN_LON)
    broadcast_sv = _state_vector(lat=50.3, lon=10.4, alt=9000.0)

    result = sim.check_with_ground_truth(broadcast_sv=broadcast_sv, true_sv=None)

    assert result.no_corroboration is True
    assert result.is_anomalous is True
    assert result.disagreement_m == float("inf")


def test_ground_truth_catches_falsified_broadcast_position():
    # Real aircraft (true_sv) stays at its real position; the broadcast lies
    # about being ~5km away. MLAT, simulating from the truth, should catch it.
    sim = MLATSimulator(_config(timing_noise_std_ns=0.0, disagreement_threshold_m=200.0), ORIGIN_LAT, ORIGIN_LON)
    true_sv = _state_vector(lat=50.3, lon=10.4, alt=9000.0)
    broadcast_sv = _state_vector(lat=50.34, lon=10.4, alt=9000.0)  # ~4.4km further north

    result = sim.check_with_ground_truth(broadcast_sv=broadcast_sv, true_sv=true_sv)

    assert result.disagreement_m > 1000.0
    assert result.is_anomalous is True


def test_ground_truth_matches_legacy_check_when_true_equals_broadcast():
    sim1 = MLATSimulator(_config(timing_noise_std_ns=50.0, seed=3), ORIGIN_LAT, ORIGIN_LON)
    sim2 = MLATSimulator(_config(timing_noise_std_ns=50.0, seed=3), ORIGIN_LAT, ORIGIN_LON)
    sv = _state_vector(lat=50.3, lon=10.4, alt=9000.0)

    r1 = sim1.check(sv)
    r2 = sim2.check_with_ground_truth(broadcast_sv=sv, true_sv=sv)

    assert r1.disagreement_m == r2.disagreement_m
