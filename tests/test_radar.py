import math
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from absproj.config import RadarConfig, RadarSiteConfig  # noqa: E402
from absproj.ingestion.normalize import StateVector  # noqa: E402
from absproj.verification.radar import (  # noqa: E402
    RadarSimulator,
    _enu_to_range_azimuth,
    _range_azimuth_to_enu,
)

SITE_LAT, SITE_LON = 50.0, 10.0


def _config(range_noise_std_m=0.0, azimuth_noise_std_deg=0.0, disagreement_threshold_m=1500.0, seed=456):
    return RadarConfig(
        site=RadarSiteConfig(lat=SITE_LAT, lon=SITE_LON),
        plot_interval_s=4.0,
        range_noise_std_m=range_noise_std_m,
        azimuth_noise_std_deg=azimuth_noise_std_deg,
        disagreement_threshold_m=disagreement_threshold_m,
        random_seed=seed,
    )


def _state_vector(lat, lon, velocity=200.0, true_track=90.0, icao24="abc123"):
    return StateVector(
        icao24=icao24, callsign="TEST", origin_country="Germany",
        time_position=None, last_contact=0,
        longitude=lon, latitude=lat, baro_altitude=9000.0, on_ground=False,
        velocity=velocity, true_track=true_track, vertical_rate=0.0,
        geo_altitude=9000.0, squawk=None, spi=False, position_source=0,
        category=3, observed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def test_range_azimuth_round_trip():
    import numpy as np
    xy = np.array([1234.5, -6789.0])
    rng, az = _enu_to_range_azimuth(xy)
    xy2 = _range_azimuth_to_enu(rng, az)
    assert math.isclose(xy[0], xy2[0], abs_tol=1e-6)
    assert math.isclose(xy[1], xy2[1], abs_tol=1e-6)


def test_azimuth_north_is_zero():
    import numpy as np
    rng, az = _enu_to_range_azimuth(np.array([0.0, 1000.0]))
    assert math.isclose(az, 0.0, abs_tol=1e-9)


def test_azimuth_east_is_90_degrees():
    import numpy as np
    rng, az = _enu_to_range_azimuth(np.array([1000.0, 0.0]))
    assert math.isclose(math.degrees(az), 90.0, abs_tol=1e-6)


def test_zero_noise_recovers_position_exactly():
    sim = RadarSimulator(_config())
    sv = _state_vector(lat=50.3, lon=10.4)
    result = sim.check(sv)
    assert result.disagreement_m < 1e-6
    assert result.is_anomalous is False


def test_zero_noise_recovers_velocity_for_known_heading():
    # Due east at 200 m/s: radar-derived velocity should match almost exactly
    # with zero measurement noise.
    sim = RadarSimulator(_config())
    sv = _state_vector(lat=50.3, lon=10.4, velocity=200.0, true_track=90.0)
    result = sim.check(sv)
    assert math.isclose(result.radar_vx, 200.0, abs_tol=1e-4)
    assert math.isclose(result.radar_vy, 0.0, abs_tol=1e-4)
    assert result.velocity_disagreement_mps < 1e-6


def test_realistic_noise_keeps_disagreement_well_under_threshold_near_site():
    # Close to the radar site, even modest angular noise translates to tiny
    # cross-range error, so disagreement should be small relative to a
    # generously-set threshold.
    sim = RadarSimulator(_config(range_noise_std_m=50.0, azimuth_noise_std_deg=0.2))
    sv = _state_vector(lat=50.05, lon=10.05)
    result = sim.check(sv)
    assert result.disagreement_m < 1500.0
    assert result.is_anomalous is False


def test_error_grows_with_range_from_site():
    # Same angular noise budget, farther from the site -> larger typical
    # cross-range error. Compare disagreement magnitude at two distances
    # using a fixed seed (deterministic) rather than a statistical test.
    near = _state_vector(lat=50.05, lon=10.05, icao24="near")
    far = _state_vector(lat=54.9, lon=14.9, icao24="far")  # near the bbox's far corner

    sim_near = RadarSimulator(_config(range_noise_std_m=50.0, azimuth_noise_std_deg=0.2, seed=1))
    sim_far = RadarSimulator(_config(range_noise_std_m=50.0, azimuth_noise_std_deg=0.2, seed=1))

    r_near = sim_near.check(near)
    r_far = sim_far.check(far)

    assert r_far.disagreement_m > r_near.disagreement_m


def test_large_noise_can_trigger_anomalous_flag():
    sim = RadarSimulator(_config(range_noise_std_m=100_000.0, azimuth_noise_std_deg=0.0, disagreement_threshold_m=1500.0))
    sv = _state_vector(lat=50.3, lon=10.4)
    result = sim.check(sv)
    assert result.disagreement_m > 1500.0
    assert result.is_anomalous is True


def test_deterministic_given_fixed_seed():
    sv = _state_vector(lat=50.3, lon=10.4)
    sim1 = RadarSimulator(_config(range_noise_std_m=50.0, azimuth_noise_std_deg=0.2, seed=9))
    sim2 = RadarSimulator(_config(range_noise_std_m=50.0, azimuth_noise_std_deg=0.2, seed=9))
    r1 = sim1.check(sv)
    r2 = sim2.check(sv)
    assert r1.disagreement_m == r2.disagreement_m
    assert r1.radar_latitude == r2.radar_latitude


def test_missing_velocity_treated_as_stationary_and_does_not_crash():
    sim = RadarSimulator(_config())
    sv = _state_vector(lat=50.3, lon=10.4, velocity=None, true_track=None)
    result = sim.check(sv)
    assert result.velocity_disagreement_mps is None
    assert result.disagreement_m < 1e-6  # zero noise, stationary assumption still exact for position


def test_aircraft_at_site_position_does_not_crash():
    sim = RadarSimulator(_config())
    sv = _state_vector(lat=SITE_LAT, lon=SITE_LON, velocity=0.0, true_track=0.0)
    result = sim.check(sv)
    assert result.disagreement_m < 1e-6


def test_no_corroboration_when_true_sv_is_none():
    sim = RadarSimulator(_config())
    broadcast_sv = _state_vector(lat=50.3, lon=10.4)
    result = sim.check_with_ground_truth(broadcast_sv=broadcast_sv, true_sv=None)
    assert result.no_corroboration is True
    assert result.is_anomalous is True
    assert result.disagreement_m == float("inf")


def test_ground_truth_catches_falsified_broadcast_position():
    sim = RadarSimulator(_config(disagreement_threshold_m=200.0))
    true_sv = _state_vector(lat=50.3, lon=10.4)
    broadcast_sv = _state_vector(lat=50.34, lon=10.4)  # ~4.4km further north than reality

    result = sim.check_with_ground_truth(broadcast_sv=broadcast_sv, true_sv=true_sv)

    assert result.disagreement_m > 1000.0
    assert result.is_anomalous is True


def test_ground_truth_matches_legacy_check_when_true_equals_broadcast():
    sim1 = RadarSimulator(_config(range_noise_std_m=50.0, azimuth_noise_std_deg=0.2, seed=3))
    sim2 = RadarSimulator(_config(range_noise_std_m=50.0, azimuth_noise_std_deg=0.2, seed=3))
    sv = _state_vector(lat=50.3, lon=10.4)

    r1 = sim1.check(sv)
    r2 = sim2.check_with_ground_truth(broadcast_sv=sv, true_sv=sv)

    assert r1.disagreement_m == r2.disagreement_m
