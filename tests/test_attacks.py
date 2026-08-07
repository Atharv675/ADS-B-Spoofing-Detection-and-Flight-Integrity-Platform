import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from absproj.attacks.drift import generate_position_drift  # noqa: E402
from absproj.attacks.ghost import generate_ghost  # noqa: E402
from absproj.attacks.hijack import generate_track_hijack  # noqa: E402
from absproj.attacks.icao_collision import generate_icao_collision  # noqa: E402
from absproj.attacks.injector import generate_all_variants, generate_variants  # noqa: E402
from absproj.attacks.replay import generate_replay  # noqa: E402
from absproj.attacks.split import build_evaluation_split  # noqa: E402
from absproj.attacks.substrate import build_substrate_pool  # noqa: E402
from absproj.attacks.types import AttackClass  # noqa: E402
from absproj.config import AttacksConfig, BBox, SeverityRange  # noqa: E402
from absproj.geo import enu_to_latlon  # noqa: E402
from absproj.ingestion.normalize import StateVector  # noqa: E402

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
BBOX = BBox(lamin=45.0, lamax=55.0, lomin=5.0, lomax=15.0)


def _make_track(icao24, n=20, start_time=T0, start_lat=50.0, start_lon=10.0,
                 speed=200.0, heading=90.0, altitude=9000.0, poll_interval_s=15.0):
    rows = []
    heading_rad = math.radians(heading)
    ve, vn = speed * math.sin(heading_rad), speed * math.cos(heading_rad)
    x, y = 0.0, 0.0
    for i in range(n):
        lat, lon, _ = enu_to_latlon(x, y, 0.0, start_lat, start_lon, 0.0)
        t = start_time + timedelta(seconds=i * poll_interval_s)
        rows.append(StateVector(
            icao24=icao24, callsign="TEST", origin_country="Germany",
            time_position=int(t.timestamp()), last_contact=int(t.timestamp()),
            longitude=lon, latitude=lat, baro_altitude=altitude, on_ground=False,
            velocity=speed, true_track=heading, vertical_rate=0.0,
            geo_altitude=altitude, squawk=None, spi=False, position_source=0,
            category=4, observed_at=t,
        ))
        x += ve * poll_interval_s
        y += vn * poll_interval_s
    return rows


def _default_attacks_config():
    return AttacksConfig(
        min_substrate_track_length=15,
        variants_per_class=8,
        train_fraction=0.5,
        holdout_class="track_hijack",
        random_seed=777,
        severity_ranges={
            "ghost": SeverityRange(min=0.0, max=1.0),
            "position_drift": SeverityRange(min=200.0, max=5000.0),
            "icao_collision": SeverityRange(min=0.3, max=1.0),
            "track_hijack": SeverityRange(min=0.0, max=3000.0),
            "replay": SeverityRange(min=120.0, max=3600.0),
        },
    )


# --- substrate ---

def test_substrate_pool_filters_short_tracks():
    rows = _make_track("aaa111", n=5) + _make_track("bbb222", n=20)
    pool = build_substrate_pool(rows, min_length=15)
    assert "aaa111" not in pool.tracks
    assert "bbb222" in pool.tracks
    assert len(pool.tracks["bbb222"]) == 20


def test_substrate_pool_splits_on_large_gap():
    early = _make_track("ccc333", n=10, start_time=T0)
    later = _make_track("ccc333", n=10, start_time=T0 + timedelta(hours=1))
    rows = early + later
    pool = build_substrate_pool(rows, min_length=8)
    # Neither contiguous run alone reaches min_length=12, so filtered out.
    pool2 = build_substrate_pool(rows, min_length=12)
    assert "ccc333" not in pool2.tracks
    assert "ccc333" in pool.tracks
    assert len(pool.tracks["ccc333"]) == 10  # longest run, not both concatenated


# --- ghost ---

def test_ghost_all_rows_attacked_and_no_corroboration():
    rng = np.random.default_rng(1)
    track = generate_ghost(rng, severity=0.0, variant_id="ghost_00", bbox=BBOX, start_time=T0, n_steps=10)
    assert len(track.broadcast_state_vectors) == 10
    assert all(track.is_attacked)
    assert all(t is None for t in track.true_state_vectors)
    assert track.attack_class == AttackClass.GHOST


def test_ghost_icao24_consistent_across_rows():
    rng = np.random.default_rng(2)
    track = generate_ghost(rng, severity=0.5, variant_id="ghost_01", bbox=BBOX, start_time=T0, n_steps=10)
    icao24s = {sv.icao24 for sv in track.broadcast_state_vectors}
    assert icao24s == {track.icao24}


def test_ghost_higher_severity_more_erratic_heading():
    rng1 = np.random.default_rng(42)
    smooth = generate_ghost(rng1, severity=0.0, variant_id="g_smooth", bbox=BBOX, start_time=T0, n_steps=30)
    rng2 = np.random.default_rng(42)
    erratic = generate_ghost(rng2, severity=1.0, variant_id="g_erratic", bbox=BBOX, start_time=T0, n_steps=30)

    def heading_variance(track):
        headings = [sv.true_track for sv in track.broadcast_state_vectors]
        diffs = [b - a for a, b in zip(headings, headings[1:])]
        return np.var(diffs)

    assert heading_variance(erratic) > heading_variance(smooth)


# --- position drift ---

def test_drift_pre_attack_rows_unchanged():
    base = _make_track("d00001", n=20)
    rng = np.random.default_rng(3)
    track = generate_position_drift(base, rng, severity=1000.0, variant_id="d_00", mode="sudden")
    start = track.params["attack_start_index"]
    for i in range(start):
        assert track.is_attacked[i] is False
        assert track.broadcast_state_vectors[i].latitude == base[i].latitude
        assert track.true_state_vectors[i] is base[i]


def test_drift_sudden_offset_is_constant_magnitude():
    base = _make_track("d00002", n=20)
    rng = np.random.default_rng(4)
    track = generate_position_drift(base, rng, severity=1000.0, variant_id="d_01", mode="sudden")
    start = track.params["attack_start_index"]

    import numpy as _np
    from absproj.geo import latlon_to_enu
    offsets = []
    for i in range(start, len(base)):
        bsv = track.broadcast_state_vectors[i]
        tsv = track.true_state_vectors[i]
        dx, dy, _ = latlon_to_enu(bsv.latitude, bsv.longitude, 0.0, tsv.latitude, tsv.longitude, 0.0)
        offsets.append(_np.hypot(dx, dy))
    assert all(abs(o - 1000.0) < 1.0 for o in offsets)


def test_drift_gradual_offset_ramps_up():
    base = _make_track("d00003", n=20)
    rng = np.random.default_rng(5)
    track = generate_position_drift(base, rng, severity=1000.0, variant_id="d_02", mode="gradual")
    start = track.params["attack_start_index"]

    from absproj.geo import latlon_to_enu
    offsets = []
    for i in range(start, min(start + 5, len(base))):
        bsv = track.broadcast_state_vectors[i]
        tsv = track.true_state_vectors[i]
        dx, dy, _ = latlon_to_enu(bsv.latitude, bsv.longitude, 0.0, tsv.latitude, tsv.longitude, 0.0)
        offsets.append(math.hypot(dx, dy))
    assert offsets == sorted(offsets)  # monotonically non-decreasing
    assert offsets[0] < offsets[-1]


def test_drift_other_fields_stay_plausible():
    base = _make_track("d00004", n=20)
    rng = np.random.default_rng(6)
    track = generate_position_drift(base, rng, severity=1000.0, variant_id="d_03", mode="sudden")
    for bsv, orig in zip(track.broadcast_state_vectors, base):
        assert bsv.velocity == orig.velocity
        assert bsv.true_track == orig.true_track
        assert bsv.baro_altitude == orig.baro_altitude


# --- icao collision ---

def test_collision_labels_and_icao24():
    victim = _make_track("v00001", n=20, start_lat=50.0, start_lon=10.0)
    intruder = _make_track("i00001", n=15, start_lat=48.0, start_lon=8.0)
    rng = np.random.default_rng(7)
    track = generate_icao_collision(victim, intruder, rng, severity=0.5, variant_id="c_00")

    victim_icao24 = victim[0].icao24
    assert all(sv.icao24 == victim_icao24 for sv in track.broadcast_state_vectors)
    n_attacked = sum(track.is_attacked)
    n_clean = len(track.is_attacked) - n_attacked
    assert n_attacked == 15
    assert n_clean == 20


def test_collision_chronological_order():
    victim = _make_track("v00002", n=20)
    intruder = _make_track("i00002", n=15)
    rng = np.random.default_rng(8)
    track = generate_icao_collision(victim, intruder, rng, severity=0.5, variant_id="c_01")
    times = [sv.observed_at for sv in track.broadcast_state_vectors]
    assert times == sorted(times)


def test_collision_true_equals_broadcast_position():
    # Identity is fraudulent, position is not.
    victim = _make_track("v00003", n=20)
    intruder = _make_track("i00003", n=15)
    rng = np.random.default_rng(9)
    track = generate_icao_collision(victim, intruder, rng, severity=0.5, variant_id="c_02")
    for bsv, tsv in zip(track.broadcast_state_vectors, track.true_state_vectors):
        assert tsv is not None
        assert bsv.latitude == tsv.latitude
        assert bsv.longitude == tsv.longitude


# --- track hijack ---

def test_hijack_pre_and_post_labels():
    base = _make_track("h00001", n=20)
    donor = _make_track("dn0001", n=20, start_lat=47.0, start_lon=7.0, heading=200.0)
    rng = np.random.default_rng(10)
    track = generate_track_hijack(base, donor, rng, severity=500.0, variant_id="hj_00")

    idx = track.params["hijack_index"]
    assert all(a is False for a in track.is_attacked[:idx])
    assert all(a is True for a in track.is_attacked[idx:])
    assert all(t is None for t in track.true_state_vectors[idx:])
    assert all(sv.icao24 == base[0].icao24 for sv in track.broadcast_state_vectors)


def test_hijack_discontinuity_grows_with_severity():
    base = _make_track("h00002", n=20)
    donor = _make_track("dn0002", n=20, start_lat=47.0, start_lon=7.0, heading=200.0)

    from absproj.geo import latlon_to_enu

    def jump_distance(severity, seed):
        rng = np.random.default_rng(seed)
        track = generate_track_hijack(base, donor, rng, severity=severity, variant_id="hj")
        idx = track.params["hijack_index"]
        last_pre = track.broadcast_state_vectors[idx - 1]
        first_post = track.broadcast_state_vectors[idx]
        dx, dy, _ = latlon_to_enu(first_post.latitude, first_post.longitude, 0.0, last_pre.latitude, last_pre.longitude, 0.0)
        return math.hypot(dx, dy)

    small = jump_distance(0.0, seed=11)
    large = jump_distance(3000.0, seed=11)
    assert large > small + 2000.0  # the extra discontinuity should dominate


# --- replay ---

def test_replay_pre_and_post_labels():
    live = _make_track("r00001", n=20)
    source = _make_track("s00001", n=20, start_lat=46.0, start_lon=6.0)
    rng = np.random.default_rng(12)
    track = generate_replay(live, source, rng, severity=600.0, variant_id="rp_00")

    idx = track.params["splice_index"]
    assert all(a is False for a in track.is_attacked[:idx])
    assert all(a is True for a in track.is_attacked[idx:])
    assert all(t is None for t in track.true_state_vectors[idx:])
    assert all(sv.icao24 == live[0].icao24 for sv in track.broadcast_state_vectors)


def test_replay_timestamps_continue_live_cadence():
    live = _make_track("r00002", n=20)
    source = _make_track("s00002", n=20, start_lat=46.0, start_lon=6.0)
    rng = np.random.default_rng(13)
    track = generate_replay(live, source, rng, severity=600.0, variant_id="rp_01")
    times = [sv.observed_at for sv in track.broadcast_state_vectors]
    assert times == sorted(times)
    assert times == [sv.observed_at for sv in live[:len(times)]]


# --- injector ---

def test_generate_variants_severity_span_and_count():
    config = _default_attacks_config()
    victim = _make_track("p00001", n=20)
    intruder = _make_track("p00002", n=20)
    from absproj.attacks.substrate import SubstratePool
    pool = SubstratePool(tracks={"p00001": victim, "p00002": intruder})
    rng = np.random.default_rng(14)

    variants = generate_variants(AttackClass.GHOST, config, pool, BBOX, rng, T0)
    assert len(variants) == config.variants_per_class
    severities = [v.severity for v in variants]
    assert severities == sorted(severities)
    assert math.isclose(severities[0], 0.0, abs_tol=1e-9)
    assert math.isclose(severities[-1], 1.0, abs_tol=1e-9)
    assert len({v.variant_id for v in variants}) == len(variants)


def test_generate_all_variants_covers_every_class():
    config = _default_attacks_config()
    tracks = {f"p{i:05d}": _make_track(f"p{i:05d}", n=20) for i in range(6)}
    from absproj.attacks.substrate import SubstratePool
    pool = SubstratePool(tracks=tracks)
    rng = np.random.default_rng(15)

    all_variants = generate_all_variants(config, pool, BBOX, rng, T0)
    assert set(all_variants.keys()) == set(AttackClass)
    for cls, variants in all_variants.items():
        assert len(variants) == config.variants_per_class
        assert all(v.attack_class == cls for v in variants)


# --- split ---

def test_holdout_class_entirely_in_test():
    config = _default_attacks_config()
    tracks = {f"p{i:05d}": _make_track(f"p{i:05d}", n=20) for i in range(6)}
    from absproj.attacks.substrate import SubstratePool
    pool = SubstratePool(tracks=tracks)
    rng = np.random.default_rng(16)

    all_variants = generate_all_variants(config, pool, BBOX, rng, T0)
    split = build_evaluation_split(all_variants, holdout_class=AttackClass.TRACK_HIJACK, train_fraction=0.5)

    assert split.train[AttackClass.TRACK_HIJACK] == []
    assert len(split.test[AttackClass.TRACK_HIJACK]) == config.variants_per_class


def test_split_train_test_disjoint_and_complete():
    config = _default_attacks_config()
    tracks = {f"p{i:05d}": _make_track(f"p{i:05d}", n=20) for i in range(6)}
    from absproj.attacks.substrate import SubstratePool
    pool = SubstratePool(tracks=tracks)
    rng = np.random.default_rng(17)

    all_variants = generate_all_variants(config, pool, BBOX, rng, T0)
    split = build_evaluation_split(all_variants, holdout_class=AttackClass.TRACK_HIJACK, train_fraction=0.5)

    for cls in AttackClass:
        if cls == AttackClass.TRACK_HIJACK:
            continue
        train_ids = {v.variant_id for v in split.train[cls]}
        test_ids = {v.variant_id for v in split.test[cls]}
        all_ids = {v.variant_id for v in all_variants[cls]}
        assert train_ids.isdisjoint(test_ids)
        assert train_ids | test_ids == all_ids
        assert len(train_ids) == round(config.variants_per_class * 0.5)


def test_split_stratifies_by_severity_not_just_position():
    config = _default_attacks_config()
    tracks = {f"p{i:05d}": _make_track(f"p{i:05d}", n=20) for i in range(6)}
    from absproj.attacks.substrate import SubstratePool
    pool = SubstratePool(tracks=tracks)
    rng = np.random.default_rng(18)

    all_variants = generate_all_variants(config, pool, BBOX, rng, T0)
    split = build_evaluation_split(all_variants, holdout_class=AttackClass.TRACK_HIJACK, train_fraction=0.5)

    train_severities = sorted(v.severity for v in split.train[AttackClass.POSITION_DRIFT])
    test_severities = sorted(v.severity for v in split.test[AttackClass.POSITION_DRIFT])
    # Both splits should span close to the full configured range, not e.g.
    # train getting only the low end and test only the high end.
    full_range = config.severity_ranges["position_drift"].max - config.severity_ranges["position_drift"].min
    assert (train_severities[-1] - train_severities[0]) > 0.5 * full_range
    assert (test_severities[-1] - test_severities[0]) > 0.5 * full_range
