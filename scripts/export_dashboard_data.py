"""Exports a real-data snapshot for the interactive dashboard artifact.

Everything in the output JSON is produced by actually running this project's
real pipeline (KalmanTrackManager, IsolationForestDetector, MLATSimulator,
RadarSimulator, IdentityTracker, EvidenceFusion, and the real attack
generators in absproj/attacks/) against real, recently-ingested OpenSky data
pulled straight from the DB -- nothing in this file is a fabricated number or
a hand-tuned probability. Two sections:

- "live_tracks": real recent aircraft, run through evaluate_scenario() with
  true_svs == broadcast_svs (this is exactly what MLATSimulator.check() /
  RadarSimulator.check() do internally for real traffic with no separate
  ground truth -- see their docstrings), i.e. the same "no ground truth"
  real-traffic path Phases 4/5 used, not the synthetic ground-truth path.
- "attack_variants": a handful of real attack variants per class, built from
  real substrate tracks pulled from this same recent window, evaluated with
  the real check_with_ground_truth() path exactly like Phase 7, so the
  dashboard's "run attack" replay shows genuinely computed detector output
  for that specific run, not a probability roll.

Run from the project root: python scripts/export_dashboard_data.py
"""
from __future__ import annotations

import dataclasses
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from absproj.attacks.injector import generate_variants  # noqa: E402
from absproj.attacks.split import build_evaluation_split  # noqa: E402
from absproj.attacks.substrate import build_substrate_pool  # noqa: E402
from absproj.attacks.types import AttackClass  # noqa: E402
from absproj.config import get_config  # noqa: E402
from absproj.evaluation.fusion import EvidenceFusion  # noqa: E402
from absproj.evaluation.pipeline import evaluate_scenario  # noqa: E402
from absproj.ml.isolation_forest import IsolationForestDetector  # noqa: E402
from absproj.storage import repository  # noqa: E402
from absproj.storage.db import get_connection  # noqa: E402
from absproj.verification.mlat import MLATSimulator  # noqa: E402
from absproj.verification.radar import RadarSimulator  # noqa: E402

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "isolation_forest.joblib"
OUT_PATH = Path(__file__).resolve().parents[1] / "reports" / "dashboard_snapshot.json"

WINDOW_MINUTES = 90        # lookback for substrate/fusion-fitting pools -- enough distinct tracks to be stable
LIVE_TRACK_COUNT = 24       # how many of the freshest real tracks to ship for the live map
LIVE_TRACK_MIN_LEN = 6
VARIANTS_PER_CLASS = 5      # small, demo-sized -- Phase 7's real benchmark already used 20/class


def sv_point(sv, extra: dict) -> dict:
    p = {
        "t": sv.observed_at.isoformat(),
        "lat": sv.latitude,
        "lon": sv.longitude,
        "alt": sv.baro_altitude,
        "velocity": sv.velocity,
        "track_deg": sv.true_track,
        "callsign": (sv.callsign or "").strip() or None,
    }
    p.update(extra)
    return p


def export_scenario_points(broadcast_svs, true_svs, is_attacked, result) -> list[dict]:
    points = []
    for i, bsv in enumerate(broadcast_svs):
        tsv = true_svs[i]
        row = result.fusion_rows[i]
        points.append(sv_point(bsv, {
            "true_lat": tsv.latitude if tsv is not None else None,
            "true_lon": tsv.longitude if tsv is not None else None,
            "is_attacked": bool(is_attacked[i]),
            "has_signal": result.has_full_signal[i],
            "nis": row.nis if row else None,
            "ml_score": row.ml_score if row else None,
            "mlat_disagreement_m": (None if row is None or row.mlat_no_corroboration else row.mlat_disagreement_m),
            "mlat_no_corroboration": row.mlat_no_corroboration if row else None,
            "radar_disagreement_m": (None if row is None or row.radar_no_corroboration else row.radar_disagreement_m),
            "radar_no_corroboration": row.radar_no_corroboration if row else None,
            "flags": {m: result.predictions[m][i] for m in result.predictions},
        }))
    return points


def main() -> None:
    config = get_config()
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(minutes=WINDOW_MINUTES)

    print(f"[export] pulling real track_state history from {window_start.isoformat()} to {now.isoformat()}")
    with get_connection(config.database) as conn:
        history = list(repository.fetch_track_state_history_for_kalman(conn, time_range=(window_start, now)))
    print(f"[export] fetched {len(history)} real state-vector rows")

    pool = build_substrate_pool(history, min_length=LIVE_TRACK_MIN_LEN)
    print(f"[export] substrate pool: {len(pool.tracks)} real contiguous tracks (min_length={LIVE_TRACK_MIN_LEN})")

    ml_detector = IsolationForestDetector(config.ml.isolation_forest)
    ml_detector.load(MODEL_PATH)

    origin_lat = (config.opensky.bbox.lamin + config.opensky.bbox.lamax) / 2.0
    origin_lon = (config.opensky.bbox.lomin + config.opensky.bbox.lomax) / 2.0
    mlat_sim = MLATSimulator(config.mlat, origin_lat, origin_lon)
    radar_sim = RadarSimulator(config.radar)

    # ---------------- attack variants (real ground-truth path) ----------------
    attacks_cfg = dataclasses.replace(config.attacks, variants_per_class=VARIANTS_PER_CLASS)
    rng = np.random.default_rng(config.attacks.random_seed + 1)
    all_variants = {
        ac: generate_variants(ac, attacks_cfg, pool, config.opensky.bbox, rng, now)
        for ac in AttackClass
    }
    split = build_evaluation_split(all_variants, holdout_class=AttackClass(config.attacks.holdout_class), train_fraction=0.5)

    used_icao24s: set[str] = set()
    for variants in all_variants.values():
        for v in variants:
            used_icao24s.add(v.icao24)
            for key, value in v.params.items():
                if key.endswith("icao24") and isinstance(value, str):
                    used_icao24s.add(value)

    scenario_results = {}
    variant_split_of: dict[str, str] = {}
    for ac in AttackClass:
        for v in split.train.get(ac, []):
            variant_split_of[v.variant_id] = "train"
        for v in split.test.get(ac, []):
            variant_split_of[v.variant_id] = "test"
        for v in split.train.get(ac, []) + split.test.get(ac, []):
            r = evaluate_scenario(
                v.broadcast_state_vectors, v.true_state_vectors, v.is_attacked,
                config, ml_detector, mlat_sim, radar_sim,
                attack_class=ac.value, variant_id=v.variant_id, split=variant_split_of[v.variant_id],
            )
            scenario_results[v.variant_id] = (v, r)
    print(f"[export] evaluated {len(scenario_results)} real attack-variant scenarios")

    # ---------------- clean pools for fitting + live map ----------------
    available_clean = [icao24 for icao24 in pool.tracks if icao24 not in used_icao24s]
    rng2 = np.random.default_rng(config.evaluation.random_seed)
    rng2.shuffle(available_clean)
    n_train = min(60, len(available_clean) // 2)
    clean_train_icao24s = available_clean[:n_train]
    clean_test_icao24s = available_clean[n_train:n_train + n_train]
    print(f"[export] clean pools: {len(clean_train_icao24s)} train / {len(clean_test_icao24s)} test (available={len(available_clean)})")

    clean_results = {}
    for icao24 in clean_train_icao24s + clean_test_icao24s:
        track = pool.tracks[icao24]
        r = evaluate_scenario(
            track, track, [False] * len(track),
            config, ml_detector, mlat_sim, radar_sim,
            attack_class=None, variant_id=icao24, split="clean",
        )
        clean_results[icao24] = (track, r)

    # ---------------- fit + calibrate fusion ----------------
    fusion_rows, fusion_labels, clean_train_rows = [], [], []
    for ac in AttackClass:
        for v in split.train.get(ac, []):
            _, r = scenario_results[v.variant_id]
            for row, attacked in zip(r.fusion_rows, r.is_attacked):
                if row is not None:
                    fusion_rows.append(row)
                    fusion_labels.append(attacked)
    for icao24 in clean_train_icao24s:
        _, r = clean_results[icao24]
        for row in r.fusion_rows:
            if row is not None:
                fusion_rows.append(row)
                fusion_labels.append(False)
                clean_train_rows.append(row)

    fusion = EvidenceFusion(random_state=config.evaluation.fusion_random_state)
    fusion.fit(fusion_rows, fusion_labels)
    calibrated_threshold = float(np.percentile(fusion.suspicion(clean_train_rows), 99)) if clean_train_rows else 0.5
    fusion.decision_threshold = calibrated_threshold
    print(f"[export] fusion fit on {len(fusion_rows)} rows ({sum(fusion_labels)} attacked); threshold={calibrated_threshold:.4f}")

    def score_fusion(result):
        fused_pred = [False] * len(result.is_attacked)
        fused_score = [None] * len(result.is_attacked)
        idx = [i for i, row in enumerate(result.fusion_rows) if row is not None]
        if idx:
            rows = [result.fusion_rows[i] for i in idx]
            probs = fusion.suspicion(rows)
            for i, p in zip(idx, probs):
                fused_pred[i] = bool(p > fusion.decision_threshold)
                fused_score[i] = float(p)
        result.predictions["fused"] = fused_pred
        return fused_score

    for _, r in scenario_results.values():
        r._fused_score = score_fusion(r)
    for _, r in clean_results.values():
        r._fused_score = score_fusion(r)

    # ---------------- build output: live tracks ----------------
    live_icao24s = sorted(
        clean_test_icao24s or list(pool.tracks.keys()),
        key=lambda k: pool.tracks[k][-1].observed_at, reverse=True,
    )[:LIVE_TRACK_COUNT]

    live_tracks_out = []
    for icao24 in live_icao24s:
        track, r = clean_results.get(icao24, (None, None))
        if track is None:
            track = pool.tracks[icao24]
            r = evaluate_scenario(track, track, [False] * len(track), config, ml_detector, mlat_sim, radar_sim,
                                   attack_class=None, variant_id=icao24, split="live")
            r._fused_score = score_fusion(r)
        points = []
        for i, sv in enumerate(track):
            row = r.fusion_rows[i]
            points.append(sv_point(sv, {
                "nis": row.nis if row else None,
                "ml_score": row.ml_score if row else None,
                "mlat_disagreement_m": (None if row is None or row.mlat_no_corroboration else row.mlat_disagreement_m),
                "radar_disagreement_m": (None if row is None or row.radar_no_corroboration else row.radar_disagreement_m),
                "fused_score": r._fused_score[i],
                "flags": {m: r.predictions[m][i] for m in r.predictions},
            }))
        live_tracks_out.append({
            "icao24": icao24,
            "category": track[0].category,
            "callsign": (track[-1].callsign or "").strip() or None,
            "n_points": len(track),
            "points": points,
        })
    print(f"[export] built {len(live_tracks_out)} live tracks for the map")

    # ---------------- build output: attack variants ----------------
    attack_variants_out = {ac.value: [] for ac in AttackClass}
    for ac in AttackClass:
        for v in sorted(split.train.get(ac, []) + split.test.get(ac, []), key=lambda v: v.severity):
            _, r = scenario_results[v.variant_id]
            points = export_scenario_points(v.broadcast_state_vectors, v.true_state_vectors, v.is_attacked, r)
            for i, sc in enumerate(r._fused_score):
                points[i]["fused_score"] = sc
                points[i]["flags"]["fused"] = r.predictions["fused"][i]
            attack_variants_out[ac.value].append({
                "variant_id": v.variant_id,
                "severity": v.severity,
                "split": "test" if v in split.test.get(ac, []) else "train",
                "n_points": len(v.broadcast_state_vectors),
                "n_attacked_points": sum(v.is_attacked),
                "points": points,
            })

    out = {
        "generated_at": now.isoformat(),
        "window_minutes": WINDOW_MINUTES,
        "note": "live_tracks: real recent OpenSky traffic run through the real pipeline (no separate ground truth, "
                "same path as MLATSimulator.check()/RadarSimulator.check()). attack_variants: real attack generators "
                "applied to real substrate tracks from this same window, evaluated with the real ground-truth-aware "
                "pipeline (evaluate_scenario), same methodology as reports/phase7_benchmark.json but at demo scale "
                f"({VARIANTS_PER_CLASS} variants/class instead of {config.attacks.variants_per_class}).",
        "fusion_decision_threshold": calibrated_threshold,
        "live_tracks": live_tracks_out,
        "attack_variants": attack_variants_out,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f)
    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"[export] wrote {OUT_PATH} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
