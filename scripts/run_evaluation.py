"""Phase 7: the full evaluation run.

1. Builds the same substrate pool + attack variants + held-out split as
   Phase 6 (scripts/verify_attacks.py), plus two pools of real clean tracks
   *not* used as attack substrate: one for fitting the evidence-fusion model
   (clean negative examples), one held out purely to measure the
   clean-traffic false-positive rate the brief asks be reported separately.
2. Runs every scenario (every attack variant, both clean pools) through
   every detection method (rule-based, NIS, ML, MLAT, radar).
3. Fits the evidence-fusion model on the train split only (non-holdout
   classes' train variants + the clean-fusion-training pool), then scores
   every scenario with it.
4. Aggregates precision/recall/F1/FPR/FNR/detection-latency per
   (attack class, split) x method, and reports clean-traffic FPR separately.
5. Writes the full result table to reports/phase7_benchmark.json (Phase 9
   reads this to build the written report) and prints a summary.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from absproj.attacks.injector import generate_all_variants  # noqa: E402
from absproj.attacks.split import build_evaluation_split  # noqa: E402
from absproj.attacks.substrate import build_substrate_pool  # noqa: E402
from absproj.attacks.types import AttackClass  # noqa: E402
from absproj.config import get_config  # noqa: E402
from absproj.evaluation.fusion import EvidenceFusion  # noqa: E402
from absproj.evaluation.metrics import compute_metrics, detection_latency_seconds  # noqa: E402
from absproj.evaluation.pipeline import METHODS, evaluate_scenario  # noqa: E402
from absproj.logging_setup import configure_logging  # noqa: E402
from absproj.ml.isolation_forest import IsolationForestDetector  # noqa: E402
from absproj.storage import repository  # noqa: E402
from absproj.storage.db import get_connection  # noqa: E402
from absproj.verification.mlat import MLATSimulator  # noqa: E402
from absproj.verification.radar import RadarSimulator  # noqa: E402

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "isolation_forest.joblib"
REPORT_PATH = Path(__file__).resolve().parents[1] / "reports" / "phase7_benchmark.json"
ALL_METHODS = list(METHODS) + ["fused"]


def _used_substrate_icao24s(all_variants) -> set[str]:
    used = set()
    for variants in all_variants.values():
        for v in variants:
            used.add(v.icao24)
            for key, value in v.params.items():
                if key.endswith("icao24") and isinstance(value, str):
                    used.add(value)
    return used


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--since", type=str, default=None,
        help="ISO timestamp (e.g. 2026-08-04T16:40:00+00:00). Restrict substrate/clean pools to "
             "track_state rows at or after this time -- e.g. to only use data collected after a "
             "schema fix, so newly-added fields (like callsign) are actually present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_config()
    configure_logging(config.logging.level, config.logging.format)

    time_range = None
    if args.since:
        since = datetime.fromisoformat(args.since)
        time_range = (since, datetime.now(timezone.utc))

    with get_connection(config.database) as conn:
        history = list(repository.fetch_track_state_history_for_kalman(conn, time_range=time_range))

    pool = build_substrate_pool(history, min_length=config.attacks.min_substrate_track_length)
    logger.info("substrate_pool_built", extra={"donor_tracks": len(pool.tracks)})

    attack_rng = np.random.default_rng(config.attacks.random_seed)
    base_start_time = datetime.now(timezone.utc)
    all_variants = generate_all_variants(config.attacks, pool, config.opensky.bbox, attack_rng, base_start_time)
    split = build_evaluation_split(
        all_variants,
        holdout_class=AttackClass(config.attacks.holdout_class),
        train_fraction=config.attacks.train_fraction,
    )

    used_icao24s = _used_substrate_icao24s(all_variants)
    available_clean = [icao24 for icao24 in pool.tracks if icao24 not in used_icao24s]
    eval_rng = np.random.default_rng(config.evaluation.random_seed)
    eval_rng.shuffle(available_clean)
    n_train = config.evaluation.clean_train_track_count
    n_test = config.evaluation.clean_test_track_count
    clean_train_icao24s = available_clean[:n_train]
    clean_test_icao24s = available_clean[n_train:n_train + n_test]
    logger.info(
        "clean_pools_built",
        extra={
            "available": len(available_clean),
            "clean_train": len(clean_train_icao24s),
            "clean_test": len(clean_test_icao24s),
        },
    )

    ml_detector = IsolationForestDetector(config.ml.isolation_forest)
    ml_detector.load(MODEL_PATH)

    origin_lat = (config.opensky.bbox.lamin + config.opensky.bbox.lamax) / 2.0
    origin_lon = (config.opensky.bbox.lomin + config.opensky.bbox.lomax) / 2.0
    mlat_sim = MLATSimulator(config.mlat, origin_lat, origin_lon)
    radar_sim = RadarSimulator(config.radar)

    # --- Step A: run every scenario through every non-fusion method ---
    results = []

    for attack_class in AttackClass:
        for scenario_split, variants in (("train", split.train[attack_class]), ("test", split.test[attack_class])):
            for v in variants:
                results.append(evaluate_scenario(
                    v.broadcast_state_vectors, v.true_state_vectors, v.is_attacked,
                    config, ml_detector, mlat_sim, radar_sim,
                    attack_class=attack_class.value, variant_id=v.variant_id, split=scenario_split,
                ))

    for scenario_split, icao24s in (("clean_train", clean_train_icao24s), ("clean_test", clean_test_icao24s)):
        for icao24 in icao24s:
            track = pool.tracks[icao24]
            results.append(evaluate_scenario(
                track, list(track), [False] * len(track),
                config, ml_detector, mlat_sim, radar_sim,
                attack_class=None, variant_id=icao24, split=scenario_split,
            ))

    logger.info("scenarios_evaluated", extra={"count": len(results)})

    # --- Step B: fit fusion on train-split rows only ---
    fusion_rows, fusion_labels = [], []
    clean_train_fusion_rows = []
    for r in results:
        if r.split not in ("train", "clean_train"):
            continue
        for row, attacked in zip(r.fusion_rows, r.is_attacked):
            if row is not None:
                fusion_rows.append(row)
                fusion_labels.append(attacked)
                if r.split == "clean_train":
                    clean_train_fusion_rows.append(row)

    fusion = EvidenceFusion(random_state=config.evaluation.fusion_random_state)
    fusion.fit(fusion_rows, fusion_labels)
    logger.info("fusion_model_fit", extra={"n_train_rows": len(fusion_rows), "n_attacked": sum(fusion_labels)})

    # Calibrate the decision threshold the same way MLAT/radar's thresholds
    # were calibrated in Phases 4-5: empirical 99th percentile of the
    # suspicion score on *clean_train* traffic (not clean_test -- that stays
    # held out so the FPR reported on it isn't circular). class_weight=
    # "balanced" during fitting (needed because attacks are a small minority
    # of train rows) pushes the default 0.5 cutoff towards over-flagging
    # clean traffic; this recalibrates it against real data instead of
    # leaving it at an arbitrary default.
    clean_train_suspicion = fusion.suspicion(clean_train_fusion_rows)
    calibrated_threshold = float(np.percentile(clean_train_suspicion, 99))
    fusion.decision_threshold = calibrated_threshold
    logger.info(
        "fusion_threshold_calibrated",
        extra={"threshold": calibrated_threshold, "clean_train_rows": len(clean_train_fusion_rows)},
    )

    # --- Step C: score every scenario with fusion ---
    for r in results:
        fused_pred = [False] * len(r.is_attacked)
        idx_with_signal = [i for i, row in enumerate(r.fusion_rows) if row is not None]
        if idx_with_signal:
            rows_with_signal = [r.fusion_rows[i] for i in idx_with_signal]
            flags = fusion.is_anomalous(rows_with_signal)
            for i, flag in zip(idx_with_signal, flags):
                fused_pred[i] = bool(flag)
        r.predictions["fused"] = fused_pred

    # --- Step D/E: aggregate metrics per (class, split) x method ---
    benchmark = []
    groups: dict[tuple[str, str], list] = {}
    for r in results:
        key = (r.attack_class or "clean", r.split)
        groups.setdefault(key, []).append(r)

    for (attack_class, scenario_split), scenario_list in sorted(groups.items()):
        for method in ALL_METHODS:
            y_true, y_pred = [], []
            latencies = []
            n_variants_with_attack = 0
            n_never_detected = 0
            for r in scenario_list:
                y_true.extend(r.is_attacked)
                y_pred.extend(r.predictions[method])
                if any(r.is_attacked):
                    n_variants_with_attack += 1
                    latency = detection_latency_seconds(r.times, r.is_attacked, r.predictions[method])
                    if latency is None:
                        n_never_detected += 1
                    else:
                        latencies.append(latency)

            m = compute_metrics(y_true, y_pred)
            benchmark.append({
                "attack_class": attack_class,
                "split": scenario_split,
                "method": method,
                "precision": m.precision, "recall": m.recall, "f1": m.f1,
                "fpr": m.fpr, "fnr": m.fnr,
                "n_rows": m.n, "n_positive": m.n_positive, "n_predicted_positive": m.n_predicted_positive,
                "n_scenarios": len(scenario_list),
                "n_scenarios_with_attack": n_variants_with_attack,
                "n_never_detected": n_never_detected,
                "mean_detection_latency_s": float(np.mean(latencies)) if latencies else None,
            })

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, indent=2)
    logger.info("benchmark_written", extra={"path": str(REPORT_PATH), "rows": len(benchmark)})

    logger.info("== Benchmark summary (test splits + clean_test only) ==")
    for row in benchmark:
        if row["split"] not in ("test", "clean_test"):
            continue
        logger.info(
            "benchmark_row",
            extra={k: v for k, v in row.items() if k not in ("n_scenarios", "n_scenarios_with_attack")},
        )


if __name__ == "__main__":
    main()
