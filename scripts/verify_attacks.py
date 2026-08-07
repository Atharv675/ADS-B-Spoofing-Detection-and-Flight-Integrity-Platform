"""Phase 6 verification: builds the substrate pool from real accumulated
traffic, generates all five attack classes' variants plus the held-out
train/test split, and runs a qualitative sanity check -- for one
representative variant per class, feed its broadcast stream through a fresh
Kalman/NIS pipeline (exactly as a real detector would, no access to ground
truth) and, where the attack has a true/broadcast split, through MLAT's
check_with_ground_truth -- and report whether NIS/MLAT actually move in the
expected direction after the attack starts.

This is a sanity check, not the evaluation. Full precision/recall/F1 across
every variant and every detector, using the held-out split properly, is
Phase 7's job -- this script's purpose is to catch a broken generator or a
wiring mistake before that much larger effort is built on top of it.
"""
from __future__ import annotations

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
from absproj.logging_setup import configure_logging  # noqa: E402
from absproj.storage import repository  # noqa: E402
from absproj.storage.db import get_connection  # noqa: E402
from absproj.tracking.track_manager import KalmanTrackManager  # noqa: E402
from absproj.verification.mlat import MLATSimulator  # noqa: E402

logger = logging.getLogger(__name__)


def summarize_variants(config, split) -> None:
    logger.info("== Variant generation summary ==")
    for attack_class in AttackClass:
        train_n = len(split.train[attack_class])
        test_n = len(split.test[attack_class])
        all_variants = split.train[attack_class] + split.test[attack_class]
        severities = [v.severity for v in all_variants]
        logger.info(
            "class_summary",
            extra={
                "class": attack_class.value,
                "train": train_n,
                "test": test_n,
                "is_holdout": attack_class == split.holdout_class,
                "severity_min": min(severities) if severities else None,
                "severity_max": max(severities) if severities else None,
            },
        )


def sanity_check_variant(mlat_sim: MLATSimulator, kalman_config, variant) -> None:
    manager = KalmanTrackManager(kalman_config)

    nis_pre, nis_post = [], []
    disagreement_pre, disagreement_post = [], []
    no_corroboration_post = 0
    post_count = 0

    for broadcast_sv, true_sv, attacked in zip(
        variant.broadcast_state_vectors, variant.true_state_vectors, variant.is_attacked
    ):
        record = manager.process(broadcast_sv)
        if record is not None:
            (nis_post if attacked else nis_pre).append(record.nis)

        mlat_result = mlat_sim.check_with_ground_truth(broadcast_sv=broadcast_sv, true_sv=true_sv)
        if attacked:
            post_count += 1
            if mlat_result.no_corroboration:
                no_corroboration_post += 1
            else:
                disagreement_post.append(mlat_result.disagreement_m)
        else:
            if not mlat_result.no_corroboration:
                disagreement_pre.append(mlat_result.disagreement_m)

    logger.info(
        "variant_sanity_check",
        extra={
            "class": variant.attack_class.value,
            "variant_id": variant.variant_id,
            "severity": variant.severity,
            "mean_nis_pre_attack": float(np.mean(nis_pre)) if nis_pre else None,
            "mean_nis_post_attack": float(np.mean(nis_post)) if nis_post else None,
            "mean_mlat_disagreement_pre_m": float(np.mean(disagreement_pre)) if disagreement_pre else None,
            "mean_mlat_disagreement_post_m": float(np.mean(disagreement_post)) if disagreement_post else None,
            "post_attack_rows": post_count,
            "post_attack_no_corroboration_rows": no_corroboration_post,
        },
    )


def main() -> None:
    config = get_config()
    configure_logging(config.logging.level, config.logging.format)

    with get_connection(config.database) as conn:
        history = list(repository.fetch_track_state_history_for_kalman(conn))

    pool = build_substrate_pool(history, min_length=config.attacks.min_substrate_track_length)
    logger.info("substrate_pool_built", extra={"donor_tracks": len(pool.tracks)})

    rng = np.random.default_rng(config.attacks.random_seed)
    base_start_time = datetime.now(timezone.utc)

    all_variants = generate_all_variants(config.attacks, pool, config.opensky.bbox, rng, base_start_time)
    split = build_evaluation_split(
        all_variants,
        holdout_class=AttackClass(config.attacks.holdout_class),
        train_fraction=config.attacks.train_fraction,
    )
    summarize_variants(config, split)

    origin_lat = (config.opensky.bbox.lamin + config.opensky.bbox.lamax) / 2.0
    origin_lon = (config.opensky.bbox.lomin + config.opensky.bbox.lomax) / 2.0
    mlat_sim = MLATSimulator(config.mlat, origin_lat, origin_lon)

    logger.info("== Per-class sanity check (one mid-severity variant each) ==")
    for attack_class in AttackClass:
        variants = all_variants[attack_class]
        mid = variants[len(variants) // 2]
        sanity_check_variant(mlat_sim, config.kalman, mid)


if __name__ == "__main__":
    main()
