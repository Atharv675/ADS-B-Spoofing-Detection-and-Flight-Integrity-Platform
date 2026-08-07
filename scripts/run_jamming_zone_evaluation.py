"""Phase 8: runs the full detection pipeline (Kalman/NIS, ML, MLAT, radar)
against the live-collected Baltic/Kaliningrad jamming-zone traffic, and
against a comparable-duration control window sampled from our existing clean
Western-Europe traffic, and reports detection activity side by side.

This is explicitly a plausibility check, not a labeled evaluation: unlike
Phase 7's synthetic testbed, there is no per-row ground truth here for which
specific broadcasts are affected by interference (real GPS jamming degrades
position accuracy/availability in ways that aren't announced per-aircraft,
per-timestamp). What this script can honestly report is whether detection
*activity* -- the rate at which each method flags updates as anomalous --
is meaningfully higher in the documented interference window than in the
clean control, not precision/recall/F1 (those require labels we don't have).
Reporting F1 here would be fabricated rigor; reporting the activity-rate
comparison, clearly caveated, is the honest version of this check.

MLAT/radar use a Baltic-region-specific simulated network (see
config.yaml's jamming_zone.mlat_receivers/radar_site) for the jamming-zone
data, and the original Phase 4/5 Western-Europe network for the control data
-- each region's data is checked against sensors actually deployed for that
region, as a real system would be. The threshold *values* (88m/3775m) are
inherited unchanged from Phase 4/5's calibration rather than independently
recalibrated for the new Baltic geometry (we don't have enough Baltic clean
traffic to calibrate against without begging the question) -- so absolute
Baltic MLAT/radar rates carry more uncertainty than the original calibrated
Western-Europe numbers. NIS and ML are geometry-independent and don't have
this caveat.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import random
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from absproj.config import get_config  # noqa: E402
from absproj.logging_setup import configure_logging  # noqa: E402
from absproj.ml.features import build_feature_frame, feature_matrix  # noqa: E402
from absproj.ml.isolation_forest import IsolationForestDetector  # noqa: E402
from absproj.storage import repository  # noqa: E402
from absproj.storage.db import get_connection  # noqa: E402
from absproj.tracking.track_manager import KalmanTrackManager  # noqa: E402
from absproj.verification.mlat import MLATSimulator  # noqa: E402
from absproj.verification.radar import RadarSimulator  # noqa: E402

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "isolation_forest.joblib"
REPORT_PATH = Path(__file__).resolve().parents[1] / "reports" / "phase8_jamming_zone.json"


def _run_region(label, state_vectors, config, mlat_sim, radar_sim, ml_detector):
    manager = KalmanTrackManager(config.kalman)
    nis_flags, kalman_rows = [], []
    mlat_flags, radar_flags = [], []

    for sv in state_vectors:
        record = manager.process(sv)
        if record is not None:
            nis_flags.append(record.is_anomalous)
            kalman_rows.append({
                "orig_index": len(kalman_rows), "time": record.time, "icao24": record.icao24,
                "category": record.category.value, "dt_seconds": record.dt_seconds,
                "innovation_x": record.innovation[0], "innovation_y": record.innovation[1],
                "innovation_z": record.innovation[2], "nis": record.nis,
                "vx": record.vx, "vy": record.vy, "vz": record.vz,
            })
        mlat_flags.append(mlat_sim.check(sv).is_anomalous)
        radar_flags.append(radar_sim.check(sv).is_anomalous)

    ml_flags = []
    if kalman_rows:
        df = build_feature_frame(kalman_rows, config.ml)
        X = feature_matrix(df)
        ml_flags = list(ml_detector.predict_is_anomaly(X))

    n = len(state_vectors)
    result = {
        "region": label,
        "n_rows": n,
        "n_tracks": len({sv.icao24 for sv in state_vectors}),
        "nis_flag_rate": float(np.mean(nis_flags)) if nis_flags else None,
        "ml_flag_rate": float(np.mean(ml_flags)) if len(ml_flags) else None,
        "mlat_flag_rate": float(np.mean(mlat_flags)) if mlat_flags else None,
        "radar_flag_rate": float(np.mean(radar_flags)) if radar_flags else None,
        "n_kalman_updates": len(kalman_rows),
    }
    logger.info("region_evaluated", extra=result)
    return result


def main() -> None:
    config = get_config()
    configure_logging(config.logging.level, config.logging.format)
    jz = config.jamming_zone

    with get_connection(config.database) as conn:
        jamming_svs = list(repository.fetch_track_state_history_for_kalman(conn, source=jz.source_tag))
        if not jamming_svs:
            logger.error("no_jamming_zone_data_run_collect_jamming_zone_first")
            sys.exit(1)

        jamming_start = min(sv.observed_at for sv in jamming_svs)
        jamming_end = max(sv.observed_at for sv in jamming_svs)

        control_bounds_start, control_bounds_end = repository.fetch_track_state_time_bounds(conn, source="opensky")
        window = timedelta(minutes=jz.control_window_minutes)
        latest_possible_start = control_bounds_end - window
        span_seconds = max(0.0, (latest_possible_start - control_bounds_start).total_seconds())
        rng = random.Random(jz.control_random_seed)
        control_start = control_bounds_start + timedelta(seconds=rng.uniform(0, span_seconds))
        control_end = control_start + window

        control_svs = list(repository.fetch_track_state_history_for_kalman(
            conn, source="opensky", time_range=(control_start, control_end),
        ))

    logger.info(
        "windows_selected",
        extra={
            "jamming_zone_window": [jamming_start.isoformat(), jamming_end.isoformat()],
            "jamming_zone_rows": len(jamming_svs),
            "control_window": [control_start.isoformat(), control_end.isoformat()],
            "control_rows": len(control_svs),
        },
    )

    ml_detector = IsolationForestDetector(config.ml.isolation_forest)
    ml_detector.load(MODEL_PATH)

    jz_origin_lat = (jz.bbox.lamin + jz.bbox.lamax) / 2.0
    jz_origin_lon = (jz.bbox.lomin + jz.bbox.lomax) / 2.0
    jz_mlat_config = dataclasses.replace(config.mlat, receivers=jz.mlat_receivers)
    jz_mlat_sim = MLATSimulator(jz_mlat_config, jz_origin_lat, jz_origin_lon)
    jz_radar_config = dataclasses.replace(config.radar, site=jz.radar_site)
    jz_radar_sim = RadarSimulator(jz_radar_config)

    control_origin_lat = (config.opensky.bbox.lamin + config.opensky.bbox.lamax) / 2.0
    control_origin_lon = (config.opensky.bbox.lomin + config.opensky.bbox.lomax) / 2.0
    control_mlat_sim = MLATSimulator(config.mlat, control_origin_lat, control_origin_lon)
    control_radar_sim = RadarSimulator(config.radar)

    jamming_result = _run_region("baltic_jamming_zone", jamming_svs, config, jz_mlat_sim, jz_radar_sim, ml_detector)
    control_result = _run_region("clean_control_western_europe", control_svs, config, control_mlat_sim, control_radar_sim, ml_detector)

    comparison = {
        "jamming_zone": jamming_result,
        "control": control_result,
        "ratios": {
            method: (
                jamming_result[f"{method}_flag_rate"] / control_result[f"{method}_flag_rate"]
                if jamming_result.get(f"{method}_flag_rate") and control_result.get(f"{method}_flag_rate")
                else None
            )
            for method in ("nis", "ml", "mlat", "radar")
        },
        "caveat": (
            "Plausibility check only: no per-row ground truth exists for real traffic, so this reports "
            "detection ACTIVITY RATE (fraction of updates flagged), not precision/recall/F1. A higher "
            "rate in the jamming zone is consistent with real interference but is not proof of it -- "
            "regional traffic mix, receiver geometry, and other confounds are not controlled for."
        ),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    logger.info("jamming_zone_comparison_complete", extra={"path": str(REPORT_PATH), "ratios": comparison["ratios"]})


if __name__ == "__main__":
    main()
