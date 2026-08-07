"""Batch-runs the SIMULATED primary radar check (see verification/radar.py for
what that does and does not mean) over all accumulated airborne track_state
history, writing results to radar_checks and detections (method='radar').
Reprocesses the full history each run, same rationale as run_kalman.py. Unlike
MLAT, this has no iterative solver (closed-form range/azimuth <-> position),
so it's fast even without a --limit flag.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from absproj.config import get_config  # noqa: E402
from absproj.logging_setup import configure_logging  # noqa: E402
from absproj.storage import repository  # noqa: E402
from absproj.storage.db import get_connection  # noqa: E402
from absproj.verification.radar import RadarSimulator  # noqa: E402

logger = logging.getLogger(__name__)

FLUSH_EVERY = 5000


def main() -> None:
    config = get_config()
    configure_logging(config.logging.level, config.logging.format)

    simulator = RadarSimulator(config.radar)

    logger.info(
        "radar_batch_starting",
        extra={
            "site": {"lat": config.radar.site.lat, "lon": config.radar.site.lon},
            "range_noise_std_m": config.radar.range_noise_std_m,
            "azimuth_noise_std_deg": config.radar.azimuth_noise_std_deg,
            "disagreement_threshold_m": config.radar.disagreement_threshold_m,
        },
    )

    disagreements = []
    anomalous_count = 0
    total = 0

    with get_connection(config.database) as conn:
        repository.truncate_radar_outputs(conn)

        buffer = []
        for sv in repository.fetch_track_state_history_for_kalman(conn):
            result = simulator.check(sv)
            total += 1
            disagreements.append(result.disagreement_m)
            if result.is_anomalous:
                anomalous_count += 1

            buffer.append(result)
            if len(buffer) >= FLUSH_EVERY:
                repository.insert_radar_checks(conn, buffer)
                repository.insert_radar_detections(conn, buffer)
                buffer = []

        if buffer:
            repository.insert_radar_checks(conn, buffer)
            repository.insert_radar_detections(conn, buffer)

    arr = np.array(disagreements)
    logger.info(
        "radar_batch_complete",
        extra={
            "updates_processed": total,
            "anomalous_updates": anomalous_count,
            "anomalous_rate": (anomalous_count / total) if total else None,
            "mean_disagreement_m": float(arr.mean()) if total else None,
            "median_disagreement_m": float(np.median(arr)) if total else None,
            "p95_disagreement_m": float(np.percentile(arr, 95)) if total else None,
            "p99_disagreement_m": float(np.percentile(arr, 99)) if total else None,
            "max_disagreement_m": float(arr.max()) if total else None,
        },
    )


if __name__ == "__main__":
    main()
