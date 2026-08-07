"""Batch-runs the SIMULATED MLAT check (see verification/mlat.py for what that
does and does not mean) over all accumulated airborne track_state history,
writing results to mlat_checks and detections (method='mlat'). Reprocesses the
full history each run, same rationale as run_kalman.py.
"""
from __future__ import annotations

import argparse
import itertools
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from absproj.config import get_config  # noqa: E402
from absproj.logging_setup import configure_logging  # noqa: E402
from absproj.storage import repository  # noqa: E402
from absproj.storage.db import get_connection  # noqa: E402
from absproj.verification.mlat import MLATSimulator  # noqa: E402

logger = logging.getLogger(__name__)

FLUSH_EVERY = 5000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N airborne track_state rows (by icao24,time order). "
             "Default: all accumulated history.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_config()
    configure_logging(config.logging.level, config.logging.format)

    origin_lat = (config.opensky.bbox.lamin + config.opensky.bbox.lamax) / 2.0
    origin_lon = (config.opensky.bbox.lomin + config.opensky.bbox.lomax) / 2.0
    simulator = MLATSimulator(config.mlat, origin_lat, origin_lon)

    logger.info(
        "mlat_batch_starting",
        extra={
            "receiver_count": len(simulator.receivers),
            "reference_receiver": config.mlat.receivers[config.mlat.reference_receiver_index].name,
            "timing_noise_std_ns": config.mlat.timing_noise_std_ns,
            "disagreement_threshold_m": config.mlat.disagreement_threshold_m,
            "limit": args.limit,
        },
    )

    disagreements = []
    anomalous_count = 0
    total = 0

    with get_connection(config.database) as conn:
        repository.truncate_mlat_outputs(conn)

        history = repository.fetch_track_state_history_for_kalman(conn)
        if args.limit is not None:
            history = itertools.islice(history, args.limit)

        buffer = []
        for sv in history:
            result = simulator.check(sv)
            total += 1
            disagreements.append(result.disagreement_m)
            if result.is_anomalous:
                anomalous_count += 1

            buffer.append(result)
            if len(buffer) >= FLUSH_EVERY:
                repository.insert_mlat_checks(conn, buffer)
                repository.insert_mlat_detections(conn, buffer)
                buffer = []

        if buffer:
            repository.insert_mlat_checks(conn, buffer)
            repository.insert_mlat_detections(conn, buffer)

    arr = np.array(disagreements)
    logger.info(
        "mlat_batch_complete",
        extra={
            "updates_processed": total,
            "anomalous_updates": anomalous_count,
            "anomalous_rate": (anomalous_count / total) if total else None,
            "mean_disagreement_m": float(arr.mean()) if total else None,
            "median_disagreement_m": float(np.median(arr)) if total else None,
            "p95_disagreement_m": float(np.percentile(arr, 95)) if total else None,
            "max_disagreement_m": float(arr.max()) if total else None,
        },
    )


if __name__ == "__main__":
    main()
