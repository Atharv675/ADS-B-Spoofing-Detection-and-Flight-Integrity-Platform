"""Batch-processes all accumulated track_state history through the per-track
Kalman filter + NIS test, writing results to kalman_updates and detections
(method='nis'). Reprocesses the full history each run (truncate + rebuild) --
simple and fully reproducible at this project's data volumes; a production
system would process incrementally instead.
"""
from __future__ import annotations

import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from absproj.config import get_config  # noqa: E402
from absproj.logging_setup import configure_logging  # noqa: E402
from absproj.storage import repository  # noqa: E402
from absproj.storage.db import get_connection  # noqa: E402
from absproj.tracking.track_manager import KalmanTrackManager  # noqa: E402

logger = logging.getLogger(__name__)

FLUSH_EVERY = 5000


def main() -> None:
    config = get_config()
    configure_logging(config.logging.level, config.logging.format)

    manager = KalmanTrackManager(config.kalman)

    nis_by_category: dict[str, list[float]] = defaultdict(list)
    anomalous_count = 0
    total_updates = 0

    with get_connection(config.database) as conn:
        repository.truncate_kalman_outputs(conn)

        buffer = []
        for sv in repository.fetch_track_state_history_for_kalman(conn):
            record = manager.process(sv)
            if record is None:
                continue

            total_updates += 1
            nis_by_category[record.category.value].append(record.nis)
            if record.is_anomalous:
                anomalous_count += 1

            buffer.append(record)
            if len(buffer) >= FLUSH_EVERY:
                repository.insert_kalman_updates(conn, buffer)
                repository.insert_nis_detections(conn, buffer)
                buffer = []

        if buffer:
            repository.insert_kalman_updates(conn, buffer)
            repository.insert_nis_detections(conn, buffer)

    logger.info(
        "kalman_batch_complete",
        extra={
            "tracks_initialized": manager.init_count,
            "track_resets": manager.reset_count,
            "updates_skipped": manager.skip_count,
            "updates_processed": total_updates,
            "anomalous_updates": anomalous_count,
            "anomalous_rate": (anomalous_count / total_updates) if total_updates else None,
        },
    )

    import numpy as np

    for category, values in sorted(nis_by_category.items()):
        arr = np.array(values)
        logger.info(
            "nis_summary_by_category",
            extra={
                "category": category,
                "count": len(arr),
                "mean_nis": float(arr.mean()),
                "median_nis": float(np.median(arr)),
                "p95_nis": float(np.percentile(arr, 95)),
                "max_nis": float(arr.max()),
            },
        )


if __name__ == "__main__":
    main()
