"""Phase 8: bounded live collection over the Kaliningrad/Baltic corridor --
see config.yaml's jamming_zone section for why this region and why live
collection rather than a historical archive.

Reuses the exact same OpenSkyClient/normalize_batch/repository machinery as
the main ingestion poller (scripts/run_ingestion.py) -- the only difference
is a different bounding box, a bounded run time instead of forever, and
tagging every row with source='baltic_jamming_zone' so it never mixes with
the main clean-traffic pipeline's data even though it lives in the same
table.
"""
from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from absproj.config import get_config  # noqa: E402
from absproj.ingestion.normalize import normalize_batch  # noqa: E402
from absproj.ingestion.opensky_client import OpenSkyClient  # noqa: E402
from absproj.logging_setup import configure_logging  # noqa: E402
from absproj.storage import repository  # noqa: E402
from absproj.storage.db import get_connection  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> None:
    config = get_config()
    configure_logging(config.logging.level, config.logging.format)

    jz = config.jamming_zone
    opensky_config = config.opensky
    opensky_config = type(opensky_config)(**{**opensky_config.__dict__, "bbox": jz.bbox, "poll_interval_seconds": jz.poll_interval_seconds})

    client = OpenSkyClient(opensky_config)

    logger.info(
        "jamming_zone_collection_starting",
        extra={"bbox": jz.bbox.__dict__, "collection_minutes": jz.collection_minutes, "source_tag": jz.source_tag},
    )

    deadline = time.monotonic() + jz.collection_minutes * 60.0
    cycles = 0
    total_states = 0

    while time.monotonic() < deadline:
        cycle_start = time.monotonic()
        body = client.get_states()
        if body is not None:
            states = normalize_batch(body)
            batch_time = datetime.fromtimestamp(body["time"], tz=timezone.utc)
            try:
                with get_connection(config.database) as conn:
                    repository.insert_raw_message(conn, batch_time, body)
                    repository.upsert_tracks(conn, states)
                    inserted = repository.insert_track_states(conn, states, source=jz.source_tag)
                total_states += inserted
                cycles += 1
                logger.info(
                    "jamming_zone_cycle_complete",
                    extra={"cycle": cycles, "states_inserted": inserted, "batch_time": batch_time.isoformat()},
                )
            except Exception:
                logger.exception("jamming_zone_cycle_db_error")
        else:
            logger.warning("jamming_zone_cycle_no_data")

        elapsed = time.monotonic() - cycle_start
        time.sleep(max(0.0, jz.poll_interval_seconds - elapsed))

    logger.info("jamming_zone_collection_complete", extra={"cycles": cycles, "total_states_inserted": total_states})


if __name__ == "__main__":
    main()
