"""Polling loop: fetch -> normalize -> persist, on a fixed interval. Any failure
in a single cycle is logged and the loop continues on the next interval -- an
API outage or a bad response must never kill the ingestion process.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from absproj.config import AppConfig
from absproj.ingestion.normalize import normalize_batch
from absproj.ingestion.opensky_client import OpenSkyClient
from absproj.storage import repository
from absproj.storage.db import get_connection

logger = logging.getLogger(__name__)


def run_cycle(client: OpenSkyClient, config: AppConfig) -> bool:
    """Runs one fetch+persist cycle. Returns True if a batch was persisted."""
    body = client.get_states()
    if body is None:
        logger.warning("poll_cycle_no_data")
        return False

    states = normalize_batch(body)
    batch_time = datetime.fromtimestamp(body["time"], tz=timezone.utc)

    try:
        with get_connection(config.database) as conn:
            repository.insert_raw_message(conn, batch_time, body)
            tracks_upserted = repository.upsert_tracks(conn, states)
            states_inserted = repository.insert_track_states(conn, states)
    except Exception:
        logger.exception("poll_cycle_db_error")
        return False

    logger.info(
        "poll_cycle_complete",
        extra={
            "batch_time": batch_time.isoformat(),
            "raw_states": len(body.get("states") or []),
            "normalized_states": len(states),
            "tracks_upserted": tracks_upserted,
            "states_inserted": states_inserted,
        },
    )
    return True


def run_forever(config: AppConfig) -> None:
    client = OpenSkyClient(config.opensky)
    logger.info(
        "ingestion_starting",
        extra={
            "poll_interval_seconds": config.opensky.poll_interval_seconds,
            "authenticated": config.opensky.has_credentials,
            "bbox": config.opensky.bbox.__dict__,
        },
    )
    while True:
        start = time.monotonic()
        try:
            run_cycle(client, config)
        except Exception:
            logger.exception("poll_cycle_unhandled_error")
        elapsed = time.monotonic() - start
        sleep_for = max(0.0, config.opensky.poll_interval_seconds - elapsed)
        time.sleep(sleep_for)
