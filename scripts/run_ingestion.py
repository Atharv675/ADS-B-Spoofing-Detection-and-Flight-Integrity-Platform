"""Entry point for the ingestion service: configure logging, load config, poll
OpenSky forever. Runs both bare (python scripts/run_ingestion.py) and inside the
`ingestion` Docker Compose service.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from absproj.config import get_config  # noqa: E402
from absproj.ingestion.poller import run_forever  # noqa: E402
from absproj.logging_setup import configure_logging  # noqa: E402


def main() -> None:
    config = get_config()
    configure_logging(config.logging.level, config.logging.format)
    run_forever(config)


if __name__ == "__main__":
    main()
