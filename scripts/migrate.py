"""Applies every .sql file in src/absproj/storage/migrations, in filename order.
Each file is expected to be idempotent (IF NOT EXISTS / ON CONFLICT), so this is
safe to re-run.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from absproj.config import get_config  # noqa: E402
from absproj.logging_setup import configure_logging  # noqa: E402
from absproj.storage.db import get_connection  # noqa: E402

MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "src" / "absproj" / "storage" / "migrations"

logger = logging.getLogger(__name__)


def main() -> None:
    config = get_config()
    configure_logging(config.logging.level, config.logging.format)

    sql_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    if not sql_files:
        logger.error("no_migrations_found", extra={"dir": str(MIGRATIONS_DIR)})
        sys.exit(1)

    with get_connection(config.database) as conn:
        for path in sql_files:
            logger.info("applying_migration", extra={"file": path.name})
            sql = path.read_text(encoding="utf-8")
            with conn.cursor() as cur:
                cur.execute(sql)
            conn.commit()
            logger.info("migration_applied", extra={"file": path.name})


if __name__ == "__main__":
    main()
