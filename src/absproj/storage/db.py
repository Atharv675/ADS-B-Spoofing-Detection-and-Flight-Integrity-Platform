"""Thin psycopg2 connection helper. No ORM -- SQL stays visible in repository.py
and the migration files."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import psycopg2
import psycopg2.extensions

from absproj.config import DatabaseConfig


@contextmanager
def get_connection(db_config: DatabaseConfig) -> Iterator[psycopg2.extensions.connection]:
    conn = psycopg2.connect(db_config.dsn)
    try:
        yield conn
    finally:
        conn.close()
