"""OpenSky Network REST client: OAuth2 client-credentials auth with anonymous
fallback, and defensive handling of rate limits / timeouts / malformed responses.

Per the engineering requirements, this module must never raise out of get_states()
for ordinary network/API failures -- callers (the poller) get None back and log,
they don't crash the ingestion loop.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

import requests

from absproj.config import OpenSkyConfig

logger = logging.getLogger(__name__)


class OpenSkyError(Exception):
    """Raised for conditions the caller should decide how to handle (rare)."""


@dataclass
class _Token:
    access_token: str
    expires_at: float  # epoch seconds


class OpenSkyClient:
    def __init__(self, config: OpenSkyConfig, session: Optional[requests.Session] = None):
        self.config = config
        self.session = session or requests.Session()
        self._token: Optional[_Token] = None

    def _fetch_token(self) -> Optional[_Token]:
        if not self.config.has_credentials:
            return None
        try:
            resp = self.session.post(
                self.config.token_url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.config.client_id,
                    "client_secret": self.config.client_secret,
                },
                timeout=self.config.request_timeout_seconds,
            )
            resp.raise_for_status()
            body = resp.json()
            return _Token(
                access_token=body["access_token"],
                expires_at=time.time() + float(body.get("expires_in", 1800)) - 30,
            )
        except (requests.RequestException, ValueError, KeyError) as exc:
            logger.warning("opensky_token_fetch_failed", extra={"error": str(exc)})
            return None

    def _auth_headers(self) -> dict:
        if not self.config.has_credentials:
            return {}
        if self._token is None or time.time() >= self._token.expires_at:
            self._token = self._fetch_token()
        if self._token is None:
            return {}
        return {"Authorization": f"Bearer {self._token.access_token}"}

    def get_states(self) -> Optional[dict[str, Any]]:
        """Fetch current state vectors within the configured bbox.

        Returns the parsed JSON body ({"time": ..., "states": [...]}), or None if
        the request failed after retries / the response was malformed. Never raises
        for ordinary HTTP/network/JSON errors.
        """
        params = {
            "lamin": self.config.bbox.lamin,
            "lamax": self.config.bbox.lamax,
            "lomin": self.config.bbox.lomin,
            "lomax": self.config.bbox.lomax,
            "extended": 1,
        }

        attempt = 0
        while attempt <= self.config.max_retries:
            attempt += 1
            try:
                headers = self._auth_headers()
                resp = self.session.get(
                    self.config.states_url,
                    params=params,
                    headers=headers,
                    timeout=self.config.request_timeout_seconds,
                )
            except requests.Timeout:
                logger.warning("opensky_request_timeout", extra={"attempt": attempt})
                self._sleep_backoff(attempt)
                continue
            except requests.RequestException as exc:
                logger.warning("opensky_request_error", extra={"attempt": attempt, "error": str(exc)})
                self._sleep_backoff(attempt)
                continue

            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                logger.warning(
                    "opensky_rate_limited",
                    extra={"attempt": attempt, "retry_after": retry_after},
                )
                self._sleep_backoff(attempt, override=float(retry_after) if retry_after else None)
                continue

            if resp.status_code == 401 and self.config.has_credentials:
                logger.warning("opensky_unauthorized_refreshing_token", extra={"attempt": attempt})
                self._token = None
                self._sleep_backoff(attempt)
                continue

            if 500 <= resp.status_code < 600:
                logger.warning(
                    "opensky_server_error",
                    extra={"attempt": attempt, "status_code": resp.status_code},
                )
                self._sleep_backoff(attempt)
                continue

            if resp.status_code != 200:
                logger.error(
                    "opensky_unexpected_status",
                    extra={"status_code": resp.status_code, "body": resp.text[:500]},
                )
                return None

            try:
                body = resp.json()
            except ValueError as exc:
                logger.error("opensky_malformed_json", extra={"error": str(exc)})
                return None

            if not isinstance(body, dict) or "states" not in body:
                logger.error("opensky_unexpected_schema", extra={"body_keys": list(body)[:10] if isinstance(body, dict) else type(body).__name__})
                return None

            return body

        logger.error("opensky_request_failed_after_retries", extra={"attempts": attempt})
        return None

    def _sleep_backoff(self, attempt: int, override: Optional[float] = None) -> None:
        delay = override if override is not None else self.config.retry_backoff_seconds * attempt
        time.sleep(min(delay, 60))
