from __future__ import annotations

import contextlib
import time
from datetime import UTC, datetime
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.engine import Connection
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from dota_predictor.logging_config import get_logger
from dota_predictor.settings import RateLimitPolicy, get_settings
from dota_predictor.storage.models import sync_cursors
from dota_predictor.storage.upsert import dialect_insert

logger = get_logger("ingest.base")


class RateLimitExhaustedError(Exception):
    """Raised when provider daily or strict quotas are exhausted."""


class RateLimitedClient:
    def __init__(self, provider: str, base_url: str | None = None, policy: RateLimitPolicy | None = None):
        self.provider = provider
        settings = get_settings()
        self.policy = policy or settings.rate_limit_for(provider)
        self.base_url = (base_url or "").rstrip("/")
        self.last_request_time = 0.0
        self.min_interval = 60.0 / max(1, self.policy.requests_per_minute)

        self.remaining_minute: int | None = None
        self.remaining_day: int | None = None

        headers = {"User-Agent": "dota-predictor/0.1 (research; data-engineering)"}
        if provider == "opendota" and settings.opendota_api_key:
            headers["Authorization"] = f"Bearer {settings.opendota_api_key}"
        if provider == "pandascore" and settings.pandascore_api_key:
            headers["Authorization"] = f"Bearer {settings.pandascore_api_key}"

        self.client = httpx.Client(
            headers=headers,
            timeout=self.policy.request_timeout_seconds,
            follow_redirects=True,
        )

    def _wait_for_rate_limit(self) -> None:
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            sleep_needed = self.min_interval - elapsed
            time.sleep(sleep_needed)

        # Respect header signals if present
        if self.remaining_minute is not None and self.remaining_minute <= 1:
            logger.warning("Approaching minute rate limit, sleeping 10s", remaining=self.remaining_minute)
            time.sleep(10.0)

        if self.remaining_day is not None and self.remaining_day <= 5:
            logger.error("Daily API quota virtually exhausted", remaining_day=self.remaining_day)
            raise RateLimitExhaustedError(f"Daily API quota exhausted for {self.provider}")

    def _update_limits(self, response: httpx.Response) -> None:
        self.last_request_time = time.time()
        for k, v in response.headers.items():
            kl = k.lower()
            if kl == "x-rate-limit-remaining-minute":
                with contextlib.suppress(ValueError):
                    self.remaining_minute = int(v)
            elif kl == "x-rate-limit-remaining-day":
                with contextlib.suppress(ValueError):
                    self.remaining_day = int(v)

    def get(self, endpoint: str, params: dict[str, Any] | None = None) -> httpx.Response:
        url = endpoint if endpoint.startswith("http") else f"{self.base_url}/{endpoint.lstrip('/')}"

        @retry(
            reraise=True,
            stop=stop_after_attempt(self.policy.max_retries),
            wait=wait_exponential(multiplier=self.policy.backoff_base_seconds, max=self.policy.backoff_max_seconds),
            retry=retry_if_exception_type((httpx.TransportError, httpx.TimeoutException)),
        )
        def _execute_req() -> httpx.Response:
            self._wait_for_rate_limit()
            resp = self.client.get(url, params=params)
            self._update_limits(resp)

            if resp.status_code == 429:
                retry_after = resp.headers.get("retry-after")
                delay = float(retry_after) if retry_after else 30.0
                logger.warning("HTTP 429 received, backing off", delay=delay, url=url)
                time.sleep(delay)
                resp.raise_for_status()

            return resp

        return _execute_req()

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> RateLimitedClient:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


def get_sync_cursor(conn: Connection, provider: str, resource: str) -> dict[str, Any] | None:
    row = (
        conn.execute(
            select(sync_cursors).where(
                sync_cursors.c.provider == provider,
                sync_cursors.c.resource == resource,
            )
        )
        .mappings()
        .first()
    )
    return dict(row) if row else None


def set_sync_cursor(
    conn: Connection,
    provider: str,
    resource: str,
    cursor: str | None,
    state: str = "idle",
    error: str | None = None,
    success: bool = True,
) -> None:
    now = datetime.now(UTC)
    stmt = dialect_insert(conn, sync_cursors).values(
        provider=provider,
        resource=resource,
        cursor=cursor,
        state=state,
        last_success_utc=now if success else None,
        retries=0 if success else sync_cursors.c.retries + 1,
        error=error,
    )
    # On conflict update
    update_dict = {
        "cursor": cursor,
        "state": state,
        "error": error,
    }
    if success:
        update_dict["last_success_utc"] = now
        update_dict["retries"] = 0

    stmt = stmt.on_conflict_do_update(
        index_elements=["provider", "resource"],
        set_=update_dict,
    )
    conn.execute(stmt)
