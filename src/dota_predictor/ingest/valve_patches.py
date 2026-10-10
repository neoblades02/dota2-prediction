from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy.engine import Engine

from dota_predictor.ingest.base import RateLimitedClient
from dota_predictor.logging_config import get_logger
from dota_predictor.storage.raw_store import save_raw_payload

logger = get_logger("ingest.valve_patches")


def sync_valve_patches(engine: Engine) -> dict[str, Any]:
    """Ingest official Valve patch feed and dotaconstants patch mappings."""
    provider = "valve_patches"
    client = RateLimitedClient(provider=provider, base_url="https://www.dota2.com")
    endpoint = "/datafeed/patchnoteslist?language=english"

    try:
        logger.info("Fetching Valve patch notes list")
        resp = client.get(endpoint)
        if resp.status_code != 200:
            logger.error("Failed to fetch Valve patches", status=resp.status_code)
            return {"status": "error", "code": resp.status_code}

        payload = resp.json()
        patches_list = payload.get("patches", [])
        now = datetime.now(UTC)

        with engine.begin() as conn:
            raw_id = save_raw_payload(
                conn=conn,
                provider=provider,
                endpoint="/datafeed/patchnoteslist",
                request_key="english",
                payload=payload,
                http_status=resp.status_code,
                schema_version="v1",
                fetched_at_utc=now,
            )

        # Also fetch dotaconstants patch mapping
        try:
            dc_resp = client.client.get("https://raw.githubusercontent.com/odota/dotaconstants/master/build/patch.json")
            if dc_resp.status_code == 200:
                dc_payload = dc_resp.json()
                with engine.begin() as conn:
                    save_raw_payload(
                        conn=conn,
                        provider="opendota",
                        endpoint="/dotaconstants/patch",
                        request_key="all",
                        payload=dc_payload,
                        http_status=200,
                        schema_version="v1",
                        fetched_at_utc=now,
                    )
        except Exception as e:
            logger.warning("Could not fetch dotaconstants patch.json", error=str(e))

        logger.info("Saved Valve patch feed", raw_id=raw_id, patches_count=len(patches_list))
        return {
            "status": "ok",
            "patches_count": len(patches_list),
            "raw_id": raw_id,
        }
    finally:
        client.close()
