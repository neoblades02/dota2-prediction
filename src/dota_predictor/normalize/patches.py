from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.engine import Connection

from dota_predictor.logging_config import get_logger
from dota_predictor.storage.models import patches, raw_ingest_objects
from dota_predictor.storage.raw_store import load_raw_payload
from dota_predictor.storage.upsert import dialect_insert

logger = get_logger("normalize.patches")


def _patch_id_for_version(conn: Connection, version: str) -> int | None:
    """Lowest patch_id carrying this version name, so the dotaconstants id beats a provisional Valve one."""
    return conn.execute(
        select(patches.c.patch_id).where(patches.c.version == version).order_by(patches.c.patch_id).limit(1)
    ).scalar_one_or_none()


def normalize_dotaconstants_patches(conn: Connection, patches_data: list[dict[str, Any]]) -> int:
    """Normalize dotaconstants patch.json list with integer IDs matching OpenDota match.patch."""
    count = 0
    for p in patches_data:
        p_id = p.get("id")
        if p_id is None:
            continue
        v_name = p.get("name")
        if _patch_id_for_version(conn, v_name) not in (None, p_id):
            continue  # Valve feed already created this version under a provisional id; never add a second row
        date_str = p.get("date")
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00")) if date_str else None

        stmt = (
            dialect_insert(conn, patches)
            .values(
                patch_id=p_id,
                version=v_name,
                published_at_utc=dt,
                effective_at_utc=dt,
                source="dotaconstants",
                confidence="confirmed",
            )
            .on_conflict_do_update(
                index_elements=["patch_id"],
                set_={
                    "version": v_name,
                    "published_at_utc": dt,
                    "effective_at_utc": dt,
                },
            )
        )
        conn.execute(stmt)
        count += 1
    logger.info("Normalized dotaconstants patches", count=count)
    return count


def normalize_valve_patches(conn: Connection) -> int:
    """Normalize raw Valve patch feed into patches table."""
    raw_row = (
        conn.execute(
            select(raw_ingest_objects)
            .where(
                raw_ingest_objects.c.provider == "valve_patches",
                raw_ingest_objects.c.endpoint == "/datafeed/patchnoteslist",
            )
            .order_by(raw_ingest_objects.c.fetched_at_utc.desc())
        )
        .mappings()
        .first()
    )

    if not raw_row:
        logger.warning("No raw Valve patch feed found to normalize")
        return 0

    payload = load_raw_payload(raw_row["payload_path"])
    patches_list = payload.get("patches", []) if isinstance(payload, dict) else []

    count = 0
    for idx, p in enumerate(patches_list):
        p_name = p.get("patch_name") or p.get("patch_number") or f"ValvePatch_{idx}"
        ts = p.get("patch_timestamp") or p.get("start_date")
        start_utc = datetime.fromtimestamp(ts, tz=UTC) if ts else None

        # First check if this version already exists in patches (e.g. from dotaconstants)
        existing = _patch_id_for_version(conn, p_name)

        if existing is not None:
            conn.execute(
                patches.update()
                .where(patches.c.patch_id == existing)
                .values(
                    effective_at_utc=start_utc or patches.c.effective_at_utc,
                    published_at_utc=start_utc or patches.c.published_at_utc,
                    source="valve_patches",
                )
            )
            count += 1
        else:
            # Assign an integer ID outside the dotaconstants range (1000 + idx)
            p_id = 1000 + idx
            stmt = (
                dialect_insert(conn, patches)
                .values(
                    patch_id=p_id,
                    version=p_name,
                    published_at_utc=start_utc,
                    effective_at_utc=start_utc,
                    source="valve_patches",
                    confidence="confirmed",
                )
                .on_conflict_do_update(
                    index_elements=["patch_id"],
                    set_={
                        "version": p_name,
                        "published_at_utc": start_utc,
                        "effective_at_utc": start_utc,
                    },
                )
            )
            conn.execute(stmt)
            count += 1

    logger.info("Normalized patches from Valve feed", count=count)
    return count


def resolve_patch_for_match(
    conn: Connection,
    match_time_utc: datetime | None,
    raw_patch_id: int | None = None,
) -> tuple[int | None, str]:
    """Resolve patch_id for a match.

    Time-based inference against the patch table is primary: it yields letter
    patches (7.41b) and is immune to provider ids that are not patch ids. The raw
    id is only a fallback when the match time is unknown.
    """
    if match_time_utc is not None:
        row = conn.execute(
            select(patches.c.patch_id)
            .where(patches.c.effective_at_utc <= match_time_utc)
            .order_by(desc(patches.c.effective_at_utc), patches.c.patch_id)
            .limit(1)
        ).scalar_one_or_none()
        if row is not None:
            return row, "inferred"

    if raw_patch_id is not None:
        exists = conn.execute(select(patches.c.patch_id).where(patches.c.patch_id == raw_patch_id)).scalar_one_or_none()
        if exists is not None:
            return exists, "confirmed"

    return None, "unknown"
