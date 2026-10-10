from __future__ import annotations

import gzip
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import insert, select
from sqlalchemy.engine import Connection

from dota_predictor.logging_config import get_logger
from dota_predictor.settings import PROJECT_ROOT, get_settings
from dota_predictor.storage.models import raw_ingest_objects

logger = get_logger("storage.raw_store")


def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_raw_payload(
    conn: Connection,
    provider: str,
    endpoint: str,
    request_key: str,
    payload: bytes | str | dict | list,
    http_status: int = 200,
    schema_version: str = "v1",
    fetched_at_utc: datetime | None = None,
) -> int:
    """Save an immutable raw payload compressed as .json.gz and record in raw_ingest_objects.

    Returns the raw_object_id.
    """
    settings = get_settings()
    now = fetched_at_utc or datetime.now(UTC)

    if isinstance(payload, (dict, list)):
        payload_bytes = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
    elif isinstance(payload, str):
        payload_bytes = payload.encode("utf-8")
    elif isinstance(payload, bytes):
        payload_bytes = payload
    else:
        payload_bytes = str(payload).encode("utf-8")

    content_hash = compute_sha256(payload_bytes)

    # Clean endpoint name for file path
    clean_endpoint = endpoint.strip("/").replace("/", "_").replace("?", "_").replace("&", "_")
    target_dir = settings.raw_store_dir / provider / clean_endpoint
    target_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{content_hash}.json.gz"
    full_path = target_dir / filename
    try:
        rel_path = full_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        rel_path = full_path.as_posix()

    if not full_path.exists():
        with gzip.open(full_path, "wb") as f:
            f.write(payload_bytes)

    # Check if this object is already logged
    existing_id = conn.execute(
        select(raw_ingest_objects.c.id).where(
            raw_ingest_objects.c.provider == provider,
            raw_ingest_objects.c.endpoint == endpoint,
            raw_ingest_objects.c.request_key == request_key,
            raw_ingest_objects.c.content_hash == content_hash,
        )
    ).scalar_one_or_none()

    if existing_id is not None:
        return existing_id

    result = conn.execute(
        insert(raw_ingest_objects).values(
            provider=provider,
            endpoint=endpoint,
            request_key=request_key,
            fetched_at_utc=now,
            payload_path=rel_path,
            content_hash=content_hash,
            http_status=http_status,
            schema_version=schema_version,
        )
    )
    obj_id = result.inserted_primary_key[0]
    logger.debug(
        "Saved raw ingest object",
        id=obj_id,
        provider=provider,
        endpoint=endpoint,
        request_key=request_key,
        hash=content_hash,
    )
    return obj_id


def load_raw_payload(payload_path: str | Path) -> Any:
    """Read and decompress raw payload from storage."""
    path = Path(payload_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    with gzip.open(path, "rb") as f:
        decompressed = f.read()

    try:
        return json.loads(decompressed.decode("utf-8"))
    except Exception:
        return decompressed.decode("utf-8", errors="replace")
