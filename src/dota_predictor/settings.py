from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def normalize_database_url(url: str) -> str:
    """Point plain Postgres URLs (as Dokploy hands them out) at the installed psycopg 3 driver."""
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix) :]
    return url


class RateLimitPolicy(BaseSettings):
    """Per-source request policy. Requests are spaced at requests_per_minute."""

    requests_per_minute: int = 55
    burst: int = 1
    max_retries: int = 6
    backoff_base_seconds: float = 1.0
    backoff_max_seconds: float = 120.0
    request_timeout_seconds: float = 30.0
    respect_retry_after: bool = True


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DOTA_", env_file=".env", extra="ignore")

    database_url: str = Field(
        default=f"sqlite:///{(PROJECT_ROOT / 'data' / 'dota_predictor.db').as_posix()}",
        validation_alias=AliasChoices("DOTA_DATABASE_URL", "DATABASE_URL"),
    )
    raw_store_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    features_dir: Path = PROJECT_ROOT / "data" / "features"
    reports_dir: Path = PROJECT_ROOT / "data" / "reports"
    models_dir: Path = PROJECT_ROOT / "models"

    opendota_api_key: str | None = Field(default=None, alias="DOTA_OPENDOTA_API_KEY")
    pandascore_api_key: str | None = Field(default=None, alias="DOTA_PANDASCORE_API_KEY")
    stratz_api_key: str | None = Field(default=None, alias="DOTA_STRATZ_API_KEY")
    ui_password: str | None = Field(default=None, alias="DOTA_UI_PASSWORD")

    @field_validator("database_url")
    @classmethod
    def _psycopg_driver(cls, v: str) -> str:
        return normalize_database_url(v)

    def providers_config(self) -> dict:
        path = PROJECT_ROOT / "config" / "providers.yaml"
        if not path.exists():
            return {}
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    def rate_limit_for(self, provider: str) -> RateLimitPolicy:
        cfg = self.providers_config().get(provider, {})
        rl = cfg.get("rate_limit", {})
        return RateLimitPolicy(**{k: v for k, v in rl.items() if k in RateLimitPolicy.model_fields})


@lru_cache
def get_settings() -> Settings:
    return Settings()
