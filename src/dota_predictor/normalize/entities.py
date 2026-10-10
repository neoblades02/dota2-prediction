from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.engine import Connection

from dota_predictor.logging_config import get_logger
from dota_predictor.storage.models import (
    hero_reference,
    player_aliases,
    players,
    team_aliases,
    teams,
    tournaments,
)
from dota_predictor.storage.upsert import dialect_insert

logger = get_logger("normalize.entities")


def get_or_create_team(
    conn: Connection,
    provider_team_id: int | str,
    name: str | None = None,
    tag: str | None = None,
    provider: str = "opendota",
    observed_at_utc: datetime | None = None,
) -> int:
    """Resolve provider team ID to canonical team_id, creating records if absent."""
    now = observed_at_utc or datetime.now(UTC)
    p_tid_str = str(provider_team_id)

    # 1. Check existing alias
    row = conn.execute(
        select(team_aliases.c.team_id).where(
            team_aliases.c.provider == provider,
            team_aliases.c.provider_team_id == p_tid_str,
        )
    ).scalar_one_or_none()

    if row is not None:
        # Update last_seen
        conn.execute(
            team_aliases.update()
            .where(
                team_aliases.c.provider == provider,
                team_aliases.c.provider_team_id == p_tid_str,
            )
            .values(last_seen_utc=now)
        )
        return row

    # 2. Check if team_id matches integer id directly in teams
    canonical_id = int(provider_team_id) if str(provider_team_id).isdigit() else None
    existing_team = None
    if canonical_id is not None:
        existing_team = conn.execute(
            select(teams.c.team_id).where(teams.c.team_id == canonical_id)
        ).scalar_one_or_none()

    if existing_team is None:
        canonical_name = name or f"Team_{p_tid_str}"
        if canonical_id is not None:
            res = conn.execute(
                dialect_insert(conn, teams)
                .values(
                    team_id=canonical_id,
                    canonical_name=canonical_name,
                    tag=tag,
                    created_utc=now,
                )
                .on_conflict_do_nothing()
            )
            final_id = canonical_id
        else:
            res = conn.execute(
                teams.insert().values(
                    canonical_name=canonical_name,
                    tag=tag,
                    created_utc=now,
                )
            )
            final_id = res.inserted_primary_key[0]
    else:
        final_id = existing_team

    # Record alias
    conn.execute(
        dialect_insert(conn, team_aliases)
        .values(
            team_id=final_id,
            provider=provider,
            provider_team_id=p_tid_str,
            name=name,
            first_seen_utc=now,
            last_seen_utc=now,
        )
        .on_conflict_do_update(
            index_elements=["provider", "provider_team_id"],
            set_={"last_seen_utc": now, "name": name or team_aliases.c.name},
        )
    )
    return final_id


def get_or_create_player(
    conn: Connection,
    account_id: int | None,
    name: str | None = None,
    personaname: str | None = None,
    provider: str = "opendota",
    observed_at_utc: datetime | None = None,
) -> int | None:
    """Resolve Steam32 account_id to canonical player_id, creating records if absent.

    Anonymous players (account_id 0 or None) return None.
    """
    if not account_id or account_id <= 0:
        return None

    now = observed_at_utc or datetime.now(UTC)
    display_name = name or personaname or f"Player_{account_id}"

    # 1. Check existing player by steam32_account_id
    existing = conn.execute(
        select(players.c.player_id).where(players.c.steam32_account_id == account_id)
    ).scalar_one_or_none()

    if existing is not None:
        player_id = existing
        # Update canonical name if better name available
        if name and not str(display_name).startswith("Player_"):
            conn.execute(players.update().where(players.c.player_id == player_id).values(canonical_name=display_name))
    else:
        conn.execute(
            dialect_insert(conn, players)
            .values(
                canonical_name=display_name,
                steam32_account_id=account_id,
                metadata_confidence="inferred",
                created_utc=now,
            )
            .on_conflict_do_nothing()
        )
        player_id = conn.execute(
            select(players.c.player_id).where(players.c.steam32_account_id == account_id)
        ).scalar_one()

    # Record alias
    conn.execute(
        dialect_insert(conn, player_aliases)
        .values(
            player_id=player_id,
            provider=provider,
            provider_player_id=str(account_id),
            alias=personaname or name,
            first_seen_utc=now,
            last_seen_utc=now,
        )
        .on_conflict_do_update(
            index_elements=["provider", "provider_player_id"],
            set_={"last_seen_utc": now, "alias": personaname or name or player_aliases.c.alias},
        )
    )
    return player_id


def normalize_leagues_payload(conn: Connection, leagues_data: list[dict[str, Any]]) -> int:
    """Upsert leagues into tournaments table."""
    count = 0
    for lg in leagues_data:
        lid = lg.get("leagueid")
        if not lid:
            continue
        stmt = (
            dialect_insert(conn, tournaments)
            .values(
                leagueid=lid,
                title=lg.get("name") or f"League {lid}",
                tier=lg.get("tier"),
                metadata_source="opendota",
            )
            .on_conflict_do_update(
                index_elements=["leagueid"],
                set_={
                    "title": lg.get("name") or tournaments.c.title,
                    "tier": lg.get("tier") or tournaments.c.tier,
                },
            )
        )
        conn.execute(stmt)
        count += 1
    return count


def normalize_heroes_payload(conn: Connection, heroes_data: list[dict[str, Any]]) -> int:
    """Upsert heroes into hero_reference table."""
    count = 0
    now = datetime.now(UTC)
    for h in heroes_data:
        hid = h.get("id")
        if not hid:
            continue
        stmt = (
            dialect_insert(conn, hero_reference)
            .values(
                hero_id=hid,
                name=h.get("name"),
                localized_name=h.get("localized_name"),
                first_seen_utc=now,
                last_seen_utc=now,
            )
            .on_conflict_do_update(
                index_elements=["hero_id"],
                set_={
                    "name": h.get("name") or hero_reference.c.name,
                    "localized_name": h.get("localized_name") or hero_reference.c.localized_name,
                    "last_seen_utc": now,
                },
            )
        )
        conn.execute(stmt)
        count += 1
    return count


def normalize_pro_players_payload(conn: Connection, players_data: list[dict[str, Any]]) -> int:
    """Upsert pro player profiles into players and player_aliases."""
    count = 0
    now = datetime.now(UTC)
    for p in players_data:
        acc_id = p.get("account_id")
        if not acc_id:
            continue
        name = p.get("name")
        personaname = p.get("personaname")
        get_or_create_player(
            conn=conn,
            account_id=acc_id,
            name=name,
            personaname=personaname,
            provider="opendota",
            observed_at_utc=now,
        )
        count += 1
    return count
