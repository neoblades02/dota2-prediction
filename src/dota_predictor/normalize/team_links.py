"""Link PandaScore teams to our (OpenDota/Valve) team ids.

Only deterministic matches link automatically. Fuzzy matches are stored as
suggestions for a human to confirm, because look-alike names are common in
lower tiers ("old blood" vs "Cold BIood" are different teams).
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func, select, union_all, update
from sqlalchemy.engine import Connection

from dota_predictor.features.asof import aware_utc
from dota_predictor.storage.models import fixtures, games, team_links, teams
from dota_predictor.storage.upsert import dialect_insert

AFFIXES = {"team", "gaming", "esports", "esport", "club"}
ACTIVE_WINDOW = timedelta(days=90)  # an org renamed under a new id keeps playing only on the new one


def norm(s: str | None) -> str:
    return re.sub(r"[\W_]+", " ", (s or "").casefold()).strip()


def core(s: str | None) -> str:
    return " ".join(w for w in norm(s).split() if w not in AFFIXES)


@dataclass
class TeamIndex:
    names: dict[str, list[int]] = field(default_factory=dict)
    tags: dict[str, list[int]] = field(default_factory=dict)
    cores: dict[str, list[int]] = field(default_factory=dict)
    games: dict[int, int] = field(default_factory=dict)
    last_played: dict[int, datetime] = field(default_factory=dict)
    display: dict[int, str] = field(default_factory=dict)

    @classmethod
    def from_conn(cls, conn: Connection) -> TeamIndex:
        sides = union_all(
            select(games.c.radiant_team_id.label("team_id"), games.c.start_utc),
            select(games.c.dire_team_id.label("team_id"), games.c.start_utc),
        ).subquery()
        idx = cls()
        for tid, n, last in conn.execute(
            select(sides.c.team_id, func.count(), func.max(sides.c.start_utc)).group_by(sides.c.team_id)
        ):
            idx.games[tid] = n
            if last is not None:
                idx.last_played[tid] = aware_utc(last)
        for tid, name, tag in conn.execute(select(teams.c.team_id, teams.c.canonical_name, teams.c.tag)):
            idx.display[tid] = name
            for key, bucket in ((norm(name), idx.names), (norm(tag), idx.tags), (core(name), idx.cores)):
                if key:
                    bucket.setdefault(key, []).append(tid)
        return idx

    def _played(self, ids: list[int]) -> list[int]:
        return sorted({i for i in ids if self.games.get(i, 0) > 0}, key=lambda i: -self.games[i])

    def _single(self, ids: list[int]) -> int | None:
        """One candidate, or the only one that played recently; None while it stays ambiguous."""
        if len(ids) == 1:
            return ids[0]
        cutoff = datetime.now(UTC) - ACTIVE_WINDOW
        active = [i for i in ids if self.last_played.get(i) and self.last_played[i] >= cutoff]
        return active[0] if len(active) == 1 else None

    def match(self, name: str | None, acronym: str | None = None) -> tuple[int | None, str | None, list[dict]]:
        """(team_id, method, suggestions). team_id is None unless the match is deterministic."""
        n, a = norm(name), norm(acronym)
        by_name = self._played(self.names.get(n, []))
        if by_name:  # several provider ids for one org: the active one, else the one with the most games
            return self._single(by_name) or by_name[0], "exact_name", []
        by_tag = self._single(self._played(self.tags.get(n, []) + self.tags.get(a, []) + self.names.get(a, [])))
        if by_tag:
            return by_tag, "exact_tag", []
        by_core = self._single(self._played(self.cores.get(core(name), []))) if core(name) else None
        if by_core:
            return by_core, "core_name", []
        return None, None, self.suggest(n)

    def suggest(self, key: str, n: int = 3) -> list[dict]:
        out: list[dict] = []
        for m in difflib.get_close_matches(key, [*self.names, *self.tags], n=10, cutoff=0.6):
            for tid in self._played(self.names.get(m, []) + self.tags.get(m, [])):
                if tid not in {s["team_id"] for s in out}:
                    out.append({"team_id": tid, "name": self.display.get(tid), "games": self.games[tid]})
        return out[:n]


def link_team(conn: Connection, ps_team: dict[str, Any], index: TeamIndex) -> int | None:
    """Our team id for a PandaScore opponent, recording the link (or suggestions) in team_links."""
    ps_id = ps_team["id"]
    existing = conn.execute(select(team_links).where(team_links.c.ps_team_id == ps_id)).mappings().first()
    if existing and (existing["method"] == "manual" or existing["team_id"] is not None):
        return existing["team_id"]
    team_id, method, suggestions = index.match(ps_team.get("name"), ps_team.get("acronym"))
    values = {
        "ps_name": ps_team.get("name"),
        "ps_acronym": ps_team.get("acronym"),
        "team_id": team_id,
        "method": method,
        "suggestions_json": json.dumps(suggestions),
        "updated_at_utc": datetime.now(UTC),
    }
    conn.execute(
        dialect_insert(conn, team_links)
        .values(ps_team_id=ps_id, **values)
        .on_conflict_do_update(index_elements=["ps_team_id"], set_=values)
    )
    return team_id


def set_link(conn: Connection, ps_team_id: int, team_id: int | None, method: str = "manual") -> None:
    """Pin a PandaScore team to one of ours (or to nothing) and update its open fixtures right away."""
    values = {"team_id": team_id, "method": method, "updated_at_utc": datetime.now(UTC)}
    conn.execute(
        dialect_insert(conn, team_links)
        .values(ps_team_id=ps_team_id, **values)
        .on_conflict_do_update(index_elements=["ps_team_id"], set_=values)
    )
    for side in ("a", "b"):
        conn.execute(
            update(fixtures)
            .where(fixtures.c[f"ps_team_{side}_id"] == ps_team_id, fixtures.c.status != "finished")
            .values({f"team_{side}_id": team_id})
        )
