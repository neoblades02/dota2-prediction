"""Point-in-time pregame features.

One ``PregameFeatureState`` serves both the offline builder (for every game:
features, then update) and the live predictor (replay every game with update
only, then features for the hypothetical game). One code path, so training and
serving cannot drift apart.
"""

from __future__ import annotations

import itertools
from collections import Counter, deque
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
from sqlalchemy import select
from sqlalchemy.engine import Engine

from dota_predictor.features.draft import DraftAdvantageEngine
from dota_predictor.features.meta import PatchMetaTracker
from dota_predictor.features.player_hero import PlayerHeroMasteryEngine
from dota_predictor.features.ratings import EloTracker
from dota_predictor.logging_config import get_logger
from dota_predictor.settings import get_settings
from dota_predictor.storage.models import game_participants, games, lineup_snapshots, patches, tournaments

logger = get_logger("features.asof")

HERO_POOL_WINDOW = timedelta(days=90)
FORM_WINDOW = 5
DEFAULT_REST_DAYS = 14.0

Part = tuple[int | None, int | None]  # (player_id, hero_id)


def aware_utc(dt: datetime | None) -> datetime | None:
    """Postgres returns aware datetimes, SQLite naive ones (stored as UTC); compare as aware UTC."""
    if dt is None:
        return None
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def naive_utc(dt: datetime) -> datetime:
    """SQLite hands back naive UTC datetimes; normalize everything to that."""
    return dt.astimezone(UTC).replace(tzinfo=None) if dt.tzinfo else dt


class PregameFeatureState:
    """Every sequential tracker. Call ``features`` before ``update`` for a game: strictly causal."""

    def __init__(
        self,
        patch_dates: dict[int, datetime] | None = None,
        team_k: float = 32.0,
        player_k: float = 24.0,
    ):
        self.patch_dates = {k: naive_utc(v) for k, v in (patch_dates or {}).items()}
        self.team_elo = EloTracker(k_factor=team_k)
        self.player_elo = EloTracker(k_factor=player_k)
        self.draft = DraftAdvantageEngine()
        self.mastery = PlayerHeroMasteryEngine()
        self.meta = PatchMetaTracker()
        self.team_form: dict[int, deque[bool]] = {}
        self.team_last_time: dict[int, datetime] = {}
        self.team_patch_games: Counter[tuple[int, int | None]] = Counter()
        self.player_recent: dict[int, deque[tuple[datetime, int]]] = {}
        self.player_pool: dict[int, Counter[int]] = {}
        self.lineup_counts: Counter[str] = Counter()
        self.pair_counts: Counter[tuple[int, int]] = Counter()
        self.h2h_wins: Counter[tuple[int, int]] = Counter()  # (winner, loser) -> games
        self.team_last_lineup: dict[int, list[int]] = {}  # most recent full five-player lineup
        self.last_game_id = 0  # highest games.game_id folded in (ids only grow, so newer rows have larger ids)

    def latest_patch_id(self, at_utc: datetime) -> int | None:
        at_utc = naive_utc(at_utc)
        live = [(eff, pid) for pid, eff in self.patch_dates.items() if eff <= at_utc]
        return max(live)[1] if live else None

    # -- per-side helpers -------------------------------------------------------------------------
    def _hero_breadth(self, pids: list[int], now: datetime) -> float:
        if not pids:
            return 0.0
        total = 0
        for pid in pids:
            recent = self.player_recent.get(pid)
            pool = self.player_pool.get(pid)
            if not recent:
                continue
            while recent and now - recent[0][0] > HERO_POOL_WINDOW:
                _, hid = recent.popleft()
                pool[hid] -= 1
                if pool[hid] <= 0:
                    del pool[hid]
            total += len(pool)
        return total / len(pids)

    def _pair_coplay(self, pids: list[int]) -> float:
        pairs = [self.pair_counts[(min(a, b), max(a, b))] for a, b in itertools.combinations(pids, 2)]
        return sum(pairs) / len(pairs) if pairs else 0.0

    def _form(self, team_id: int) -> float:
        hist = self.team_form.get(team_id)
        return sum(hist) / len(hist) if hist else 0.5

    def _rest_days(self, team_id: int, now: datetime) -> float:
        last = self.team_last_time.get(team_id)
        return (now - last).total_seconds() / 86400.0 if last else DEFAULT_REST_DAYS

    # -- public API -------------------------------------------------------------------------------
    def features(
        self,
        start_utc: datetime,
        radiant_id: int,
        dire_id: int,
        r_parts: list[Part],
        d_parts: list[Part],
        patch_id: int | None,
        r_lineup_hash: str | None = None,
        d_lineup_hash: str | None = None,
    ) -> dict[str, Any]:
        now = naive_utc(start_utc)
        r_pids = [p for p, _ in r_parts if p is not None]
        d_pids = [p for p, _ in d_parts if p is not None]
        r_heroes = [h for _, h in r_parts if h]
        d_heroes = [h for _, h in d_parts if h]
        r_ph = [(p, h) for p, h in r_parts if h]
        d_ph = [(p, h) for p, h in d_parts if h]

        elo_r, elo_d = self.team_elo.get_rating(radiant_id), self.team_elo.get_rating(dire_id)
        games_r, games_d = self.team_elo.get_games_count(radiant_id), self.team_elo.get_games_count(dire_id)

        p_elo_r = sum(self.player_elo.get_rating(p) for p in r_pids) / len(r_pids) if r_pids else 1500.0
        p_elo_d = sum(self.player_elo.get_rating(p) for p in d_pids) / len(d_pids) if d_pids else 1500.0
        p_games_r = sum(self.player_elo.get_games_count(p) for p in r_pids) / len(r_pids) if r_pids else 0.0
        p_games_d = sum(self.player_elo.get_games_count(p) for p in d_pids) / len(d_pids) if d_pids else 0.0

        form_r, form_d = self._form(radiant_id), self._form(dire_id)
        rest_diff = min(30.0, max(-30.0, self._rest_days(radiant_id, now) - self._rest_days(dire_id, now)))

        breadth_r, breadth_d = self._hero_breadth(r_pids, now), self._hero_breadth(d_pids, now)
        shared_r = self.lineup_counts[r_lineup_hash] if r_lineup_hash else 0
        shared_d = self.lineup_counts[d_lineup_hash] if d_lineup_hash else 0
        coplay_r, coplay_d = self._pair_coplay(r_pids), self._pair_coplay(d_pids)

        draft = self.draft.compute_draft_features(r_heroes, d_heroes, patch_id=patch_id)
        base_wrs = {h: self.draft.get_patch_hero_winrate(h, patch_id) for h in r_heroes + d_heroes}
        comfort = self.mastery.compute_lineup_comfort(r_ph, d_ph, base_hero_winrates=base_wrs)
        side_bias = self.meta.get_radiant_side_advantage(patch_id)

        patch_eff = self.patch_dates.get(patch_id) if patch_id is not None else None
        days_since_patch = (now - patch_eff).total_seconds() / 86400.0 if patch_eff and patch_eff <= now else 0.0
        patch_games_r = self.team_patch_games[(radiant_id, patch_id)]
        patch_games_d = self.team_patch_games[(dire_id, patch_id)]

        h2h_r, h2h_d = self.h2h_wins[(radiant_id, dire_id)], self.h2h_wins[(dire_id, radiant_id)]

        return {
            "patch_id": patch_id,
            "days_since_patch": round(days_since_patch, 2),
            "elo_radiant": round(elo_r, 2),
            "elo_dire": round(elo_d, 2),
            "elo_diff": round(elo_r - elo_d, 2),
            "team_games_radiant": games_r,
            "team_games_dire": games_d,
            "player_elo_radiant": round(p_elo_r, 2),
            "player_elo_dire": round(p_elo_d, 2),
            "player_elo_diff": round(p_elo_r - p_elo_d, 2),
            "player_games_radiant": round(p_games_r, 1),
            "player_games_dire": round(p_games_d, 1),
            "form_radiant_5": round(form_r, 3),
            "form_dire_5": round(form_d, 3),
            "form_diff_5": round(form_r - form_d, 3),
            "rest_days_diff": round(rest_diff, 2),
            "lineup_5man_shared_radiant": shared_r,
            "lineup_5man_shared_dire": shared_d,
            "lineup_5man_shared_diff": shared_r - shared_d,
            "pairwise_coplay_radiant": round(coplay_r, 2),
            "pairwise_coplay_dire": round(coplay_d, 2),
            "pairwise_coplay_diff": round(coplay_r - coplay_d, 2),
            "hero_breadth_radiant": round(breadth_r, 2),
            "hero_breadth_dire": round(breadth_d, 2),
            "hero_breadth_diff": round(breadth_r - breadth_d, 2),
            **draft,
            **comfort,
            "radiant_side_bias": side_bias,
            "patch_games_radiant": patch_games_r,
            "patch_games_dire": patch_games_d,
            "patch_games_diff": patch_games_r - patch_games_d,
            "h2h_games": h2h_r + h2h_d,
            "h2h_diff": round((h2h_r - h2h_d) / (h2h_r + h2h_d + 5.0), 4),  # shrunk toward 0 for few meetings
        }

    def update(
        self,
        start_utc: datetime,
        radiant_id: int,
        dire_id: int,
        r_parts: list[Part],
        d_parts: list[Part],
        patch_id: int | None,
        radiant_win: bool,
        r_lineup_hash: str | None = None,
        d_lineup_hash: str | None = None,
    ) -> None:
        now = naive_utc(start_utc)
        outcome_r = 1.0 if radiant_win else 0.0
        r_pids = [p for p, _ in r_parts if p is not None]
        d_pids = [p for p, _ in d_parts if p is not None]
        r_heroes = [h for _, h in r_parts if h]
        d_heroes = [h for _, h in d_parts if h]

        self.team_elo.update(radiant_id, dire_id, outcome_r)
        self.player_elo.update_groups(r_pids, d_pids, outcome_r)
        self.draft.update(r_heroes, d_heroes, radiant_win, patch_id=patch_id)
        self.mastery.update([(p, h) for p, h in r_parts if h], [(p, h) for p, h in d_parts if h], radiant_win)
        self.meta.update(patch_id, radiant_win)

        for team_id, won in ((radiant_id, radiant_win), (dire_id, not radiant_win)):
            self.team_form.setdefault(team_id, deque(maxlen=FORM_WINDOW)).append(won)
            self.team_last_time[team_id] = now
            self.team_patch_games[(team_id, patch_id)] += 1
        self.h2h_wins[(radiant_id, dire_id) if radiant_win else (dire_id, radiant_id)] += 1
        for team_id, pids in ((radiant_id, r_pids), (dire_id, d_pids)):
            if len(pids) == 5:
                self.team_last_lineup[team_id] = pids

        for pid, hid in r_parts + d_parts:
            if pid is not None and hid:
                self.player_recent.setdefault(pid, deque()).append((now, hid))
                self.player_pool.setdefault(pid, Counter())[hid] += 1

        for h in (r_lineup_hash, d_lineup_hash):
            if h:
                self.lineup_counts[h] += 1
        for pids in (r_pids, d_pids):
            for a, b in itertools.combinations(pids, 2):
                self.pair_counts[(min(a, b), max(a, b))] += 1


def load_games(engine: Engine, after_game_id: int = 0) -> tuple[list[dict[str, Any]], dict[int, datetime]]:
    """Load decided games (only those with game_id > after_game_id) chronologically, with participants."""
    with engine.begin() as conn:
        patch_dates = {
            pid: naive_utc(eff)
            for pid, eff in conn.execute(select(patches.c.patch_id, patches.c.effective_at_utc)).fetchall()
            if pid is not None and eff is not None
        }
        game_rows = (
            conn.execute(
                select(
                    games.c.game_id,
                    games.c.valve_match_id,
                    games.c.start_utc,
                    games.c.radiant_team_id,
                    games.c.dire_team_id,
                    games.c.radiant_win,
                    games.c.patch_id,
                    tournaments.c.tier,
                )
                .select_from(games.outerjoin(tournaments, games.c.tournament_id == tournaments.c.tournament_id))
                .where(
                    games.c.radiant_win.is_not(None),
                    games.c.radiant_team_id.is_not(None),
                    games.c.dire_team_id.is_not(None),
                    games.c.start_utc.is_not(None),
                    games.c.game_id > after_game_id,
                )
                .order_by(games.c.start_utc.asc(), games.c.game_id.asc())
            )
            .mappings()
            .fetchall()
        )
        part_rows = conn.execute(
            select(
                game_participants.c.game_id,
                game_participants.c.player_id,
                game_participants.c.hero_id,
                game_participants.c.side,
            ).where(game_participants.c.game_id > after_game_id)
        ).fetchall()
        lineup_rows = conn.execute(
            select(lineup_snapshots.c.game_id, lineup_snapshots.c.team_id, lineup_snapshots.c.player_set_hash).where(
                lineup_snapshots.c.game_id > after_game_id
            )
        ).fetchall()

    parts: dict[int, dict[str, list[Part]]] = {}
    for gid, pid, hid, side in part_rows:
        parts.setdefault(gid, {"radiant": [], "dire": []})[side].append((pid, hid))
    lineups = {(gid, tid): h for gid, tid, h in lineup_rows}

    out = []
    for g in game_rows:
        gid = g["game_id"]
        gp = parts.get(gid, {"radiant": [], "dire": []})
        out.append(
            {
                "game_id": gid,
                "valve_match_id": g["valve_match_id"],
                "start_utc": naive_utc(g["start_utc"]),
                "radiant_team_id": g["radiant_team_id"],
                "dire_team_id": g["dire_team_id"],
                "radiant_win": bool(g["radiant_win"]),
                "patch_id": g["patch_id"],
                "league_tier": g["tier"],
                "r_parts": gp["radiant"],
                "d_parts": gp["dire"],
                "r_lineup_hash": lineups.get((gid, g["radiant_team_id"])),
                "d_lineup_hash": lineups.get((gid, g["dire_team_id"])),
            }
        )
    return out, patch_dates


def _state_args(g: dict[str, Any]) -> dict[str, Any]:
    return {
        "start_utc": g["start_utc"],
        "radiant_id": g["radiant_team_id"],
        "dire_id": g["dire_team_id"],
        "r_parts": g["r_parts"],
        "d_parts": g["d_parts"],
        "patch_id": g["patch_id"],
        "r_lineup_hash": g["r_lineup_hash"],
        "d_lineup_hash": g["d_lineup_hash"],
    }


def replay_state(engine: Engine) -> PregameFeatureState:
    """Replay every stored game so the state reflects everything known right now (live prediction)."""
    all_games, patch_dates = load_games(engine)
    state = PregameFeatureState(patch_dates=patch_dates)
    for g in all_games:
        state.update(radiant_win=g["radiant_win"], **_state_args(g))
    state.last_game_id = max((g["game_id"] for g in all_games), default=0)
    logger.info("Replayed feature state", games=len(all_games))
    return state


def fold_new_games(engine: Engine, state: PregameFeatureState) -> int:
    """Apply games stored since the last replay or fold, in place. Returns how many were applied.

    ponytail: a late-ingested game older than ones already applied is folded after them; a nightly
    full replay would fix that ordering if Elo drift from it ever shows up.
    """
    new_games, patch_dates = load_games(engine, after_game_id=state.last_game_id)
    state.patch_dates = {k: naive_utc(v) for k, v in patch_dates.items()}
    for g in new_games:
        state.update(radiant_win=g["radiant_win"], **_state_args(g))
    state.last_game_id = max([state.last_game_id, *(g["game_id"] for g in new_games)])
    return len(new_games)


def build_pregame_features(
    engine: Engine,
    from_utc: datetime | None = None,
    to_utc: datetime | None = None,
    output_path: Path | None = None,
) -> pl.DataFrame:
    """Build the leakage-free training dataset: for each game, features as of its start, then update."""
    settings = get_settings()
    out_file = output_path or (settings.features_dir / "pregame_features_v1.parquet")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    lo = naive_utc(from_utc) if from_utc else None
    hi = naive_utc(to_utc) if to_utc else None

    all_games, patch_dates = load_games(engine)
    state = PregameFeatureState(patch_dates=patch_dates)
    logger.info("Building point-in-time features", total_eligible_games=len(all_games))

    records = []
    for g in all_games:
        args = _state_args(g)
        if (lo is None or g["start_utc"] >= lo) and (hi is None or g["start_utc"] <= hi):
            records.append(
                {
                    "game_id": g["game_id"],
                    "valve_match_id": g["valve_match_id"],
                    "cutoff_at_utc": g["start_utc"].isoformat(),
                    "league_tier": g["league_tier"],
                    "radiant_team_id": g["radiant_team_id"],
                    "dire_team_id": g["dire_team_id"],
                    "has_lineup": int(len(g["r_parts"]) == 5 and len(g["d_parts"]) == 5),
                    **state.features(**args),
                    "target_radiant_win": int(g["radiant_win"]),
                }
            )
        state.update(radiant_win=g["radiant_win"], **args)

    df = pl.DataFrame(records)
    if not df.is_empty():
        df.write_parquet(out_file)
        logger.info("Exported pregame features parquet", rows=len(df), columns=len(df.columns), path=str(out_file))
    return df
