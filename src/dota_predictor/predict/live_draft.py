from __future__ import annotations

import difflib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sqlalchemy import select
from sqlalchemy.engine import Engine

from dota_predictor.features.asof import PregameFeatureState, fold_new_games, replay_state
from dota_predictor.logging_config import get_logger
from dota_predictor.modeling.baseline_elo import elo_win_prob
from dota_predictor.modeling.lightgbm_model import predict_artifact
from dota_predictor.normalize.games import compute_lineup_hash
from dota_predictor.settings import get_settings
from dota_predictor.storage.models import hero_reference, players, teams
from dota_predictor.storage.repository import get_engine

logger = get_logger("predict.live_draft")


def _load_artifact(path: Path, label: str) -> dict[str, Any] | None:
    # joblib/pickle artifacts written by this project's own `train lightgbm`; never load one from elsewhere.
    if path.exists():
        return joblib.load(path)
    logger.warning("No trained model found; falling back to Elo only", model=label, path=str(path))
    return None


class LiveDraftPredictor:
    """Replays every stored game into the same feature state used for training, then scores a match.

    With heroes it uses the post-draft model; without heroes (an upcoming match) the pre-draft model.
    Players default to each team's most recent full lineup.
    """

    def __init__(
        self,
        engine: Engine | None = None,
        model_path: Path | None = None,
        predraft_model_path: Path | None = None,
    ):
        self.engine = engine or get_engine()
        self.state: PregameFeatureState = replay_state(self.engine)
        self.hero_map: dict[str, int] = {}
        self.hero_id_to_name: dict[int, str] = {}
        self.team_map: dict[str, int] = {}
        self.team_id_to_name: dict[int, str] = {}
        self.player_map: dict[str, int] = {}
        self.player_id_to_name: dict[int, str] = {}
        self.account_to_player: dict[int, int] = {}
        self._load_name_maps()

        models_dir = get_settings().models_dir
        self.artifact = _load_artifact(model_path or models_dir / "lightgbm_pregame_v1.joblib", "post_draft")
        self.predraft_artifact = _load_artifact(
            predraft_model_path or models_dir / "lightgbm_predraft_v1.joblib", "pre_draft"
        )

    def _load_name_maps(self) -> None:
        with self.engine.begin() as conn:
            for hid, loc_name, raw_name in conn.execute(
                select(hero_reference.c.hero_id, hero_reference.c.localized_name, hero_reference.c.name)
            ):
                self.hero_id_to_name[hid] = loc_name or raw_name or f"Hero_{hid}"
                if loc_name:
                    self.hero_map[loc_name.lower()] = hid
                if raw_name:
                    self.hero_map[raw_name.replace("npc_dota_hero_", "").replace("_", " ").lower()] = hid
            games_known = self.state.team_elo.get_games_count
            for tid, tname, tag in conn.execute(select(teams.c.team_id, teams.c.canonical_name, teams.c.tag)):
                self.team_id_to_name[tid] = tname
                for key in {k.lower() for k in (tname, tag) if k}:
                    # Providers list some orgs under several ids; the one with the most games is the real one.
                    if key not in self.team_map or games_known(tid) > games_known(self.team_map[key]):
                        self.team_map[key] = tid
            for pid, pname, account in conn.execute(
                select(players.c.player_id, players.c.canonical_name, players.c.steam32_account_id)
            ):
                self.player_id_to_name[pid] = pname
                if account:
                    self.account_to_player[account] = pid
                if pname:
                    self.player_map[pname.lower()] = pid

    def refresh(self, reload_names: bool = False) -> int:
        """Fold games stored since startup into the state and reload names. Returns games added."""
        added = fold_new_games(self.engine, self.state)
        if added or reload_names:
            self._load_name_maps()
        return added

    def resolve_hero(self, query: str | int) -> int:
        if isinstance(query, int) or str(query).isdigit():
            hid = int(query)
            if hid in self.hero_id_to_name:
                return hid
            raise ValueError(f"Unknown hero ID: {query}")
        clean = str(query).strip().lower()
        if clean in self.hero_map:
            return self.hero_map[clean]
        matches = difflib.get_close_matches(clean, list(self.hero_map.keys()), n=1, cutoff=0.6)
        if matches:
            return self.hero_map[matches[0]]
        raise ValueError(f"Could not resolve hero name: '{query}'")

    def resolve_team(self, query: str | int) -> int:
        """Exact name/tag match, else a single close match; never silently pick among look-alikes."""
        if isinstance(query, int) or str(query).isdigit():
            return int(query)
        clean = str(query).strip().lower()
        if clean in self.team_map:
            return self.team_map[clean]
        matches = difflib.get_close_matches(clean, list(self.team_map.keys()), n=3, cutoff=0.8)
        ids = {self.team_map[m] for m in matches}
        if len(ids) == 1:
            return ids.pop()
        hint = f" Close matches: {matches}" if matches else ""
        raise ValueError(f"Could not resolve team name: '{query}'.{hint} Pass the numeric team id instead.")

    def resolve_player(self, query: str | int | None) -> int | None:
        if query is None:
            return None
        if isinstance(query, int) or str(query).isdigit():
            return int(query)
        clean = str(query).strip().lower()
        if clean in self.player_map:
            return self.player_map[clean]
        matches = difflib.get_close_matches(clean, list(self.player_map.keys()), n=1, cutoff=0.6)
        return self.player_map[matches[0]] if matches else None

    def _lineup(self, team_id: int, names: list[str | int] | None) -> tuple[list[int | None], str]:
        if names:
            return [self.resolve_player(p) for p in names], "given"
        latest = self.state.team_last_lineup.get(team_id)
        return (list(latest), "latest_known") if latest else ([None] * 5, "unknown")

    def predict_live_draft(
        self,
        radiant_team: str | int,
        dire_team: str | int,
        radiant_heroes: list[str | int] | None = None,
        dire_heroes: list[str | int] | None = None,
        radiant_players: list[str | int] | None = None,
        dire_players: list[str | int] | None = None,
        patch_id: int | None = None,
        start_utc: datetime | None = None,
    ) -> dict[str, Any]:
        """Win probability for a game starting now (or at ``start_utc``); heroes optional (pre-draft)."""
        now = start_utc or datetime.now(UTC)
        r_team_id, d_team_id = self.resolve_team(radiant_team), self.resolve_team(dire_team)
        r_team_name = self.team_id_to_name.get(r_team_id) or f"Team_{r_team_id}"
        d_team_name = self.team_id_to_name.get(d_team_id) or f"Team_{d_team_id}"

        pre_draft = not radiant_heroes and not dire_heroes
        r_hero_ids = [self.resolve_hero(h) for h in (radiant_heroes or [])]
        d_hero_ids = [self.resolve_hero(h) for h in (dire_heroes or [])]
        if not pre_draft and (len(r_hero_ids) != 5 or len(d_hero_ids) != 5):
            raise ValueError(f"Expected 5 heroes per side, got {len(r_hero_ids)} and {len(d_hero_ids)}")

        r_pids, r_source = self._lineup(r_team_id, radiant_players)
        d_pids, d_source = self._lineup(d_team_id, dire_players)
        if patch_id is None:
            patch_id = self.state.latest_patch_id(now)

        def lineup_hash(pids: list[int | None]) -> str | None:
            return compute_lineup_hash(pids)[1] if all(p is not None for p in pids) else None

        feats = self.state.features(
            now,
            r_team_id,
            d_team_id,
            list(zip(r_pids, r_hero_ids or [None] * 5, strict=True)),
            list(zip(d_pids, d_hero_ids or [None] * 5, strict=True)),
            patch_id,
            lineup_hash(r_pids),
            lineup_hash(d_pids),
        )
        p_elo = float(elo_win_prob(feats["elo_diff"]))
        artifact = self.predraft_artifact if pre_draft else self.artifact
        # A side with no known player never occurs in training (every game has ten real players), and the
        # model extrapolates it to ~1%/99%. Team Elo alone is the honest answer for such a side.
        no_players = not any(r_pids) or not any(d_pids)
        if artifact is not None and not no_players:
            X = np.array([[feats[f] for f in artifact["features"]]], dtype=float)
            p_final = float(predict_artifact(artifact, X, np.array([feats["elo_diff"]]))[0])
            model_name = "lightgbm"
        else:
            p_final = p_elo
            model_name = "elo_unknown_lineup" if artifact is not None else "elo_only"

        counter = self.state.draft.get_counter_rate
        pairings = [
            {
                "radiant_hero": self.hero_id_to_name[r_h],
                "dire_hero": self.hero_id_to_name[d_h],
                "radiant_advantage_pct": round((counter(r_h, d_h, patch_id=patch_id) - 0.5) * 100, 1),
            }
            for r_h in r_hero_ids
            for d_h in d_hero_ids
        ]
        pairings.sort(key=lambda x: abs(x["radiant_advantage_pct"]), reverse=True)

        p_bo3 = 3 * p_final**2 - 2 * p_final**3
        p_bo5 = p_final**3 * (10.0 - 15.0 * p_final + 6.0 * p_final**2)
        return {
            "matchup": {
                "radiant_team": r_team_name,
                "dire_team": d_team_name,
                "radiant_elo": feats["elo_radiant"],
                "dire_elo": feats["elo_dire"],
                "elo_diff": feats["elo_diff"],
                "radiant_games_known": feats["team_games_radiant"],
                "dire_games_known": feats["team_games_dire"],
            },
            "draft": {
                "radiant_heroes": [self.hero_id_to_name[h] for h in r_hero_ids],
                "dire_heroes": [self.hero_id_to_name[h] for h in d_hero_ids],
                "radiant_players": [self.player_id_to_name.get(p) or str(p) if p else None for p in r_pids],
                "dire_players": [self.player_id_to_name.get(p) or str(p) if p else None for p in d_pids],
                "radiant_lineup_source": r_source,
                "dire_lineup_source": d_source,
                "patch_id": patch_id,
                "features": feats,
                "player_comfort_diff_pct": round(feats["player_hero_comfort_diff"] * 100.0, 1),
                "radiant_side_bias_pct": round(feats["radiant_side_bias"] * 100.0, 1),
                "key_matchup_counters": pairings[:5],
            },
            "probabilities": {
                "pre_draft_radiant_win_pct": round(p_elo * 100.0, 1),
                "post_draft_radiant_win_pct": round(p_final * 100.0, 1),
                "post_draft_dire_win_pct": round((1.0 - p_final) * 100.0, 1),
                "draft_swing_pct": round((p_final - p_elo) * 100.0, 1),
                "predicted_winner": r_team_name if p_final >= 0.5 else d_team_name,
                "bo3_series_radiant_pct": round(p_bo3 * 100.0, 1),
                "bo5_series_radiant_pct": round(p_bo5 * 100.0, 1),
                "mode": "pre_draft" if pre_draft else "post_draft",
                "model": model_name,
                "p_radiant": p_final,
            },
        }

    def predict_side_neutral(
        self,
        team_a: str | int,
        team_b: str | int,
        players_a: list[str | int] | None = None,
        players_b: list[str | int] | None = None,
        start_utc: datetime | None = None,
    ) -> dict[str, Any]:
        """Per-map P(team A) before sides are known: average of A-as-Radiant and A-as-Dire."""
        predict = self.predict_live_draft
        ab = predict(team_a, team_b, radiant_players=players_a, dire_players=players_b, start_utc=start_utc)
        ba = predict(team_b, team_a, radiant_players=players_b, dire_players=players_a, start_utc=start_utc)
        p_rad = ab["probabilities"]["p_radiant"]
        p_dire = 1.0 - ba["probabilities"]["p_radiant"]
        return {"p_map": (p_rad + p_dire) / 2.0, "p_as_radiant": p_rad, "p_as_dire": p_dire, "detail": ab}
