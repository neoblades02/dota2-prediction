"""Fixture-level predictions on top of the warm LiveDraftPredictor, plus the prediction log.

The predictor is shared by background jobs and web requests, so every use goes
through one lock. Log rows reuse the ``predictions`` table (target_type
"fixture"); grading reuses ``prediction_results``.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import and_, select
from sqlalchemy.engine import Connection, Engine

from dota_predictor.logging_config import get_logger
from dota_predictor.predict.live_draft import LiveDraftPredictor
from dota_predictor.predict.markets import FORMATS, series_markets
from dota_predictor.settings import get_settings
from dota_predictor.storage.models import fixtures, prediction_results, predictions
from dota_predictor.storage.upsert import dialect_insert

logger = get_logger("predict.service")

LOW_DATA_GAMES = 20
MAJOR_TIERS = ("s", "a")


def model_version() -> str:
    """Short hash of the trained artifacts in use, so predictions from different models never mix."""
    h = hashlib.sha256()
    found = False
    for name in ("lightgbm_predraft_v1.joblib", "lightgbm_pregame_v1.joblib"):
        path = get_settings().models_dir / name
        if path.exists():
            h.update(path.read_bytes())
            found = True
    return h.hexdigest()[:12] if found else "elo_only"


class PredictionService:
    def __init__(self, engine: Engine, predictor: LiveDraftPredictor):
        self.engine = engine
        self.predictor = predictor
        self.lock = threading.Lock()
        self.low_data_games = LOW_DATA_GAMES
        self.model_version = model_version()

    def refresh(self, reload_names: bool = False) -> int:
        with self.lock:
            return self.predictor.refresh(reload_names=reload_names)

    def _games(self, team_id: int) -> int:
        return self.predictor.state.team_elo.get_games_count(team_id)

    def _pre_draft(self, a: int, b: int) -> dict[str, Any]:
        with self.lock:
            res = self.predictor.predict_side_neutral(a, b)
            games_a, games_b = self._games(a), self._games(b)
        m = res["detail"]["matchup"]
        return {
            "p_map": res["p_map"],
            "games_a": games_a,
            "games_b": games_b,
            "elo_a": m["radiant_elo"],
            "elo_b": m["dire_elo"],
            "low_data": min(games_a, games_b) < self.low_data_games
            or res["detail"]["probabilities"]["model"] == "elo_unknown_lineup",
            "model": res["detail"]["probabilities"]["model"],
        }

    def predict_fixture(self, fx: Mapping[str, Any]) -> dict[str, Any] | None:
        """Pre-draft prediction with markets from the fixture's current score; None when it can't be made."""
        a, b, bo = fx.get("team_a_id"), fx.get("team_b_id"), fx.get("best_of")
        if not a or not b or bo not in FORMATS:
            return None
        base = self._pre_draft(a, b)
        live = fx.get("status") == "running"
        score = (fx.get("score_a") or 0, fx.get("score_b") or 0) if live else (0, 0)
        try:
            mk = series_markets(base["p_map"], bo, *score)
        except ValueError:  # provider score inconsistent with format; fall back to the pre-match view
            mk = series_markets(base["p_map"], bo)
        return base | {"markets": mk, "p_series": mk["match_winner"]["a"], "score": list(score)}

    def predict_live_game(self, fx: Mapping[str, Any], live: Mapping[str, Any]) -> dict[str, Any] | None:
        """Post-draft map probability for team A once all ten heroes are known; series odds from the score."""
        a, b, bo = fx.get("team_a_id"), fx.get("team_b_id"), fx.get("best_of")
        if not a or not b or bo not in FORMATS:
            return None
        rad, dire = live["radiant"], live["dire"]
        if len(rad) != 5 or len(dire) != 5 or not all(h for _, h in [*rad, *dire]):
            return None
        a_is_radiant = live["radiant_team_id"] == a
        with self.lock:
            to_pid = self.predictor.account_to_player.get

            def lineup(side: list[tuple[int, int]]) -> list[int | None] | None:
                pids = [to_pid(acc) for acc, _ in side]
                return pids if any(pids) else None  # nobody known: fall back to the latest known lineup

            try:
                out = self.predictor.predict_live_draft(
                    live["radiant_team_id"],
                    live["dire_team_id"],
                    radiant_heroes=[h for _, h in rad],
                    dire_heroes=[h for _, h in dire],
                    radiant_players=lineup(rad),
                    dire_players=lineup(dire),
                )
            except ValueError as e:  # e.g. a hero id newer than our reference data
                logger.warning("Live draft prediction skipped", fixture=fx.get("fixture_id"), error=str(e))
                return None
        p_rad = out["probabilities"]["p_radiant"]
        p_now = p_rad if a_is_radiant else 1.0 - p_rad
        base = self._pre_draft(a, b)
        sa, sb = fx.get("score_a") or 0, fx.get("score_b") or 0
        try:
            win_after = series_markets(base["p_map"], bo, sa + 1, sb)["match_winner"]["a"]
            lose_after = series_markets(base["p_map"], bo, sa, sb + 1)["match_winner"]["a"]
        except ValueError:
            return None
        heroes = out["draft"]
        return base | {
            "p_map_now": p_now,
            "p_series": p_now * win_after + (1.0 - p_now) * lose_after,
            "map_number": sa + sb + 1,
            "heroes_a": heroes["radiant_heroes"] if a_is_radiant else heroes["dire_heroes"],
            "heroes_b": heroes["dire_heroes"] if a_is_radiant else heroes["radiant_heroes"],
            "a_side": "radiant" if a_is_radiant else "dire",
            "match_id": live.get("match_id"),
        }

    def log(self, conn: Connection, fx: Mapping[str, Any], kind: str, pred: Mapping[str, Any], map_number: int = 0):
        """Append one prediction row; the same (fixture, kind, map, model) is written only once."""
        now = datetime.now(UTC)
        fid = fx["fixture_id"]
        conn.execute(
            dialect_insert(conn, predictions)
            .values(
                prediction_key=f"{fid}|{kind}|{map_number}|{self.model_version}",
                target_type="fixture",
                target_ref=fid,
                prediction_type=kind,
                created_at_utc=now,
                cutoff_at_utc=now,
                model_version=self.model_version,
                p_team_a=pred["p_series"],
                team_a_id=fx.get("team_a_id"),
                team_b_id=fx.get("team_b_id"),
                roster_confidence="low_data" if pred.get("low_data") else "ok",
                payload_json=json.dumps({"map_number": map_number, **pred}, default=str),
            )
            .on_conflict_do_nothing(index_elements=["prediction_key"])
        )

    def grade(self, conn: Connection) -> int:
        """Settle every locked prediction whose fixture has finished. Draws and cancellations are void."""
        rows = conn.execute(
            select(
                predictions.c.prediction_id,
                predictions.c.p_team_a,
                fixtures.c.winner,
                fixtures.c.status,
                fixtures.c.source,
            )
            .join(fixtures, fixtures.c.fixture_id == predictions.c.target_ref)
            .outerjoin(prediction_results, prediction_results.c.prediction_id == predictions.c.prediction_id)
            .where(
                predictions.c.target_type == "fixture",
                predictions.c.prediction_type == "locked",
                prediction_results.c.id.is_(None),
                fixtures.c.status.in_(["finished", "canceled"]),
            )
        ).all()
        now = datetime.now(UTC)
        for pid, p, winner, status, source in rows:
            if status == "canceled" or winner not in ("a", "b"):
                result, outcome = "void", None
            else:
                outcome = 1 if winner == "a" else 0
                result = "won" if (p >= 0.5) == bool(outcome) else "lost"
            conn.execute(
                prediction_results.insert().values(
                    prediction_id=pid,
                    result_status=result,
                    realized_outcome=outcome,
                    evaluated_at_utc=now,
                    settled_source=source,
                )
            )
        return len(rows)

    def record(self, conn: Connection, recent: int = 50) -> dict[str, Any]:
        """Accuracy, Brier, log loss and calibration of graded locked predictions."""
        rows = (
            conn.execute(
                select(
                    predictions.c.p_team_a,
                    prediction_results.c.realized_outcome,
                    predictions.c.roster_confidence,
                    predictions.c.created_at_utc,
                    fixtures.c.fixture_id,
                    fixtures.c.team_a_name,
                    fixtures.c.team_b_name,
                    fixtures.c.tier,
                    fixtures.c.league_name,
                    fixtures.c.begin_at_utc,
                    fixtures.c.score_a,
                    fixtures.c.score_b,
                )
                .join(prediction_results, prediction_results.c.prediction_id == predictions.c.prediction_id)
                .join(fixtures, fixtures.c.fixture_id == predictions.c.target_ref)
                .where(and_(predictions.c.prediction_type == "locked", prediction_results.c.result_status != "void"))
                .order_by(fixtures.c.begin_at_utc.desc())
            )
            .mappings()
            .all()
        )

        def summary(sub: list[Mapping[str, Any]]) -> dict[str, Any]:
            if not sub:
                return {"n": 0}
            ps = [min(max(r["p_team_a"], 1e-6), 1 - 1e-6) for r in sub]
            ys = [r["realized_outcome"] for r in sub]
            return {
                "n": len(sub),
                "accuracy": sum((p >= 0.5) == bool(y) for p, y in zip(ps, ys, strict=True)) / len(sub),
                "brier": sum((p - y) ** 2 for p, y in zip(ps, ys, strict=True)) / len(sub),
                "log_loss": -sum(math.log(p if y else 1 - p) for p, y in zip(ps, ys, strict=True)) / len(sub),
            }

        buckets = []
        for i in range(10):
            lo, hi = i / 10, (i + 1) / 10
            sub = [r for r in rows if lo <= r["p_team_a"] < hi or (i == 9 and r["p_team_a"] == 1.0)]
            if sub:
                buckets.append(
                    {
                        "lo": lo,
                        "hi": hi,
                        "n": len(sub),
                        "mean_p": sum(r["p_team_a"] for r in sub) / len(sub),
                        "freq": sum(r["realized_outcome"] for r in sub) / len(sub),
                    }
                )
        return {
            "overall": summary(rows),
            "major": summary([r for r in rows if r["tier"] in MAJOR_TIERS]),
            "other_tiers": summary([r for r in rows if r["tier"] not in MAJOR_TIERS]),
            "low_data": summary([r for r in rows if r["roster_confidence"] == "low_data"]),
            "calibration": buckets,
            "recent": [dict(r) for r in rows[:recent]],
        }
