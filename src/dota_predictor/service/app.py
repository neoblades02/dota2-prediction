"""Web app: JSON API plus one static page, with the job loops running in the same process.

Startup never blocks the port: an empty database is filled first (bootstrap), then the
predictor is built and the jobs start. Until then the API answers 503 with the phase,
and /healthz stays green so the container isn't restarted mid-bootstrap.
"""

from __future__ import annotations

import asyncio
import json
import secrets
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from sqlalchemy import func, or_, select
from sqlalchemy.engine import Engine

from dota_predictor.features.asof import aware_utc
from dota_predictor.logging_config import get_logger
from dota_predictor.normalize.team_links import set_link
from dota_predictor.predict.markets import fair_odds, market_rows
from dota_predictor.service.jobs import INTERVALS, JobRunner
from dota_predictor.settings import get_settings
from dota_predictor.storage.models import (
    fixture_odds,
    fixtures,
    prediction_results,
    predictions,
    schedule_tournaments,
    team_links,
    teams,
)
from dota_predictor.storage.upsert import dialect_insert

logger = get_logger("service.app")

STATIC = Path(__file__).parent / "static"
FINISHED_WINDOW = timedelta(days=3)
BOOT_RETRY_SECONDS = 300  # a failed first fill (OpenDota hiccup) retries on its own; finished steps are idempotent


def _out(obj: Any) -> Any:
    """JSON-ready copy with every datetime as an explicit UTC ISO string (SQLite hands back naive ones)."""
    if isinstance(obj, datetime):
        return aware_utc(obj).isoformat().replace("+00:00", "Z")
    if isinstance(obj, dict):
        return {k: _out(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_out(v) for v in obj]
    return obj


class LinkBody(BaseModel):
    team_id: int | None


class OddsBody(BaseModel):
    odds_a: float | None = None
    odds_b: float | None = None


class State:
    def __init__(self, engine: Engine):
        self.engine = engine
        self.phase = "starting"
        self.error: str | None = None
        self.runner: JobRunner | None = None
        self.tasks: list[asyncio.Task] = []


def _boot(state: State) -> None:
    from dota_predictor.pipeline import bootstrap, has_games
    from dota_predictor.predict.live_draft import LiveDraftPredictor
    from dota_predictor.predict.service import PredictionService

    if not has_games(state.engine):
        state.phase = "bootstrapping"
        bootstrap(state.engine)
    state.phase = "loading"
    service = PredictionService(state.engine, LiveDraftPredictor(engine=state.engine))
    state.runner = JobRunner(state.engine, service)


def create_app(
    engine: Engine | None = None,
    runner: JobRunner | None = None,
    password: str | None = None,
    start_background: bool = True,
) -> FastAPI:
    from dota_predictor.storage.repository import get_engine, init_db

    password = password or get_settings().ui_password
    if not password:
        raise RuntimeError("DOTA_UI_PASSWORD must be set: the dashboard is password-protected.")
    engine = engine or get_engine()
    state = State(engine)
    if runner is not None:
        state.runner, state.phase = runner, "ready"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if start_background:
            await asyncio.to_thread(init_db, engine)

            async def boot() -> None:
                while True:
                    try:
                        if state.runner is None:
                            await asyncio.to_thread(_boot, state)
                        # Schedule first, so live games attach to scheduled matches instead of becoming duplicates.
                        await asyncio.to_thread(state.runner.run, "schedule")
                        state.phase, state.error = "ready", None
                        state.tasks += [
                            asyncio.create_task(state.runner.loop(name, INTERVALS[name] if name == "schedule" else 0))
                            for name in INTERVALS
                        ]
                        return
                    except Exception as e:  # noqa: BLE001 - shown on the page; the bootstrap resumes on retry
                        logger.exception("Startup failed; retrying", retry_in_s=BOOT_RETRY_SECONDS)
                        state.phase, state.error = "failed", f"{type(e).__name__}: {e}"
                        await asyncio.sleep(BOOT_RETRY_SECONDS)

            state.tasks.append(asyncio.create_task(boot()))
        yield
        for t in state.tasks:
            t.cancel()

    app = FastAPI(title="Dota 2 predictions", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    basic = HTTPBasic(realm="dota-predictor")

    def auth(creds: HTTPBasicCredentials = Depends(basic)) -> None:
        if not secrets.compare_digest(creds.password.encode(), password.encode()):
            raise HTTPException(401, "Wrong password", headers={"WWW-Authenticate": 'Basic realm="dota-predictor"'})

    def ready() -> JobRunner:
        if state.phase != "ready" or state.runner is None:
            raise HTTPException(503, {"phase": state.phase, "error": state.error})
        return state.runner

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"ok": state.phase != "failed", "phase": state.phase}

    @app.get("/", dependencies=[Depends(auth)])
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/api/status", dependencies=[Depends(auth)])
    def status() -> dict[str, Any]:
        out: dict[str, Any] = {"phase": state.phase, "error": state.error, "server_time": datetime.now(UTC)}
        if state.runner is not None and state.phase == "ready":
            out |= state.runner.snapshot()
        return _out(out)

    def _fixture_view(runner: JobRunner, fx: dict[str, Any], with_prediction: bool) -> dict[str, Any]:
        a_name = fx["team_a_name"] or ("TBD" if not fx["ps_team_a_id"] and fx["source"] == "pandascore" else "?")
        b_name = fx["team_b_name"] or ("TBD" if not fx["ps_team_b_id"] and fx["source"] == "pandascore" else "?")
        view = fx | {
            "team_a_name": a_name,
            "team_b_name": b_name,
            "tbd": fx["source"] == "pandascore" and (not fx["ps_team_a_id"] or not fx["ps_team_b_id"]),
            "needs_mapping": (bool(fx["ps_team_a_id"]) and not fx["team_a_id"])
            or (bool(fx["ps_team_b_id"]) and not fx["team_b_id"]),
        }
        if with_prediction and (pred := runner.service.predict_fixture(fx)):
            pred["rows"] = [
                {"label": label, "outcomes": [{"name": n, "p": p, "odds": fair_odds(p)} for n, p in outs]}
                for label, outs in market_rows(pred["markets"], a_name, b_name)
            ]
            view["prediction"] = pred
        if live := runner.live.get(fx["fixture_id"]):
            view["live"] = {k: live[k] for k in ("prediction", "seen_at")} | {
                "game_time": live["game"].get("game_time"),
                "radiant_lead": live["game"].get("radiant_lead"),
                "match_id": live["game"].get("match_id"),
            }
        return view

    @app.get("/api/fixtures", dependencies=[Depends(auth)])
    def list_fixtures(
        status: str = Query("upcoming", pattern="^(upcoming|running|finished)$"),
        tournament: str | None = None,
    ) -> list[dict[str, Any]]:
        runner = ready()
        live_ids = list(runner.live)  # a live game counts as running even before the schedule says so
        if status == "running":
            q = select(fixtures).where(or_(fixtures.c.status == "running", fixtures.c.fixture_id.in_(live_ids)))
        elif status == "upcoming":
            q = select(fixtures).where(fixtures.c.status == "upcoming", fixtures.c.fixture_id.not_in(live_ids))
        else:
            q = select(fixtures).where(fixtures.c.status == status)
        if status == "finished":
            q = q.where(fixtures.c.begin_at_utc >= datetime.now(UTC) - FINISHED_WINDOW)
            q = q.order_by(fixtures.c.begin_at_utc.desc())
        else:
            q = q.order_by(fixtures.c.begin_at_utc.asc())
        if tournament:
            q = q.where(fixtures.c.tournament_key == tournament)
        with engine.connect() as conn:
            rows = [dict(r) for r in conn.execute(q.limit(200)).mappings()]
            book = {
                r.fixture_id: {"a": r.odds_a, "b": r.odds_b}
                for r in conn.execute(
                    select(fixture_odds).where(fixture_odds.c.fixture_id.in_([r["fixture_id"] for r in rows]))
                )
            }
            locked = {}
            if rows:
                locked = {
                    r.target_ref: {"p_series": r.p_team_a, "result": r.result_status}
                    for r in conn.execute(
                        select(predictions.c.target_ref, predictions.c.p_team_a, prediction_results.c.result_status)
                        .outerjoin(
                            prediction_results, prediction_results.c.prediction_id == predictions.c.prediction_id
                        )
                        .where(
                            predictions.c.prediction_type == "locked",
                            predictions.c.target_ref.in_([r["fixture_id"] for r in rows]),
                        )
                    )
                }
        out = []
        for fx in rows:
            view = _fixture_view(runner, fx, with_prediction=status != "finished")
            if fx["fixture_id"] in locked:
                view["locked"] = locked[fx["fixture_id"]]
            view["book_odds"] = book.get(fx["fixture_id"])
            out.append(view)
        return _out(out)

    @app.put("/api/fixtures/{fixture_id}/odds", dependencies=[Depends(auth)])
    def put_odds(fixture_id: str, body: OddsBody) -> dict[str, Any]:
        """Save the bookmaker's match-winner prices. A side left out keeps its value; 0 or null clears it."""
        ready()
        given = {k: v for k, v in body.model_dump().items() if k in body.model_fields_set}
        if any(v is not None and 0 < v <= 1 for v in given.values()):
            raise HTTPException(422, "Decimal odds must be greater than 1")
        given = {k: (v if v else None) for k, v in given.items()}
        with engine.begin() as conn:
            if conn.execute(select(fixtures.c.fixture_id).where(fixtures.c.fixture_id == fixture_id)).first() is None:
                raise HTTPException(404, "No such fixture")
            now = datetime.now(UTC)
            conn.execute(
                dialect_insert(conn, fixture_odds)
                .values(fixture_id=fixture_id, updated_at_utc=now, **given)
                .on_conflict_do_update(index_elements=["fixture_id"], set_={"updated_at_utc": now, **given})
            )
            row = conn.execute(select(fixture_odds).where(fixture_odds.c.fixture_id == fixture_id)).mappings().one()
        return _out(dict(row))

    @app.get("/api/fixtures/{fixture_id}", dependencies=[Depends(auth)])
    def get_fixture(fixture_id: str) -> dict[str, Any]:
        runner = ready()
        with engine.connect() as conn:
            fx = conn.execute(select(fixtures).where(fixtures.c.fixture_id == fixture_id)).mappings().first()
            if fx is None:
                raise HTTPException(404, "No such fixture")
            history = []
            for r in conn.execute(
                select(predictions).where(predictions.c.target_ref == fixture_id).order_by(predictions.c.created_at_utc)
            ):
                payload = json.loads(r.payload_json or "{}")
                history.append(
                    {
                        "kind": r.prediction_type,
                        "created_at": r.created_at_utc,
                        "p_series": r.p_team_a,
                        "model_version": r.model_version,
                        "map_number": payload.get("map_number"),
                        "p_map": payload.get("p_map_now", payload.get("p_map")),
                    }
                )
        return _out(_fixture_view(runner, dict(fx), with_prediction=fx["status"] != "finished") | {"history": history})

    @app.get("/api/tournaments", dependencies=[Depends(auth)])
    def list_tournaments(status: str = Query("running", pattern="^(running|upcoming)$")) -> list[dict[str, Any]]:
        ready()
        with engine.connect() as conn:
            ts = [
                dict(r)
                for r in conn.execute(
                    select(schedule_tournaments).where(schedule_tournaments.c.status == status)
                ).mappings()
            ]
            counts: dict[str, dict[str, int]] = {}
            for key, st, a, b, pa, pb in conn.execute(
                select(
                    fixtures.c.tournament_key,
                    fixtures.c.status,
                    fixtures.c.team_a_id,
                    fixtures.c.team_b_id,
                    fixtures.c.ps_team_a_id,
                    fixtures.c.ps_team_b_id,
                ).where(fixtures.c.tournament_key.in_([t["tournament_key"] for t in ts]))
            ):
                c = counts.setdefault(
                    key, {"total": 0, "predicted": 0, "tbd": 0, "needs_mapping": 0, "running": 0, "finished": 0}
                )
                c["total"] += 1
                c["running"] += st == "running"
                c["finished"] += st == "finished"
                if not pa or not pb:
                    c["tbd"] += 1
                elif a and b:
                    c["predicted"] += st != "finished"
                else:
                    c["needs_mapping"] += 1
        tier_rank = {"s": 0, "a": 1, "b": 2, "c": 3, "d": 4}
        ts.sort(
            key=lambda t: (
                tier_rank.get(t["tier"] or "", 9),
                aware_utc(t["begin_at_utc"]) or datetime.max.replace(tzinfo=UTC),
            )
        )
        return _out([t | {"counts": counts.get(t["tournament_key"], {"total": 0})} for t in ts])

    @app.get("/api/record", dependencies=[Depends(auth)])
    def record() -> dict[str, Any]:
        runner = ready()
        with engine.connect() as conn:
            return _out(runner.service.record(conn))

    @app.get("/api/links", dependencies=[Depends(auth)])
    def list_links(unlinked: bool = True) -> list[dict[str, Any]]:
        ready()
        with engine.connect() as conn:
            q = select(team_links).order_by(team_links.c.ps_name)
            if unlinked:
                q = q.where(team_links.c.team_id.is_(None))
            used = dict(
                conn.execute(
                    select(fixtures.c.ps_team_a_id, func.count())
                    .where(fixtures.c.status != "finished")
                    .group_by(fixtures.c.ps_team_a_id)
                ).all()
            )
            for k, v in conn.execute(
                select(fixtures.c.ps_team_b_id, func.count())
                .where(fixtures.c.status != "finished")
                .group_by(fixtures.c.ps_team_b_id)
            ).all():
                used[k] = used.get(k, 0) + v
            rows = [dict(r) for r in conn.execute(q).mappings()]
        for r in rows:
            r["suggestions"] = json.loads(r.pop("suggestions_json") or "[]")
            r["open_fixtures"] = used.get(r["ps_team_id"], 0)
        rows.sort(key=lambda r: -r["open_fixtures"])
        return _out(rows)

    @app.put("/api/links/team/{ps_team_id}", dependencies=[Depends(auth)])
    def put_link(ps_team_id: int, body: LinkBody) -> dict[str, Any]:
        ready()
        with engine.begin() as conn:
            if (
                conn.execute(select(team_links.c.ps_team_id).where(team_links.c.ps_team_id == ps_team_id)).first()
                is None
            ):
                raise HTTPException(404, "Unknown PandaScore team")
            if (
                body.team_id is not None
                and conn.execute(select(teams.c.team_id).where(teams.c.team_id == body.team_id)).first() is None
            ):
                raise HTTPException(422, "No team with that id in the database")
            set_link(conn, ps_team_id, body.team_id)
            row = conn.execute(select(team_links).where(team_links.c.ps_team_id == ps_team_id)).mappings().one()
        return _out(dict(row))

    @app.get("/api/teams/search", dependencies=[Depends(auth)])
    def search_teams(q: str = Query(..., min_length=2, max_length=60)) -> list[dict[str, Any]]:
        runner = ready()
        like = f"%{q.strip()}%"
        with engine.connect() as conn:
            rows = conn.execute(
                select(teams.c.team_id, teams.c.canonical_name, teams.c.tag)
                .where(or_(teams.c.canonical_name.ilike(like), teams.c.tag.ilike(like)))
                .limit(50)
            ).all()
        games = runner.service.predictor.state.team_elo.get_games_count
        out = [{"team_id": t, "name": n, "tag": tag, "games": games(t)} for t, n, tag in rows]
        return sorted(out, key=lambda r: -r["games"])[:10]

    app.state.dashboard = state
    return app
