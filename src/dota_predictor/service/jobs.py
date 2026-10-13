"""Background jobs: schedule (PandaScore), running (live scores and drafts), results (new games, grading).

Each job runs one cycle at a time in a worker thread. A failing cycle is logged and
recorded for the status bar; the loop keeps going.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.engine import Connection, Engine

from dota_predictor.features.asof import aware_utc
from dota_predictor.ingest import opendota_live, pandascore
from dota_predictor.ingest.explorer import (
    sync_explorer_match_index,
    sync_explorer_player_matches,
    sync_explorer_teams,
)
from dota_predictor.ingest.opendota import recover_match_details
from dota_predictor.logging_config import get_logger
from dota_predictor.normalize.team_links import norm, set_link
from dota_predictor.pipeline import refresh_reference
from dota_predictor.predict.service import PredictionService
from dota_predictor.service.coverage import coverage_alerts
from dota_predictor.storage.models import (
    SERIES_TYPE_TO_BEST_OF,
    fixtures,
    predictions,
    pro_match_index,
    tournaments,
)
from dota_predictor.storage.upsert import dialect_insert

logger = get_logger("service.jobs")

INTERVALS = {"schedule": 600, "running": 60, "results": 600}  # seconds; ~1,800 OpenDota calls a day of the free 3,000
LOCK_LEAD = timedelta(minutes=5)  # lock the pre-match prediction this close to the scheduled start
LIVE_WINDOW_BEFORE, LIVE_WINDOW_AFTER = timedelta(hours=12), timedelta(hours=6)
STALE_FIXTURE = timedelta(hours=12)
ROSTER_MATCH = 3  # live players shared with a linked id's last lineup for the live id to be the same team
UNAVAILABLE_RETRY = timedelta(days=2)  # OpenDota fills player rows a little after a game ends
RECOVER_AFTER = timedelta(hours=3)  # a queued game this old that nobody fetched is tried by id regardless
RECOVER_BATCH = 50  # ponytail: per results cycle; raise if the free OpenDota quota is not the bottleneck
ENDED_POLLS = 2  # a game missing from this many live polls in a row has ended (one miss may be the feed blinking)
GIVE_UP = timedelta(hours=6)  # ponytail: a requested game not back within this after the last request is dropped
INDEX_OVERLAP = timedelta(days=2)
REFERENCE_EVERY = timedelta(hours=24)
OPEN = ("upcoming", "running")


@dataclass
class JobStatus:
    last_ok: datetime | None = None
    last_error: str | None = None
    last_error_at: datetime | None = None
    runs: int = 0
    info: dict[str, Any] = field(default_factory=dict)


def _now() -> datetime:
    return datetime.now(UTC)


def _default_live_fetch() -> list[dict[str, Any]]:
    with opendota_live.make_client() as client:
        return opendota_live.fetch_live(client)


class JobRunner:
    def __init__(
        self,
        engine: Engine,
        service: PredictionService,
        live_fetch: Callable[[], list[dict[str, Any]]] | None = None,
        schedule_enabled: bool | None = None,
    ):
        self.engine = engine
        self.service = service
        self.live_fetch = live_fetch or _default_live_fetch
        self.schedule_enabled = pandascore.enabled() if schedule_enabled is None else schedule_enabled
        self.status = {name: JobStatus() for name in INTERVALS}
        self.live: dict[str, dict[str, Any]] = {}  # fixture_id -> latest live game and draft prediction
        self.feed: set[int] = set()  # match ids seen on the live feed and not yet gone
        self.gone: dict[int, int] = {}  # match id -> live polls in a row it has been missing from
        self.alerts: list[dict[str, Any]] = []
        self._last_reference_sync: datetime | None = None

    # --- loop plumbing -------------------------------------------------------

    def run(self, name: str) -> None:
        st = self.status[name]
        st.runs += 1
        try:
            st.info = getattr(self, f"{name}_once")() or {}
            st.last_ok = _now()
        except Exception as e:  # noqa: BLE001 - a job must never kill its loop
            logger.exception("Job cycle failed", job=name)
            st.last_error, st.last_error_at = f"{type(e).__name__}: {e}", _now()

    async def loop(self, name: str, start_delay: float = 0) -> None:
        await asyncio.sleep(start_delay)
        while True:
            await asyncio.to_thread(self.run, name)
            await asyncio.sleep(INTERVALS[name])

    # --- schedule ------------------------------------------------------------

    def schedule_once(self) -> dict[str, Any]:
        if not self.schedule_enabled:
            return {"skipped": "no PandaScore key"}
        res = pandascore.sync_schedule(self.engine)
        with self.engine.begin() as conn:
            res["pre_match_logged"] = self._log_pre_match(conn)
            res["locked"] = self._lock_due(conn, _now(), set())
            self.alerts = coverage_alerts(conn, _now(), True)
        return res

    def _open_fixtures(self, conn: Connection) -> list[dict[str, Any]]:
        rows = conn.execute(
            select(fixtures).where(
                fixtures.c.status.in_(OPEN), fixtures.c.team_a_id.is_not(None), fixtures.c.team_b_id.is_not(None)
            )
        ).mappings()
        return [dict(r) for r in rows]

    def _log_pre_match(self, conn: Connection) -> int:
        n = 0
        for fx in self._open_fixtures(conn):
            if fx["status"] == "upcoming" and (pred := self.service.predict_fixture(fx)):
                self.service.log(conn, fx, "pre_match", pred)
                n += 1
        return n

    def _lock_due(self, conn: Connection, now: datetime, live_ids: set[str]) -> int:
        """The graded prediction: pre-match view, written once as the match is about to start.

        A fixture is locked once, whatever the model version, and never after a map has been played: by then
        the ratings may already include that map, so there is no honest pre-match prediction left to make.
        """
        n = 0
        locked = {
            r[0]
            for r in conn.execute(select(predictions.c.target_ref).where(predictions.c.prediction_type == "locked"))
        }
        for fx in self._open_fixtures(conn):
            if fx["fixture_id"] in locked or (fx["score_a"] or 0) + (fx["score_b"] or 0):
                continue
            begin = aware_utc(fx["begin_at_utc"])
            due = fx["status"] == "running" or fx["fixture_id"] in live_ids or (begin and begin <= now + LOCK_LEAD)
            if due and (pred := self.service.predict_fixture({**fx, "status": "upcoming"})):
                self.service.log(conn, fx, "locked", pred)
                n += 1
        return n

    # --- running -------------------------------------------------------------

    def running_once(self) -> dict[str, Any]:
        info: dict[str, Any] = {}
        if self.schedule_enabled:
            try:
                info["pandascore_running"] = pandascore.sync_running(self.engine)
            except Exception as e:  # noqa: BLE001 - keep OpenDota live working when PandaScore fails
                logger.warning("PandaScore running sync failed", error=str(e))
                info["pandascore_error"] = f"{type(e).__name__}: {e}"
        games = opendota_live.parse_live(self.live_fetch())
        now = _now()
        live_now: dict[str, dict[str, Any]] = {}
        with self.engine.begin() as conn:
            for g in games:
                self._queue_game(conn, g, now)
                fx = self._attach(conn, g, now)
                if fx is None:
                    continue
                pred = self.service.predict_live_game(fx, g)
                if pred:
                    self.service.log(conn, fx, "live_draft", pred, pred["map_number"])
                live_now[fx["fixture_id"]] = {"game": g, "prediction": pred, "seen_at": now}
            info["locked"] = self._lock_due(conn, now, set(live_now))
            info["series_updates"] = self._series_updates(conn)
            info["expired"] = self._expire_stale(conn, now)
        self.live = live_now
        info["live_games"] = len(games)
        info["recovered"] = self._recover(self._ended({g["match_id"] for g in games if g.get("match_id")}))
        info |= self.settle_and_check(now)  # so a series grades within a minute of ending
        return info

    def _ended(self, now_ids: set[int]) -> list[int]:
        """Games that left the live feed for good; a single missing poll is not trusted."""
        ended = []
        for m in list(self.feed):
            if m in now_ids:
                self.gone.pop(m, None)
                continue
            self.gone[m] = self.gone.get(m, 0) + 1
            if self.gone[m] >= ENDED_POLLS:
                ended.append(m)
                self.feed.discard(m)
                self.gone.pop(m)
        self.feed |= now_ids
        return ended

    def _recover(self, match_ids: list[int]) -> dict[str, int]:
        """Fetch games that just ended, by id, and fold them into the ratings right away."""
        if not match_ids:
            return {}
        with self.engine.connect() as conn:
            rows = (
                conn.execute(
                    select(pro_match_index).where(
                        pro_match_index.c.match_id.in_(match_ids),
                        pro_match_index.c.detail_status.in_(("pending", "unavailable", "requested")),
                    )
                )
                .mappings()
                .all()
            )
        out = recover_match_details(self.engine, rows)
        if out.get("fetched"):
            self.service.refresh()
        return out

    def _queue_game(self, conn: Connection, g: Mapping[str, Any], now: datetime) -> None:
        """Every live league game is indexed, whatever its league, so the results job collects it once finished."""
        if not g.get("match_id"):
            return
        conn.execute(
            dialect_insert(conn, pro_match_index)
            .values(
                match_id=g["match_id"],
                start_time_utc=g.get("started_at") or now,
                radiant_team_id=g["radiant_team_id"],
                radiant_name=g.get("radiant_team_name"),
                dire_team_id=g["dire_team_id"],
                dire_name=g.get("dire_team_name"),
                leagueid=g["league_id"],
                series_id=g.get("series_id"),
                detail_status="pending",
                first_seen_utc=now,
                last_seen_utc=now,
            )
            .on_conflict_do_nothing(index_elements=["match_id"])
        )

    def _near(self, conn: Connection, now: datetime, cond: Any) -> list[dict[str, Any]]:
        out = []
        for fx in conn.execute(select(fixtures).where(cond)).mappings():
            begin = aware_utc(fx["begin_at_utc"])
            if begin is None or now - LIVE_WINDOW_BEFORE <= begin <= now + LIVE_WINDOW_AFTER:
                out.append(dict(fx))
        return out

    def _attach(self, conn: Connection, g: Mapping[str, Any], now: datetime) -> dict[str, Any] | None:
        """The open fixture a live game belongs to, creating an OpenDota-only one if the schedule lacks it.

        Returns None when the fixture of this pairing already finished (the feed can lag behind results).
        """
        r, d = g["radiant_team_id"], g["dire_team_id"]
        pair = or_(
            and_(fixtures.c.team_a_id == r, fixtures.c.team_b_id == d),
            and_(fixtures.c.team_a_id == d, fixtures.c.team_b_id == r),
        )
        near = self._near(conn, now, pair)
        if not any(f["source"] == "pandascore" for f in near):  # an OpenDota-only copy must not hide the schedule
            near = self._link_from_live(conn, now, g) + near
        open_ = [fx for fx in near if fx["status"] in OPEN]
        if near and not open_:
            return None
        if open_:
            fx = min(
                open_, key=lambda f: (f["source"] != "pandascore", abs((aware_utc(f["begin_at_utc"]) or now) - now))
            )
            values: dict[str, Any] = {"updated_at_utc": now}
            if g.get("series_id") and not fx["od_series_id"]:
                values["od_series_id"] = g["series_id"]
            conn.execute(update(fixtures).where(fixtures.c.fixture_id == fx["fixture_id"]).values(values))
            if fx["source"] == "pandascore":  # an OpenDota-only copy made before the schedule knew this match
                conn.execute(
                    update(fixtures)
                    .where(fixtures.c.source == "opendota", fixtures.c.status.in_(OPEN), pair)
                    .values(status="canceled", updated_at_utc=now)
                )
            return fx | values

        fid = f"od:{g['series_id']}" if g.get("series_id") else f"od:m{g['match_id']}"
        league = conn.execute(
            select(tournaments.c.title, tournaments.c.tier).where(tournaments.c.leagueid == g["league_id"])
        ).first()
        refresh = {
            "status": "running",
            "team_a_name": g.get("radiant_team_name"),
            "team_b_name": g.get("dire_team_name"),
            "updated_at_utc": now,
        }
        conn.execute(
            dialect_insert(conn, fixtures)
            .values(
                fixture_id=fid,
                source="opendota",
                leagueid=g["league_id"],
                league_name=league.title if league else f"League {g['league_id']}",
                tier=league.tier if league else None,
                begin_at_utc=g.get("started_at") or now,
                best_of=3,
                team_a_id=r,
                team_b_id=d,
                od_series_id=g.get("series_id"),
                score_a=0,
                score_b=0,
                **refresh,
            )
            .on_conflict_do_update(index_elements=["fixture_id"], set_=refresh, where=fixtures.c.status.in_(OPEN))
        )
        row = dict(conn.execute(select(fixtures).where(fixtures.c.fixture_id == fid)).mappings().one())
        return row if row["status"] in OPEN else None

    def _link_from_live(self, conn: Connection, now: datetime, g: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Fix a scheduled match's team links from a live game played at the same time.

        A team cannot play two matches at once, so a scheduled match that shares one live team and the start
        window is this game: its other side is the other live team. A side whose linked id last fielded the
        players now on the field is that live team under a new id (OpenDota often has look-alike duplicate
        team ids). Sides also resolve by an exact name match. Any link may be corrected, manual ones too: a
        manual link is a name guess, while a game being played is evidence. Applied only if exactly one
        scheduled match fits.
        """
        r, d = g["radiant_team_id"], g["dire_team_id"]
        live_names = {r: norm(g.get("radiant_team_name")), d: norm(g.get("dire_team_name"))}
        live_accounts = {r: [a for a, _ in g["radiant"]], d: [a for a, _ in g["dire"]]}
        cond = and_(
            fixtures.c.source == "pandascore",
            fixtures.c.status.in_(OPEN),
            fixtures.c.ps_team_a_id.is_not(None),
            fixtures.c.ps_team_b_id.is_not(None),
        )
        resolved = []
        for fx in self._near(conn, now, cond):
            sides = {}
            for side in ("a", "b"):
                tid = fx[f"team_{side}_id"]
                if tid in (r, d):
                    sides[side] = tid
                    continue
                roster = [
                    t for t in (r, d) if tid and self.service.shared_players(tid, live_accounts[t]) >= ROSTER_MATCH
                ]
                if len(roster) == 1:
                    sides[side] = roster[0]
                    continue
                hits = [t for t in (r, d) if live_names[t] and live_names[t] == norm(fx[f"team_{side}_name"])]
                sides[side] = hits[0] if len(hits) == 1 else None
            a, b = sides["a"], sides["b"]
            a = a or (b and (d if b == r else r))
            b = b or (a and (d if a == r else r))
            if a and b and {a, b} == {r, d}:
                resolved.append((fx, a, b))
        if len(resolved) != 1:
            return []
        fx, a, b = resolved[0]
        for side, tid in (("a", a), ("b", b)):
            if fx[f"team_{side}_id"] != tid:
                set_link(conn, fx[f"ps_team_{side}_id"], tid, method="live_match")
                fx[f"team_{side}_id"] = tid
        logger.info("Linked schedule teams from a live game", fixture=fx["fixture_id"], team_a=a, team_b=b)
        return [fx]

    def _series_updates(self, conn: Connection) -> int:
        n = 0
        for fx in self._open_fixtures(conn):
            maps = (fx["score_a"] or 0) + (fx["score_b"] or 0)
            if fx["status"] == "running" and maps and (pred := self.service.predict_fixture(fx)):
                self.service.log(conn, fx, "series_update", pred, maps)
                n += 1
        return n

    def _expire_stale(self, conn: Connection, now: datetime) -> int:
        """A fixture neither provider has touched for half a day is dropped; its locked prediction grades void."""
        res = conn.execute(
            update(fixtures)
            .where(
                fixtures.c.status.in_(OPEN),
                fixtures.c.updated_at_utc < now - STALE_FIXTURE,
            )
            .values(status="canceled", updated_at_utc=now)
        )
        return res.rowcount or 0

    # --- results -------------------------------------------------------------

    def results_once(self) -> dict[str, Any]:
        now = _now()
        with self.engine.begin() as conn:
            latest = conn.execute(select(func.max(pro_match_index.c.start_time_utc))).scalar()
            conn.execute(
                update(pro_match_index)
                .where(
                    pro_match_index.c.detail_status == "unavailable",
                    pro_match_index.c.start_time_utc >= now - UNAVAILABLE_RETRY,
                )
                .values(detail_status="pending", detail_error=None)
            )
        since = aware_utc(latest) - INDEX_OVERLAP if latest else datetime(2022, 1, 1, tzinfo=UTC)
        info: dict[str, Any] = {
            "index": sync_explorer_match_index(self.engine, since_utc=since),
            "players": sync_explorer_player_matches(self.engine),
            "team_names": sync_explorer_teams(self.engine, only_placeholders=True),
        }
        info["recovered"] = recover_match_details(self.engine, self._explorer_misses(now))
        reload_names = False
        if self._last_reference_sync is None or now - self._last_reference_sync >= REFERENCE_EVERY:
            info["reference"] = refresh_reference(self.engine)
            self._last_reference_sync, reload_names = now, True
        info["games_added"] = self.service.refresh(reload_names=reload_names)
        info |= self.settle_and_check(now)
        return info

    def _explorer_misses(self, now: datetime) -> list[Mapping[str, Any]]:
        """Games to fetch by id: those asked from Valve and awaited, and any still unfilled hours after starting."""
        unfilled = or_(
            pro_match_index.c.detail_status == "pending",
            and_(
                pro_match_index.c.detail_status == "unavailable",
                pro_match_index.c.detail_error == "explorer: no player rows",
            ),
        )
        with self.engine.connect() as conn:
            return (
                conn.execute(
                    select(pro_match_index)
                    .where(
                        pro_match_index.c.start_time_utc >= now - UNAVAILABLE_RETRY,
                        or_(
                            and_(
                                pro_match_index.c.detail_status == "requested",
                                pro_match_index.c.last_seen_utc >= now - GIVE_UP,
                            ),
                            and_(unfilled, pro_match_index.c.start_time_utc <= now - RECOVER_AFTER),
                        ),
                    )
                    .order_by(pro_match_index.c.start_time_utc)
                    .limit(RECOVER_BATCH)
                )
                .mappings()
                .all()
            )

    def settle_and_check(self, now: datetime) -> dict[str, Any]:
        with self.engine.begin() as conn:
            settled = self._settle_od_fixtures(conn)
            graded = self.service.grade(conn)
            self.alerts = coverage_alerts(conn, now, self.schedule_enabled)
        return {"od_settled": settled, "graded": graded, "alerts": len(self.alerts)}

    def _settle_od_fixtures(self, conn: Connection) -> int:
        """Series score and winner for OpenDota-only fixtures, from completed games of the same series."""
        n = 0
        rows = conn.execute(
            select(fixtures).where(fixtures.c.source == "opendota", fixtures.c.status.in_(OPEN))
        ).mappings()
        for fx in [dict(r) for r in rows]:
            if fx["od_series_id"]:
                cond = pro_match_index.c.series_id == fx["od_series_id"]
            else:
                cond = pro_match_index.c.match_id == int(fx["fixture_id"].removeprefix("od:m"))
            games = conn.execute(
                select(
                    pro_match_index.c.radiant_team_id,
                    pro_match_index.c.dire_team_id,
                    pro_match_index.c.radiant_win,
                    pro_match_index.c.series_type,
                ).where(cond, pro_match_index.c.radiant_win.is_not(None))
            ).all()
            if not games:
                continue
            wins: Counter[int] = Counter()
            for r, d, rw, _ in games:
                wins[r if rw else d] += 1
            best_of = 1 if not fx["od_series_id"] else SERIES_TYPE_TO_BEST_OF.get(games[0].series_type, fx["best_of"])
            sa, sb = wins[fx["team_a_id"]], wins[fx["team_b_id"]]
            done = sa + sb == 2 if best_of == 2 else max(sa, sb) >= best_of // 2 + 1
            winner = None if not done else "a" if sa > sb else "b" if sb > sa else "draw"
            conn.execute(
                update(fixtures)
                .where(fixtures.c.fixture_id == fx["fixture_id"])
                .values(
                    best_of=best_of,
                    score_a=sa,
                    score_b=sb,
                    status="finished" if done else fx["status"],
                    winner=winner,
                    updated_at_utc=_now(),
                )
            )
            n += 1
        return n

    # --- status for the page -------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        return {
            "schedule_source": "pandascore+opendota" if self.schedule_enabled else "opendota_live_only",
            "model_version": self.service.model_version,
            "jobs": {
                name: {
                    "last_ok": st.last_ok,
                    "last_error": st.last_error,
                    "last_error_at": st.last_error_at,
                    "runs": st.runs,
                    "every_s": INTERVALS[name],
                    "info": st.info,
                }
                for name, st in self.status.items()
            },
            "alerts": self.alerts,
        }
