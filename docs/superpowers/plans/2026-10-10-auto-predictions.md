# Auto-predictions service Implementation Plan

> Executed natively in the session that wrote it (user said "go build"). Steps use checkbox syntax for tracking.

**Goal:** Unattended service on Dokploy that captures running and upcoming tournaments, predicts every fixture with markets, updates live, grades itself, and serves a dashboard.

**Architecture:** One FastAPI process holds a warm `LiveDraftPredictor`; three asyncio job loops (schedule, running, results) write to the shared Postgres; a single static HTML page reads a small JSON API.

**Tech Stack:** Python 3.12, SQLAlchemy Core, FastAPI, uvicorn, psycopg 3, httpx, LightGBM artifacts from local training.

**Spec:** `docs/superpowers/specs/2026-10-10-auto-predictions-design.md`

## Global Constraints

- Same schema and code must run on SQLite (tests, local) and Postgres (server).
- No new dependencies beyond `fastapi`, `uvicorn`, `psycopg[binary]`.
- Never auto-link a team or league on a fuzzy match.
- `DOTA_UI_PASSWORD` required to start the server; `/healthz` stays unauthenticated.
- Exactly one replica; one predictor in memory; never build a second state copy.
- OpenDota live poll every 2 min; PandaScore schedule every 10 min.

## Review Focus

- A PandaScore match with one or both opponents TBD must store, show as TBD, and never crash prediction.
- A team not in our DB must show "needs team mapping", not a wrong prediction.
- A PandaScore outage or missing key must leave OpenDota live fixtures working.
- Re-running any job must not duplicate fixtures, links, or prediction rows.
- Aware vs naive datetimes from Postgres must not break comparisons in job code.

## Reuse decisions

- The existing `predictions` and `prediction_results` tables become the prediction log and grading store: `target_type='fixture'`, `target_ref=<fixture id>`, `prediction_type` in `pre_match|locked|live_draft|series_update`, markets and flags in `payload_json`.
- The existing `tournaments` table stays the OpenDota league table. PandaScore tournaments go in a new `schedule_tournaments` table.

---

### Task 1: Database portability

**Files:** Modify `settings.py`, `storage/repository.py`, `ingest/base.py`, `ingest/explorer.py`, `ingest/opendota.py`, `normalize/entities.py`, `normalize/games.py`, `normalize/patches.py`, `pyproject.toml`. Create `storage/upsert.py`. Test `tests/test_upsert.py`.

**Produces:** `dialect_insert(conn, table) -> Insert` with `.on_conflict_do_update(index_elements=, set_=)` and `.on_conflict_do_nothing()` on both dialects. Settings read `DATABASE_URL` or `DOTA_DATABASE_URL`; `postgresql://` becomes `postgresql+psycopg://`.

- [ ] Test: every upsert statement used in the codebase compiles for sqlite and postgresql dialects; URL rewrite.
- [ ] Implement helper, replace all `sqlite_insert` call sites, settings alias.
- [ ] Run full suite.

### Task 2: Series markets for every format

**Files:** Modify `predict/markets.py`, `cli.py`. Test `tests/test_markets.py`.

**Produces:** `series_markets(p: float, best_of: int, score_a: int = 0, score_b: int = 0) -> dict` with keys `match_winner`, `correct_score`, `total_maps` (Bo3/Bo5), `handicap` (Bo3), `map1_match` (Bo3 from 0-0), `draw` (Bo2). `bo3_markets` kept as a thin wrapper.

- [ ] Tests: sums to 1 for Bo1/2/3/5, conditional cases (1-0 in Bo3 => p + q·p), Bo5 3-0 = p³.
- [ ] Implement; CLI uses it for any best_of.

### Task 3: Schema for schedule, fixtures, links

**Files:** Modify `storage/models.py`. Test via later tasks.

**Produces:** tables `schedule_tournaments`, `fixtures`, `team_links` (columns per spec).

### Task 4: Team and league linking

**Files:** Create `normalize/team_links.py`. Test `tests/test_team_links.py`.

**Produces:** `link_team(conn, ps_team: dict, index: TeamIndex) -> int | None`, `TeamIndex.from_conn(conn)`, `set_manual_link(conn, ps_team_id, team_id)`, `link_league(conn, ps_tournament_row) -> int | None`.

- [ ] Tests: exact name and acronym link; "old blood" vs "Cold BIood" stays unlinked with suggestion; manual link survives re-run; team with zero games not auto-linked.

### Task 5: PandaScore ingest

**Files:** Create `ingest/pandascore.py`. Fixtures `tests/fixtures/pandascore_*.json`. Test `tests/test_pandascore.py`.

**Produces:** `fetch_all(client, path) -> list[dict]` (pages via `x-total`), `upsert_tournaments(conn, rows)`, `upsert_matches(conn, rows, index) -> int`, `sync_schedule(engine) -> dict`, `sync_running(engine) -> list[dict]`.

- [ ] Tests: recorded responses upsert; TBD opponents; rerun is idempotent; finished match sets winner.

### Task 6: Incremental state

**Files:** Modify `features/asof.py`, `predict/live_draft.py`. Test `tests/test_features_state.py`.

**Produces:** `load_games(engine, after_game_id: int = 0)`, `replay_state` sets `state.last_game_id`, `fold_new_games(engine, state) -> int`, `LiveDraftPredictor.refresh() -> int`.

- [ ] Test: replay of first half + fold of second half gives identical features to a full replay.

### Task 7: Prediction service

**Files:** Create `predict/service.py`. Test `tests/test_service.py`.

**Produces:** `PredictionService(engine, predictor)` with `predict_fixture(fx) -> dict | None`, `predict_live_game(fx, live) -> dict | None`, `log(conn, fx_id, kind, result, map_number)`, `grade(conn) -> int`, `record(conn) -> dict`. One `threading.Lock`.

- [ ] Tests: side-neutral p equals CLI method; low_data when < 20 games; log idempotent per (fixture, kind, map, model_version); grading Brier/log loss on hand example.

### Task 8: Jobs and coverage

**Files:** Create `service/__init__.py`, `service/jobs.py`, `service/coverage.py`, `ingest/opendota_live.py`. Test `tests/test_jobs.py`.

**Produces:** `JobRunner(service, engine)` with `schedule_once()`, `running_once()`, `results_once()`, `status() -> dict`; `coverage_alerts(conn, now) -> list[dict]`; `parse_live(payload) -> list[dict]`.

- [ ] Tests: recorded `/live` with no PandaScore key creates `od:` fixture and a `live_draft` row once ten heroes exist; coverage alert raised for unlinked active premium league; job error recorded and loop continues.

### Task 9: Web app and page

**Files:** Create `service/app.py`, `service/static/index.html`. Modify `cli.py` (`serve`). Test `tests/test_app.py`.

- [ ] Tests: 401 without password; `/healthz` open; fixtures, tournaments, record, links endpoints return seeded data; PUT link updates.

### Task 10: Deployment files and docs

**Files:** Create `Dockerfile`, `.dockerignore`. Modify `.env.example`, `README.md`, `pyproject.toml`, `config/providers.yaml`.

- [ ] Docker build is not available locally; verify the server starts with `dota-predictor serve` against the real local DB and the page loads.
