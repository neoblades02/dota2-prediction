# Auto-fetch, auto-predict, and dashboard: design

Date: 2026-10-10 · Status: draft for review

## Goal

Run unattended on the user's Dokploy server so that:

1. Every major tournament that is running or coming soon is captured, with its matches.
2. Upcoming matches get a pre-match prediction with betting markets.
3. Matches that start get updated predictions: post-draft once heroes are picked, and series markets after every map.
4. Finished matches are pulled in automatically, ratings update, and every logged prediction is graded.
5. One web page shows all of it, plus a running track record and a coverage check.

Training stays on the user's PC. The server only predicts. Both use the same Postgres database.

## Non-goals

- Retraining on the server.
- Scraping bookmaker odds. The user pastes odds into the page by hand.
- Manual fixture entry, notifications, user accounts, multiple replicas.

## Database

The user's Postgres, already reachable from the PC and set as `DATABASE_URL` in `.env`, becomes the one shared database.

- The server writes to it: new games, fixtures, predictions.
- The PC reads from it to train, so the PC no longer needs its own ingest.
- SQLite stays the default when no URL is set, and stays the test database.

Changes needed:
- Settings accept `DATABASE_URL` as well as `DOTA_DATABASE_URL`, since Dokploy names it that way. A `postgresql://` URL is rewritten to `postgresql+psycopg://`.
- New dependency `psycopg[binary]`.
- Every upsert currently uses SQLite's insert construct, across eight files. Verified 2026-10-10: these fail to compile for Postgres. One helper in `storage/` picks the dialect's insert from the connection, and all call sites use it.
- One-time fill: run the existing pipeline inside the container against Postgres: init, reference, patches, explorer index, players, teams, normalize. About 40 minutes of API time, plus normalize. No migration script.

## Data sources

PandaScore is the primary schedule source. OpenDota is always on and is the fallback: if `DOTA_PANDASCORE_API_KEY` is unset or PandaScore calls keep failing, the app still runs, but matches only appear once their lobby is live.

| Source | Used for | Poll | Budget |
|---|---|---|---|
| PandaScore `/dota2/tournaments/running` and `/upcoming` | Tournament list: league, stage, tier, dates, teams | 10 min | ~12 req/h of 1,000/h |
| PandaScore `/dota2/matches/upcoming` | Every scheduled match: teams or TBD, start time, Bo format, tier | 10 min | ~6 req/h |
| PandaScore `/dota2/matches/running` | Series score while a match is live (`results`) | 2 min | ~30 req/h |
| PandaScore `/dota2/matches/past` | Winners of fixtures that just finished, for grading | 10 min | ~6 req/h |
| OpenDota `/live` | Hero picks and player accounts during the draft; creates fixtures PandaScore does not have | 2 min | ~720 req/day of ~3,000/day |
| OpenDota `/explorer`, existing incremental sync | Completed games, so ratings stay current, OpenDota-only fixtures get graded, and coverage can be checked | 30 min | ~150 req/day |

Every PandaScore list call pages through all results using `x-total`, and fetches all tiers. Tier filtering happens only in the page.

Verified 2026-10-10:
- The PandaScore key returns running tournaments and upcoming and running Dota 2 matches, with `number_of_games`, `opponents`, `league`, `tournament.tier`, `results`.
- PandaScore listed BLAST Slam Season 8 Playoffs at tier A plus five tier D events as running. It listed zero upcoming tournaments and zero upcoming series.
- OpenDota's last 14 days show five leagues. Each one that is still running is also in PandaScore.
- OpenDota `/live` returns `league_id`, `series_id`, `team_id_radiant/dire`, and `players[].account_id/hero_id`.

## Components

### 1. Tournament coverage (`service/coverage.py`)

"Major" means PandaScore tier S or A, or an OpenDota league with tier `premium`.

- `tournaments` table: PandaScore tournament id, league, serie, stage name, tier, begin_at, end_at, status, OpenDota league id when linked.
- Running and upcoming tournaments are stored even before any match is scheduled, so the page can show an event before its first match appears.
- Every 30 minutes the coverage check compares the two sources:
  - An OpenDota premium or professional league with games in the last 3 days but no PandaScore tournament raises a **missing from schedule** alert.
  - A PandaScore tier S or A tournament that is running but has no fixtures in the next 24 hours and no live game raises a **no matches found** alert.
- Alerts show in the page's status bar and on the Tournaments tab. Ongoing majors are covered twice in practice: by PandaScore, and by OpenDota's live feed for any league game.
- PandaScore-to-OpenDota league linking follows the team-link rule: auto-link only on an exact normalized name match, otherwise a suggestion in the mapping tab.

### 2. Team linking (`normalize/team_links.py`)

PandaScore team ids differ from our team ids, which are OpenDota/Valve ids. A `team_links` table maps them. OpenDota fixtures need no linking.

- Auto-link only on an exact, case-insensitive match of name or acronym against a team with at least one game.
- Anything else is stored as **unlinked** with up to three fuzzy suggestions. It is never auto-linked. On today's fixtures, fuzzy matching sent "old blood" to "Cold BIood", a different team. 16 of 19 names resolved; 3 had no match.
- The page has a Mapping tab to confirm a suggestion or enter a team id. Manual links are never overwritten.
- A fixture with an unlinked team gets no prediction and shows "needs team mapping".

### 3. Fixtures and prediction log (`storage/models.py`)

`fixtures`: id (text PK, `ps:<pandascore match id>` or `od:<opendota series id>`), source, tournament id, begin_at, best_of, team A/B ids (ours, nullable while TBD), team A/B PandaScore ids, tier, status, series score A/B, winner, updated_at.

An OpenDota live game attaches to the running fixture whose two linked team ids match. If none matches, it creates an `od:` fixture. For OpenDota fixtures, best_of comes from `series_type`, otherwise Bo3.

`prediction_log`, append-only: id, fixture_id, created_at, kind (`pre_match`, `locked`, `live_draft`, `series_update`), map_number, p_map_a, p_series_a, markets JSON, model_version (SHA-256 prefix of the artifact file), low_data flag.

Logging rules:
- `pre_match` row when a fixture first has both teams known. The page always shows a fresh in-memory recomputation; the log does not get a row per poll.
- `locked` row at the last schedule poll before begin_at. This is the row graded for the track record. OpenDota fixtures lock at first sight.
- `live_draft` row once all ten heroes of a map are known.
- `series_update` row after each map result.

### 4. Markets (`predict/markets.py`, extended)

Closed-form from the per-map probability, maps independent, as today:
- Bo1: winner.
- Bo2: 2-0, 1-1 draw, 0-2.
- Bo3: existing markets.
- Bo5: winner, correct score 3-0 to 0-3, total maps.
- Any format, conditional on the current series score, so live markets update after each map.

### 5. Live state (`predict/service.py`)

One `LiveDraftPredictor` stays in memory for the life of the process. Measured locally: about 700 MB, 12.5 s to build at startup, 1.5 ms per match prediction.

- New games are folded into the state in place. `load_games` gains an optional filter so only games not yet applied are loaded, sorted by start time. No second copy of the state is ever built.
- One lock guards the predictor, since background jobs and web requests share it.
- A team with fewer than 20 known games sets `low_data`. The page shows a warning badge; the track record reports these separately.

### 6. Jobs (`service/jobs.py`)

Plain asyncio loops started with the web app. Blocking code runs via `asyncio.to_thread`. Each job:
- catches and logs its own errors and keeps running,
- records last success time and last error, shown in the status bar,
- uses the existing rate-limited HTTP client and respects rate-limit headers.

Jobs:
- `schedule`: PandaScore tournaments, upcoming and past matches, every 10 min. Skipped when no key is set.
- `running`: PandaScore running matches and OpenDota live, every 2 min.
- `results`: explorer sync, normalize new games, fold into state, grade, coverage check, every 30 min.

### 7. Web app (`service/app.py`, `service/static/index.html`)

FastAPI + uvicorn, started by `dota-predictor serve`. New dependencies: `fastapi`, `uvicorn`.

API:
- `GET /api/tournaments?status=running|upcoming`, with fixture counts and alerts
- `GET /api/fixtures?status=upcoming|running|finished&tournament=`
- `GET /api/fixtures/{id}`, with prediction history
- `GET /api/record`: accuracy, Brier, log loss, calibration buckets, split by tier and low-data
- `GET /api/links?unlinked=1`, `PUT /api/links/team/{pandascore_id}`, `PUT /api/links/league/{pandascore_id}`
- `GET /api/status`: job health, coverage alerts, active schedule source
- `GET /healthz`, no auth, for Dokploy

Page: one HTML file, plain JavaScript, polls every 30 s. Tabs:
- **Tournaments**: running and upcoming events by tier and date, with scheduled, predicted and TBD match counts, and coverage alerts.
- **Upcoming**: match cards by start time with series probability, markets table, low-data badge, and odds boxes. Pasted odds stay in the viewer's browser only and show the edge per outcome. Default filter tiers S/A/B, toggle for C/D.
- **Live**: score, current map probability through the draft, live series markets.
- **Track record**: headline numbers, calibration chart, table of graded predictions.
- **Mapping**: unlinked teams and leagues with suggestions.

Auth: HTTP basic auth on everything except `/healthz`, password from `DOTA_UI_PASSWORD`. The app refuses to start without it.

### 8. Deployment

- `Dockerfile` on `python:3.12-slim`, installs the package, runs `dota-predictor serve`.
- Dokploy app with one volume at `/app/data` for model files and the raw payload store. Exactly one replica.
- Env vars: `DATABASE_URL`, `DOTA_UI_PASSWORD`, optional `DOTA_PANDASCORE_API_KEY`, optional `DOTA_OPENDOTA_API_KEY`.
- Settings gain `DOTA_MODELS_DIR` so models can live on the volume.
- Server size: Dokploy asks for 2 GB RAM minimum; 4 GB is comfortable with this app's ~700 MB plus Postgres.
- After a local retrain against the shared Postgres: copy the two `.joblib` files to the volume over SSH, restart the container. Server and PC must run the same code commit.
- Model files are pickle-based. Only files the user trained are ever placed on the volume; no upload endpoint exists.

## Testing

- Upserts: every upsert helper call compiles for both SQLite and Postgres dialects. One opt-in test runs the normalize step against a real Postgres when `DOTA_TEST_DATABASE_URL` is set.
- Markets: every format sums to 1, conditional markets match hand-computed cases, Bo3 matches the existing results.
- Team links: exact name and acronym auto-link; "old blood" stays unlinked; manual links survive a re-run.
- PandaScore parsing: recorded JSON responses in `tests/fixtures/` upsert into `tournaments` and `fixtures` correctly, including TBD opponents, multi-page results and rescheduled matches.
- Coverage: an OpenDota premium league with recent games and no PandaScore tournament raises the alert; a linked one does not.
- OpenDota fallback: a recorded `/live` response with no PandaScore key creates an `od:` fixture and a `live_draft` row once ten heroes are present.
- Incremental state: folding new games gives the same features as a full replay.
- Grading: Brier and log loss on a hand-checked example.
- API: FastAPI test client with a seeded SQLite DB; requests without the password get 401; `/healthz` stays open.

## Risks

- PandaScore listed no upcoming tournaments on 2026-10-10, though DreamLeague Season 30 and PARI Universe main events likely follow their qualifiers. Events probably appear only days ahead, once matches are scheduled. A full long-range calendar would need Liquipedia's approved API, which the user can apply for; it is not part of this build.
- PandaScore's terms may restrict betting-related use. A competitor's blog says so; not confirmed from PandaScore. The OpenDota fallback keeps the app working if PandaScore is dropped.
- The Postgres port is open to the internet. Restrict it in the server firewall to the PC's and server's addresses, and use a long password.
- Lower-tier teams often have few or zero games on record, so many tier C/D predictions will be low-data.
- OpenDota `/live` lags the broadcast by a few minutes; post-draft predictions arrive after the draft ends on stream.
- Bo5 and live markets inherit the independent-maps assumption.
