"""Canonical schema (SQLAlchemy Core).

Single source of truth for the database. `db init` runs metadata.create_all
plus a schema_migrations ledger. Portable types only (no JSONB) so the same
schema works on SQLite (dev) and PostgreSQL (DOTA_DATABASE_URL).

Temporal columns follow the blueprint convention:
  *_utc event time | known_at | recorded_at | source | confidence
"""

from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Table,
    Text,
    UniqueConstraint,
)

metadata = MetaData()

SCHEMA_VERSION = "0002_schedule"

providers = Table(
    "providers",
    metadata,
    Column("provider_id", Integer, primary_key=True),
    Column("name", Text, nullable=False, unique=True),
    Column("policy_version", Text),
    Column("notes", Text),
)

raw_ingest_objects = Table(
    "raw_ingest_objects",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("provider", Text, nullable=False),
    Column("endpoint", Text, nullable=False),
    Column("request_key", Text, nullable=False),
    Column("fetched_at_utc", DateTime(timezone=True), nullable=False),
    Column("payload_path", Text, nullable=False),
    Column("content_hash", Text, nullable=False),
    Column("http_status", Integer),
    Column("schema_version", Text),
    UniqueConstraint("provider", "endpoint", "request_key", "content_hash", name="uq_raw"),
    Index("ix_raw_provider_endpoint", "provider", "endpoint"),
)

sync_cursors = Table(
    "sync_cursors",
    metadata,
    Column("provider", Text, primary_key=True),
    Column("resource", Text, primary_key=True),
    Column("cursor", Text),
    Column("state", Text, nullable=False, default="idle"),  # idle|running|done|error
    Column("last_success_utc", DateTime(timezone=True)),
    Column("retries", Integer, nullable=False, default=0),
    Column("error", Text),
)

# --- Shallow pro-match index from /proMatches -------------------------------

pro_match_index = Table(
    "pro_match_index",
    metadata,
    Column("match_id", BigInteger, primary_key=True),
    Column("start_time_utc", DateTime(timezone=True)),
    Column("duration_s", Integer),
    Column("radiant_team_id", Integer),
    Column("radiant_name", Text),
    Column("dire_team_id", Integer),
    Column("dire_name", Text),
    Column("leagueid", Integer),
    Column("league_name", Text),
    Column("series_id", BigInteger),
    Column("series_type", Integer),
    Column("radiant_score", Integer),
    Column("dire_score", Integer),
    Column("radiant_win", Boolean),
    Column("version", Text),
    Column("detail_status", Text, nullable=False, default="pending"),  # pending|fetched|unavailable|quarantined
    Column("detail_error", Text),
    Column("first_seen_utc", DateTime(timezone=True), nullable=False),
    Column("last_seen_utc", DateTime(timezone=True), nullable=False),
    Index("ix_pmi_start", "start_time_utc"),
    Index("ix_pmi_detail_status", "detail_status"),
    Index("ix_pmi_series", "series_id"),
)

# --- Canonical entities -----------------------------------------------------

teams = Table(
    "teams",
    metadata,
    Column("team_id", Integer, primary_key=True),
    Column("canonical_name", Text, nullable=False),
    Column("tag", Text),
    Column("created_utc", DateTime(timezone=True), nullable=False),
)

team_aliases = Table(
    "team_aliases",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("team_id", Integer, ForeignKey("teams.team_id"), nullable=False),
    Column("provider", Text, nullable=False),
    Column("provider_team_id", Text, nullable=False),
    Column("name", Text),
    Column("first_seen_utc", DateTime(timezone=True), nullable=False),
    Column("last_seen_utc", DateTime(timezone=True), nullable=False),
    UniqueConstraint("provider", "provider_team_id", name="uq_team_alias"),
)

players = Table(
    "players",
    metadata,
    Column("player_id", Integer, primary_key=True),
    Column("canonical_name", Text),
    # Steam32 account ID is the stable cross-provider identity. Nullable:
    # anonymous/hidden accounts (account_id 0 or missing) get no link.
    Column("steam32_account_id", BigInteger, unique=True),
    Column("metadata_confidence", Text, default="inferred"),
    Column("created_utc", DateTime(timezone=True), nullable=False),
)

player_aliases = Table(
    "player_aliases",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("player_id", Integer, ForeignKey("players.player_id"), nullable=False),
    Column("provider", Text, nullable=False),
    Column("provider_player_id", Text, nullable=False),
    Column("alias", Text),
    Column("first_seen_utc", DateTime(timezone=True), nullable=False),
    Column("last_seen_utc", DateTime(timezone=True), nullable=False),
    UniqueConstraint("provider", "provider_player_id", name="uq_player_alias"),
)

tournaments = Table(
    "tournaments",
    metadata,
    Column("tournament_id", Integer, primary_key=True),
    Column("leagueid", Integer, unique=True),  # Valve/OpenDota league id
    Column("title", Text, nullable=False),
    Column("tier", Text),  # from OpenDota leagues tier field; not independently verified
    Column("region", Text),
    Column("organizer", Text),
    Column("start_utc", DateTime(timezone=True)),
    Column("end_utc", DateTime(timezone=True)),
    Column("metadata_source", Text, default="opendota"),
)

SERIES_TYPE_TO_BEST_OF = {0: 1, 1: 3, 2: 5}  # Valve series_type encoding

series = Table(
    "series",
    metadata,
    Column("series_id", Integer, primary_key=True),
    Column("valve_series_id", BigInteger, unique=True),  # nullable: some games unmapped
    Column("tournament_id", Integer, ForeignKey("tournaments.tournament_id")),
    Column("team_a_id", Integer, ForeignKey("teams.team_id")),
    Column("team_b_id", Integer, ForeignKey("teams.team_id")),
    Column("best_of", Integer),
    Column("series_type_raw", Integer),
    Column("scheduled_start_utc", DateTime(timezone=True)),
    Column("status", Text, default="unknown"),  # scheduled|in_progress|completed|postponed|canceled|unknown
    Column("winner_team_id", Integer, ForeignKey("teams.team_id")),
    Column("source", Text, default="opendota"),
    Column("confidence", Text, default="inferred"),  # confirmed|inferred|unknown
)

games = Table(
    "games",
    metadata,
    Column("game_id", Integer, primary_key=True),
    Column("valve_match_id", BigInteger, nullable=False, unique=True),
    Column("series_id", Integer, ForeignKey("series.series_id")),
    Column("tournament_id", Integer, ForeignKey("tournaments.tournament_id")),
    Column("radiant_team_id", Integer, ForeignKey("teams.team_id")),
    Column("dire_team_id", Integer, ForeignKey("teams.team_id")),
    Column("start_utc", DateTime(timezone=True)),
    Column("duration_s", Integer),
    Column("radiant_win", Boolean),
    Column("patch_id", Integer, ForeignKey("patches.patch_id")),
    Column("patch_confidence", Text, default="unknown"),  # confirmed|inferred|unknown
    Column("parse_status", Text, default="parsed"),  # parsed|unparsed|partial|quarantined
    Column("source", Text, default="opendota"),
    Column("raw_object_id", Integer, ForeignKey("raw_ingest_objects.id")),
    Column("ingested_at_utc", DateTime(timezone=True), nullable=False),
    Index("ix_games_start", "start_utc"),
    Index("ix_games_series", "series_id"),
)

game_participants = Table(
    "game_participants",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("game_id", Integer, ForeignKey("games.game_id"), nullable=False),
    Column("player_id", Integer, ForeignKey("players.player_id")),
    Column("account_id", BigInteger),  # raw provider value kept for audit
    Column("hero_id", Integer),
    Column("side", Text),  # radiant|dire
    Column("team_id", Integer, ForeignKey("teams.team_id")),
    Column("player_slot", Integer),
    Column("role_estimate", Integer),  # 1-5, only when supported
    Column("role_confidence", Text, default="unknown"),
    Column("standin_flag", Boolean, default=False),
    Column("name_snapshot", Text),
    Column("personaname_snapshot", Text),
    UniqueConstraint("game_id", "player_slot", name="uq_participant_slot"),
    Index("ix_gp_player", "player_id"),
    Index("ix_gp_game", "game_id"),
)

draft_actions = Table(
    "draft_actions",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("game_id", Integer, ForeignKey("games.game_id"), nullable=False),
    Column("order_seq", Integer, nullable=False),
    Column("is_pick", Boolean, nullable=False),
    Column("hero_id", Integer, nullable=False),
    Column("side", Text),  # radiant|dire (team 0/1 raw)
    Column("team_number_raw", Integer),
    Column("source", Text, default="opendota"),
    UniqueConstraint("game_id", "order_seq", name="uq_draft_order"),
)

lineup_snapshots = Table(
    "lineup_snapshots",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("game_id", Integer, ForeignKey("games.game_id"), nullable=False),
    Column("team_id", Integer, ForeignKey("teams.team_id"), nullable=False),
    # sorted canonical player_id list, JSON-encoded; hash for fast equality
    Column("player_ids_json", Text, nullable=False),
    Column("player_set_hash", Text, nullable=False),
    Column("roles_json", Text),
    Column("substitutions", Integer, default=0),
    Column("confidence", Text, default="inferred"),
    UniqueConstraint("game_id", "team_id", name="uq_lineup"),
    Index("ix_lineup_hash", "player_set_hash"),
)

roster_events = Table(
    "roster_events",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("team_id", Integer, ForeignKey("teams.team_id"), nullable=False),
    Column("player_id", Integer, ForeignKey("players.player_id"), nullable=False),
    Column("event_type", Text, nullable=False),  # join|leave|standin|role_swap|unknown
    Column("announced_at_utc", DateTime(timezone=True)),
    Column("effective_at_utc", DateTime(timezone=True)),
    Column("known_at_utc", DateTime(timezone=True)),
    Column("recorded_at_utc", DateTime(timezone=True), nullable=False),
    Column("source", Text),
    Column("confidence", Text, default="inferred"),
)

patches = Table(
    "patches",
    metadata,
    Column("patch_id", Integer, primary_key=True),  # Valve patch number (e.g. 60)
    Column("version", Text),  # e.g. "7.39e" — from dotaconstants/patch feed
    Column("published_at_utc", DateTime(timezone=True)),
    Column("effective_at_utc", DateTime(timezone=True)),
    Column("source", Text),
    Column("confidence", Text, default="inferred"),
)

patch_changes = Table(
    "patch_changes",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("patch_id", Integer, ForeignKey("patches.patch_id"), nullable=False),
    Column("object_type", Text),  # hero|item|general
    Column("object_id", Text),
    Column("change_text", Text),
)

hero_reference = Table(
    "hero_reference",
    metadata,
    Column("hero_id", Integer, primary_key=True),
    Column("name", Text),  # npc_dota_hero_* internal name
    Column("localized_name", Text),
    Column("first_seen_utc", DateTime(timezone=True)),
    Column("last_seen_utc", DateTime(timezone=True)),
)

# --- Forward pipeline entities (fixtures, features, predictions) ------------

fixture_snapshots = Table(
    "fixture_snapshots",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("fixture_key", Text, nullable=False),
    Column("observed_at_utc", DateTime(timezone=True), nullable=False),
    Column("scheduled_start_utc", DateTime(timezone=True)),
    Column("expected_teams_json", Text),
    Column("expected_lineups_json", Text),
    Column("source", Text),
    Index("ix_fixture_key", "fixture_key"),
)

feature_snapshots = Table(
    "feature_snapshots",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("target_type", Text, nullable=False),  # game|series
    Column("target_id", Text, nullable=False),
    Column("cutoff_at_utc", DateTime(timezone=True), nullable=False),
    Column("feature_version", Text, nullable=False),
    Column("feature_payload", Text, nullable=False),  # JSON
    Column("feature_hash", Text, nullable=False),
    Column("source_watermark", Text),
    Column("created_at_utc", DateTime(timezone=True), nullable=False),
    UniqueConstraint("target_type", "target_id", "cutoff_at_utc", "feature_version", name="uq_feature"),
)

predictions = Table(
    "predictions",
    metadata,
    Column("prediction_id", Integer, primary_key=True),
    Column("prediction_key", Text, nullable=False, unique=True),
    Column("target_type", Text, nullable=False),  # game|series
    Column("target_ref", Text, nullable=False),
    Column("prediction_type", Text, nullable=False),  # pre_match|pre_series|post_draft
    Column("created_at_utc", DateTime(timezone=True), nullable=False),
    Column("cutoff_at_utc", DateTime(timezone=True), nullable=False),
    Column("model_version", Text, nullable=False),
    Column("feature_version", Text),
    Column("p_team_a", Float),
    Column("team_a_id", Integer),
    Column("team_b_id", Integer),
    Column("roster_confidence", Text),
    Column("payload_json", Text),  # full immutable record
    Index("ix_pred_target", "target_type", "target_ref"),
)

prediction_results = Table(
    "prediction_results",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("prediction_id", Integer, ForeignKey("predictions.prediction_id"), nullable=False),
    Column("result_status", Text, nullable=False),  # won|lost|void|canceled|unresolved
    Column("realized_outcome", Integer),  # 1 = team_a won
    Column("evaluated_at_utc", DateTime(timezone=True)),
    Column("settled_source", Text),
    UniqueConstraint("prediction_id", name="uq_result_pred"),
)

model_runs = Table(
    "model_runs",
    metadata,
    Column("run_id", Integer, primary_key=True),
    Column("model_version", Text, nullable=False),
    Column("task", Text),
    Column("train_cutoff_utc", DateTime(timezone=True)),
    Column("test_start_utc", DateTime(timezone=True)),
    Column("test_end_utc", DateTime(timezone=True)),
    Column("dataset_hash", Text),
    Column("code_commit", Text),
    Column("params_json", Text),
    Column("metrics_json", Text),
    Column("artifact_path", Text),
    Column("created_at_utc", DateTime(timezone=True), nullable=False),
)


# --- Live schedule (PandaScore + OpenDota live) ------------------------------
# The prediction log reuses `predictions` (target_type="fixture") and grading
# reuses `prediction_results`.

schedule_tournaments = Table(
    "schedule_tournaments",
    metadata,
    Column("tournament_key", Text, primary_key=True),  # ps:<pandascore tournament id>
    Column("league_name", Text),
    Column("serie_name", Text),
    Column("stage_name", Text),
    Column("tier", Text),  # pandascore s|a|b|c|d
    Column("begin_at_utc", DateTime(timezone=True)),
    Column("end_at_utc", DateTime(timezone=True)),
    Column("status", Text, nullable=False),  # upcoming|running|finished
    Column("updated_at_utc", DateTime(timezone=True), nullable=False),
)

fixtures = Table(
    "fixtures",
    metadata,
    Column("fixture_id", Text, primary_key=True),  # ps:<match id> | od:<series id> | od:m<match id>
    Column("source", Text, nullable=False),  # pandascore|opendota
    Column("tournament_key", Text),
    Column("league_name", Text),
    Column("leagueid", Integer),  # OpenDota league, when known
    Column("tier", Text),
    Column("begin_at_utc", DateTime(timezone=True)),
    Column("best_of", Integer, nullable=False),
    Column("team_a_id", Integer),  # our team ids; null while TBD or unlinked
    Column("team_b_id", Integer),
    Column("team_a_name", Text),
    Column("team_b_name", Text),
    Column("ps_team_a_id", BigInteger),
    Column("ps_team_b_id", BigInteger),
    Column("status", Text, nullable=False),  # upcoming|running|finished|canceled
    Column("score_a", Integer, nullable=False, default=0),
    Column("score_b", Integer, nullable=False, default=0),
    Column("winner", Text),  # a|b|draw
    Column("od_series_id", BigInteger),
    Column("updated_at_utc", DateTime(timezone=True), nullable=False),
    Index("ix_fixtures_status_begin", "status", "begin_at_utc"),
    Index("ix_fixtures_od_series", "od_series_id"),
)

team_links = Table(
    "team_links",
    metadata,
    Column("ps_team_id", BigInteger, primary_key=True),
    Column("ps_name", Text),
    Column("ps_acronym", Text),
    Column("team_id", Integer),  # null = unlinked
    Column("method", Text),  # exact_name|exact_tag|core_name|manual
    Column("suggestions_json", Text),  # [{"team_id":..,"name":..,"games":..}]
    Column("updated_at_utc", DateTime(timezone=True), nullable=False),
)
