from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import typer

from dota_predictor.features.asof import build_pregame_features
from dota_predictor.ingest.explorer import (
    sync_explorer_match_index,
    sync_explorer_player_matches,
    sync_explorer_teams,
)
from dota_predictor.ingest.opendota import (
    sync_match_details,
    sync_pro_match_index,
    sync_reference_data,
    sync_tournament_matches,
)
from dota_predictor.ingest.valve_patches import sync_valve_patches
from dota_predictor.logging_config import configure_logging, get_logger
from dota_predictor.modeling.abstention import compute_selective_prediction_tiers
from dota_predictor.modeling.evaluate import walk_forward_evaluation
from dota_predictor.modeling.lightgbm_model import train_lightgbm_model
from dota_predictor.quality.coverage import generate_coverage_report
from dota_predictor.settings import get_settings
from dota_predictor.storage.repository import get_engine, init_db

app = typer.Typer(name="dota-predictor", help="Dota 2 Competitive Prediction Engine CLI")
db_app = typer.Typer(name="db", help="Database management commands")
ingest_app = typer.Typer(name="ingest", help="Data ingestion commands")
normalize_app = typer.Typer(name="normalize", help="Entity and match normalization commands")
quality_app = typer.Typer(name="quality", help="Data quality and coverage checks")
features_app = typer.Typer(name="features", help="Point-in-time feature engineering commands")
train_app = typer.Typer(name="train", help="Model training commands")
eval_app = typer.Typer(name="evaluate", help="Model evaluation and backtesting commands")
predict_app = typer.Typer(name="predict", help="Real-time and live draft prediction commands")

app.add_typer(db_app)
app.add_typer(ingest_app)
app.add_typer(normalize_app)
app.add_typer(quality_app)
app.add_typer(features_app)
app.add_typer(train_app)
app.add_typer(eval_app)
app.add_typer(predict_app)

logger = get_logger("cli")


@app.callback()
def main(
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="Logging level (DEBUG, INFO, WARNING, ERROR)"),
) -> None:
    configure_logging(level=log_level)


@db_app.command("init")
def db_init_cmd() -> None:
    """Initialize database tables and default providers."""
    init_db()
    typer.secho("Database initialized successfully.", fg=typer.colors.GREEN)


@ingest_app.command("tournaments")
def ingest_tournaments_cmd(
    tier: str = typer.Option("premium", "--tier", "-t", help="Tournament tier (premium, professional)"),
    min_league_id: int = typer.Option(14000, "--min-id", help="Min league ID (14000 = ~2022)"),
    limit: int | None = typer.Option(None, "--limit", "-l", help="Max tournaments to fetch"),
) -> None:
    """Ingest match lists for tournaments from 2022 to present."""
    engine = get_engine()
    res = sync_tournament_matches(engine, tier=tier, min_league_id=min_league_id, max_tournaments=limit)
    typer.secho(f"Tournament matches sync finished: {res}", fg=typer.colors.GREEN)


@ingest_app.command("pro-index")
def ingest_pro_index_cmd(
    since: str | None = typer.Option("2022-01-01", "--since", help="Cutoff date YYYY-MM-DD"),
    max_pages: int | None = typer.Option(None, "--max-pages", "-n", help="Max pages to fetch"),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Resume from last cursor"),
) -> None:
    """Ingest pro match index from OpenDota /proMatches."""
    since_dt = datetime.strptime(since, "%Y-%m-%d").replace(tzinfo=UTC) if since else None
    engine = get_engine()
    res = sync_pro_match_index(engine, since_utc=since_dt, max_pages=max_pages, resume=resume)
    typer.secho(f"Pro match index sync finished: {res}", fg=typer.colors.GREEN)


@ingest_app.command("match-details")
def ingest_match_details_cmd(
    limit: int | None = typer.Option(100, "--limit", "-l", help="Number of pending matches to fetch"),
) -> None:
    """Fetch pending match details from OpenDota /matches/{match_id}."""
    engine = get_engine()
    res = sync_match_details(engine, limit=limit)
    typer.secho(f"Match details sync finished: {res}", fg=typer.colors.GREEN)


@ingest_app.command("explorer-index")
def ingest_explorer_index_cmd(
    tiers: str = typer.Option("premium,professional", "--tiers", help="Comma-separated OpenDota league tiers"),
    since: str = typer.Option("2022-01-01", "--since", help="Earliest match date YYYY-MM-DD"),
) -> None:
    """Bulk-index league matches via OpenDota /explorer (thousands of matches per request)."""
    since_dt = datetime.strptime(since, "%Y-%m-%d").replace(tzinfo=UTC)
    res = sync_explorer_match_index(get_engine(), tiers=tuple(t.strip() for t in tiers.split(",")), since_utc=since_dt)
    typer.secho(f"Explorer index sync finished: {res}", fg=typer.colors.GREEN)


@ingest_app.command("explorer-players")
def ingest_explorer_players_cmd(
    limit: int | None = typer.Option(None, "--limit", "-l", help="Max pending matches to process"),
    batch_size: int = typer.Option(1000, "--batch-size", help="Matches per explorer request"),
) -> None:
    """Fetch heroes and players for pending index rows via /explorer and normalize them."""
    res = sync_explorer_player_matches(get_engine(), batch_size=batch_size, limit=limit)
    typer.secho(f"Explorer player sync finished: {res}", fg=typer.colors.GREEN)


@ingest_app.command("explorer-teams")
def ingest_explorer_teams_cmd() -> None:
    """Fill canonical team names and tags from OpenDota via /explorer."""
    res = sync_explorer_teams(get_engine())
    typer.secho(f"Explorer team sync finished: {res}", fg=typer.colors.GREEN)


@ingest_app.command("reference")
def ingest_reference_cmd() -> None:
    """Ingest reference datasets: /proPlayers, /teams, /leagues, /heroStats."""
    engine = get_engine()
    res = sync_reference_data(engine)
    typer.secho(f"Reference data sync finished: {res}", fg=typer.colors.GREEN)


@ingest_app.command("patches")
def ingest_patches_cmd() -> None:
    """Ingest official Valve patch feed."""
    engine = get_engine()
    res = sync_valve_patches(engine)
    typer.secho(f"Valve patches sync finished: {res}", fg=typer.colors.GREEN)


@normalize_app.command("all")
def normalize_all_cmd() -> None:
    """Normalize reference data, patches, and match details into canonical entities."""
    from dota_predictor.pipeline import normalize_all

    for step, count in normalize_all(get_engine()).items():
        typer.echo(f"Normalized {count} {step.replace('_', ' ')}.")


@db_app.command("bootstrap")
def db_bootstrap_cmd(
    since: str = typer.Option("2022-01-01", "--since", help="Earliest match date YYYY-MM-DD"),
) -> None:
    """First-time fill of an empty database: reference, patches, every league match since --since, names."""
    from dota_predictor.pipeline import bootstrap

    init_db()
    res = bootstrap(get_engine(), since_utc=datetime.strptime(since, "%Y-%m-%d").replace(tzinfo=UTC))
    typer.secho(f"Bootstrap finished: {res}", fg=typer.colors.GREEN)


@quality_app.command("report")
def quality_report_cmd(
    output_dir: Path | None = typer.Option(None, "--output", "-o", help="Report output folder"),
) -> None:
    """Run data quality checks and emit coverage report."""
    engine = get_engine()
    audit = generate_coverage_report(engine, output_dir=output_dir)

    typer.secho("\n--- Data Quality & Coverage Summary ---", fg=typer.colors.CYAN, bold=True)
    for category, metrics in audit.items():
        if isinstance(metrics, dict):
            typer.secho(f"\n[{category}]", fg=typer.colors.YELLOW)
            for k, v in metrics.items():
                typer.echo(f"  {k}: {v}")
        else:
            typer.echo(f"{category}: {metrics}")


@features_app.command("build")
def features_build_cmd(
    task: str = typer.Option("pregame", "--task", help="Prediction task (pregame)"),
    from_date: str | None = typer.Option(None, "--from", help="Start cutoff YYYY-MM-DD"),
    to_date: str | None = typer.Option(None, "--to", help="End cutoff YYYY-MM-DD"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Parquet output file path"),
) -> None:
    """Build point-in-time feature Parquet dataset."""
    from_dt = datetime.strptime(from_date, "%Y-%m-%d").replace(tzinfo=UTC) if from_date else None
    to_dt = datetime.strptime(to_date, "%Y-%m-%d").replace(tzinfo=UTC) if to_date else None
    engine = get_engine()
    df = build_pregame_features(engine, from_utc=from_dt, to_utc=to_dt, output_path=output)
    typer.secho(
        f"Built pregame feature dataset: {len(df)} rows, {len(df.columns)} columns.",
        fg=typer.colors.GREEN,
    )


def _load_features(data: Path | None) -> pl.DataFrame:
    path = data or (get_settings().features_dir / "pregame_features_v1.parquet")
    if not path.exists():
        typer.secho(f"Feature dataset not found at {path}. Run 'features build' first.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    return pl.read_parquet(path)


@train_app.command("lightgbm")
def train_lightgbm_cmd(
    data: Path | None = typer.Option(None, "--data", "-d", help="Parquet feature dataset path"),
    test_ratio: float = typer.Option(0.2, "--test-ratio", help="Out-of-sample holdout ratio (most recent games)"),
    val_ratio: float = typer.Option(0.1, "--val-ratio", help="Validation block before the test block"),
    elo_init: bool = typer.Option(False, "--elo-init/--no-elo-init", help="Boost residuals on top of Elo log-odds"),
    mirror: bool = typer.Option(True, "--mirror/--no-mirror", help="Radiant/Dire swap augmentation"),
    pre_draft: bool = typer.Option(False, "--pre-draft", help="Train the no-hero variant used for upcoming matches"),
) -> None:
    """Train the LightGBM pregame model (early-stopped and calibrated on a validation block)."""
    df = _load_features(data)
    res = train_lightgbm_model(
        df, test_ratio=test_ratio, val_ratio=val_ratio, elo_init=elo_init, use_mirror=mirror, pre_draft=pre_draft
    )

    typer.secho("\n--- Training Results ---", fg=typer.colors.CYAN, bold=True)
    typer.echo(
        f"Train: {res['train_samples']} | Val: {res['val_samples']} | Test: {res['test_samples']} | "
        f"Best iteration: {res['best_iteration']}"
    )
    typer.secho("\n[LightGBM Model Metrics (Test)]", fg=typer.colors.GREEN)
    for k, v in res["lightgbm_metrics"].items():
        typer.echo(f"  {k}: {v}")
    typer.secho("\n[Elo Baseline Metrics (Same Test Set)]", fg=typer.colors.YELLOW)
    for k, v in res["elo_baseline_metrics"].items():
        typer.echo(f"  {k}: {v}")
    typer.secho("\n[Feature Importances (gain)]", fg=typer.colors.BLUE)
    total = sum(res["feature_importances"].values()) or 1.0
    for f, imp in sorted(res["feature_importances"].items(), key=lambda x: x[1], reverse=True):
        typer.echo(f"  {f:32s}: {imp / total * 100:5.1f}%")
    typer.secho(f"\nTrained artifact saved: {res['artifact_path']}", fg=typer.colors.MAGENTA)


@train_app.command("selective")
def train_selective_cmd(
    data: Path | None = typer.Option(None, "--data", "-d", help="Parquet feature dataset path"),
    test_ratio: float = typer.Option(0.2, "--test-ratio", help="Out-of-sample holdout ratio"),
) -> None:
    """Accuracy, precision, and F1 across confidence tiers (abstention analysis) on the test block."""
    settings = get_settings()
    df = _load_features(data)
    res = train_lightgbm_model(
        df, test_ratio=test_ratio, output_model_path=settings.models_dir / "_selective_tmp.joblib"
    )
    (settings.models_dir / "_selective_tmp.joblib").unlink(missing_ok=True)

    report_path = settings.reports_dir / "selective_prediction_report.json"
    tiers = compute_selective_prediction_tiers(res["y_test"], res["p_test"], output_report_path=report_path)

    typer.secho("\n--- Selective Prediction (Abstention) Analysis ---", fg=typer.colors.CYAN, bold=True)
    typer.echo("Confidence Threshold | Coverage | Matches | Accuracy | Precision | Recall | F1-Score")
    typer.echo("-" * 80)
    for t in tiers:
        typer.echo(
            f"{t['tier_rule']:20s} | {t['coverage_pct']:7.1f}% | {t['samples']:7d} | "
            f"{t['accuracy_pct']:7.2f}% | {t['precision_pct']:8.2f}% | {t['recall_pct']:6.1f}% | "
            f"{t['f1_score_pct']:7.2f}%"
        )
    typer.secho(f"\nReport written to {report_path}", fg=typer.colors.GREEN)


@eval_app.command("walk-forward")
def eval_walk_forward_cmd(
    data: Path | None = typer.Option(None, "--data", "-d", help="Parquet feature dataset path"),
    splits: int = typer.Option(4, "--splits", "-s", help="Number of forward test blocks"),
    mirror: bool = typer.Option(True, "--mirror/--no-mirror", help="Radiant/Dire swap augmentation"),
) -> None:
    """Expanding-window walk-forward comparing Elo, LightGBM, and LightGBM-on-Elo."""
    df = _load_features(data)
    summary = walk_forward_evaluation(df, n_splits=splits, use_mirror=mirror)
    typer.secho("\nWalk-Forward Evaluation Complete!", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"Samples: {summary['total_samples']} across {summary['n_folds']} folds")
    for name, m in summary["mean"].items():
        typer.echo(
            f"  {name:14s} log_loss={m['log_loss']:.4f} brier={m['brier_score']:.4f} acc={m['accuracy'] * 100:.1f}% "
            f"auc={m['roc_auc']:.4f} ece={m['ece']:.4f} | premium acc={m['premium_accuracy'] * 100:.1f}%"
        )


@predict_app.command("draft")
def predict_draft_cmd(
    radiant: str = typer.Option(..., "--radiant", "-r", help="Radiant team name or ID"),
    dire: str = typer.Option(..., "--dire", "-d", help="Dire team name or ID"),
    radiant_heroes: str = typer.Option(
        ..., "--radiant-heroes", "-rh", help="Comma-separated 5 Radiant hero names or IDs"
    ),
    dire_heroes: str = typer.Option(..., "--dire-heroes", "-dh", help="Comma-separated 5 Dire hero names or IDs"),
    radiant_players: str | None = typer.Option(
        None, "--radiant-players", "-rp", help="Comma-separated 5 Radiant players"
    ),
    dire_players: str | None = typer.Option(None, "--dire-players", "-dp", help="Comma-separated 5 Dire players"),
    patch: int | None = typer.Option(None, "--patch", "-p", help="Patch ID (default: latest known patch)"),
) -> None:
    """Predict live win probability and draft counters for an ongoing or completed draft."""
    from dota_predictor.predict.live_draft import LiveDraftPredictor

    r_heroes = [h.strip() for h in radiant_heroes.split(",") if h.strip()]
    d_heroes = [h.strip() for h in dire_heroes.split(",") if h.strip()]
    r_players = [p.strip() for p in radiant_players.split(",") if p.strip()] if radiant_players else None
    d_players = [p.strip() for p in dire_players.split(",") if p.strip()] if dire_players else None

    predictor = LiveDraftPredictor()
    result = predictor.predict_live_draft(
        radiant_team=radiant,
        dire_team=dire,
        radiant_heroes=r_heroes,
        dire_heroes=d_heroes,
        radiant_players=r_players,
        dire_players=d_players,
        patch_id=patch,
    )

    m = result["matchup"]
    d = result["draft"]
    p = result["probabilities"]

    typer.secho("\n=======================================================", fg=typer.colors.CYAN)
    typer.secho("        LIVE PRE-GAME POST-DRAFT FORECAST", fg=typer.colors.CYAN, bold=True)
    typer.secho("=======================================================", fg=typer.colors.CYAN)

    typer.echo(f"\n[Matchup] {m['radiant_team']} (Radiant) vs {m['dire_team']} (Dire)")
    typer.echo(
        f"  Ratings: {m['radiant_team']} ({m['radiant_elo']}) vs "
        f"{m['dire_team']} ({m['dire_elo']}) | Diff: {m['elo_diff']:+.1f}"
    )

    typer.secho(f"\n[Draft Composition - Patch {d['patch_id']}]", fg=typer.colors.YELLOW)
    typer.echo(f"  Radiant ({m['radiant_team']}): {', '.join(d['radiant_heroes'])}")
    typer.echo(f"  Dire    ({m['dire_team']}):    {', '.join(d['dire_heroes'])}")

    typer.echo("\n[Key Counter Matchup Pairings]")
    for c in d["key_matchup_counters"]:
        adv = c["radiant_advantage_pct"]
        adv_str = f"Radiant favored ({adv:+.1f}%)" if adv > 0 else f"Dire favored ({abs(adv):.1f}%)"
        typer.echo(f"  - {c['radiant_hero']} vs {c['dire_hero']}: {adv_str}")

    typer.secho("\n[Model Probability Forecast]", fg=typer.colors.GREEN, bold=True)
    r_name, d_name = m["radiant_team"], m["dire_team"]
    typer.echo(
        f"  Pre-Draft Elo Win Probability:   {r_name} {p['pre_draft_radiant_win_pct']}% | "
        f"{d_name} {100 - p['pre_draft_radiant_win_pct']:.1f}%"
    )
    typer.echo(
        f"  Post-Draft Win Probability:      {r_name} {p['post_draft_radiant_win_pct']}% | "
        f"{d_name} {p['post_draft_dire_win_pct']}%"
    )
    pe_r, pe_d = d["features"]["player_elo_radiant"], d["features"]["player_elo_dire"]
    typer.echo(f"  Lineup (player) Elo:             {r_name} {pe_r:.0f} | {d_name} {pe_d:.0f}")
    swing = p["draft_swing_pct"]
    swing_str = f"+{swing:.1f}% toward Radiant" if swing > 0 else f"{abs(swing):.1f}% toward Dire"
    typer.echo(f"  Model vs team-Elo adjustment:    {swing_str} (roster + draft + context)")
    if "player_comfort_diff_pct" in d:
        p_swing = d["player_comfort_diff_pct"]
        p_str = f"+{p_swing:.1f}% toward Radiant" if p_swing > 0 else f"{abs(p_swing):.1f}% toward Dire"
        typer.echo(f"  Player Signature Comfort:        {p_str}")
    if "radiant_side_bias_pct" in d:
        typer.echo(f"  Map Side Bias (Patch {d['patch_id']}):    Radiant {d['radiant_side_bias_pct']:+.1f}%")
    typer.echo(
        f"  BO3 Series Probability:          {r_name} {p['bo3_series_radiant_pct']}% | "
        f"{d_name} {100 - p['bo3_series_radiant_pct']:.1f}%"
    )
    typer.echo(
        f"  BO5 Series Probability:          {r_name} {p['bo5_series_radiant_pct']}% | "
        f"{d_name} {100 - p['bo5_series_radiant_pct']:.1f}%"
    )
    typer.echo(
        f"  Games known for ratings:         {m['radiant_team']} {m['radiant_games_known']} | "
        f"{m['dire_team']} {m['dire_games_known']} | model={p['model']}"
    )
    top_p = max(p["post_draft_radiant_win_pct"], p["post_draft_dire_win_pct"])
    typer.secho(
        f"\n  Projected Game Winner:           {p['predicted_winner']} ({top_p}%)\n",
        fg=typer.colors.GREEN,
        bold=True,
    )


@predict_app.command("match")
def predict_match_cmd(
    team_a: str = typer.Option(..., "--team-a", "-a", help="First team name, tag, or ID"),
    team_b: str = typer.Option(..., "--team-b", "-b", help="Second team name, tag, or ID"),
    players_a: str | None = typer.Option(
        None, "--players-a", help="Comma-separated 5 players (default: latest lineup)"
    ),
    players_b: str | None = typer.Option(
        None, "--players-b", help="Comma-separated 5 players (default: latest lineup)"
    ),
    best_of: int = typer.Option(3, "--best-of", help="Series length: 1, 2, 3, or 5"),
) -> None:
    """Pre-draft forecast for an upcoming match; side-neutral (averages both radiant/dire orientations)."""
    from dota_predictor.predict.live_draft import LiveDraftPredictor
    from dota_predictor.predict.markets import fair_odds, market_rows, series_markets

    pa = [p.strip() for p in players_a.split(",") if p.strip()] if players_a else None
    pb = [p.strip() for p in players_b.split(",") if p.strip()] if players_b else None
    predictor = LiveDraftPredictor()
    res = predictor.predict_side_neutral(team_a, team_b, players_a=pa, players_b=pb)
    p, p_rad, p_dire, ab = res["p_map"], res["p_as_radiant"], res["p_as_dire"], res["detail"]
    mk = series_markets(p, best_of)
    series = mk["match_winner"]["a"]
    m, d, pr = ab["matchup"], ab["draft"], ab["probabilities"]
    a_name, b_name = m["radiant_team"], m["dire_team"]

    typer.secho(f"\n[Upcoming match] {a_name} vs {b_name}  (Bo{best_of}, {pr['mode']}, model={pr['model']})", bold=True)
    typer.echo(
        f"  Team Elo: {a_name} {m['radiant_elo']:.0f} ({m['radiant_games_known']} games) | "
        f"{b_name} {m['dire_elo']:.0f} ({m['dire_games_known']} games)"
    )
    f = d["features"]
    typer.echo(f"  Lineup Elo: {a_name} {f['player_elo_radiant']:.0f} | {b_name} {f['player_elo_dire']:.0f}")
    typer.echo(f"  {a_name} lineup ({d['radiant_lineup_source']}): {', '.join(str(x) for x in d['radiant_players'])}")
    typer.echo(f"  {b_name} lineup ({d['dire_lineup_source']}): {', '.join(str(x) for x in d['dire_players'])}")
    typer.echo(f"  Head-to-head games: {f['h2h_games']} | form (last 5) diff: {f['form_diff_5']:+.2f}")
    typer.echo(f"  Per-game P({a_name}): {p * 100:.1f}%  [as Radiant {p_rad * 100:.1f}% | as Dire {p_dire * 100:.1f}%]")
    typer.secho(
        f"  Bo{best_of} series: {a_name} {series * 100:.1f}% | {b_name} {(1 - series) * 100:.1f}%  "
        f"-> {a_name if series >= 0.5 else b_name}",
        fg=typer.colors.GREEN,
        bold=True,
    )
    typer.secho(f"\n[Bo{best_of} markets: model probability (fair decimal odds)]", bold=True)
    for label, outcomes in market_rows(mk, a_name, b_name):
        cells = " | ".join(f"{name} {prob * 100:.1f}% ({fair_odds(prob)})" for name, prob in outcomes)
        typer.echo(f"  {label:<18}{cells}")
    if "map1_match" in mk:
        typer.echo("  (Map 2 / Match is identical: maps are modeled as independent)")


@app.command("serve")
def serve_cmd(
    host: str = typer.Option("0.0.0.0", "--host", envvar="HOST", help="Interface to bind"),
    port: int = typer.Option(8000, "--port", envvar="PORT", help="Port to listen on"),
) -> None:
    """Run the dashboard, API and background jobs (schedule, live, results) in one process."""
    import uvicorn

    from dota_predictor.service.app import create_app

    uvicorn.run(create_app(), host=host, port=port, proxy_headers=True, forwarded_allow_ips="*")


if __name__ == "__main__":
    app()
