from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.engine import Engine

from dota_predictor.logging_config import get_logger
from dota_predictor.quality.checks import run_data_quality_audit
from dota_predictor.settings import get_settings

logger = get_logger("quality.coverage")


def generate_coverage_report(engine: Engine, output_dir: Path | None = None) -> dict[str, Any]:
    """Generate and write comprehensive data quality and coverage report."""
    settings = get_settings()
    out_dir = output_dir or settings.reports_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with engine.begin() as conn:
        audit = run_data_quality_audit(conn)

    now_iso = datetime.now(UTC).isoformat()
    audit["generated_at_utc"] = now_iso

    # 1. Write JSON report
    json_path = out_dir / "coverage.json"
    json_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    # 2. Write Markdown report
    lines = [
        "# Dota 2 Competitive Prediction Engine — Data Quality & Coverage Report",
        f"**Generated:** {now_iso}",
        "",
        "## Summary Counts",
        f"- **Indexed Matches:** {audit['counts']['indexed_matches']}",
        f"- **Normalized Games:** {audit['counts']['normalized_games']}",
        f"- **Game Participants:** {audit['counts']['participants']}",
        f"- **Canonical Players:** {audit['counts']['canonical_players']}",
        f"- **Canonical Teams:** {audit['counts']['canonical_teams']}",
        f"- **Lineup Snapshots:** {audit['counts']['lineup_snapshots']}",
        f"- **Patches Tracked:** {audit['counts']['patches']}",
        f"- **Tournaments:** {audit['counts']['tournaments']}",
        "",
        "## Data Integrity & Quality Checks",
        f"- **Duplicate Games:** {audit['quality_metrics']['duplicate_match_ids']} (Acceptance: 0)",
        f"- **Incomplete Lineups (≠10 players):** {audit['quality_metrics']['games_with_non_10_participants']}",
        f"- **Unknown Game Results:** {audit['quality_metrics']['games_with_unknown_results']}",
        f"- **Missing Team Identifiers:** {audit['quality_metrics']['games_with_missing_teams']}",
        f"- **Anonymous Player Slots:** {audit['quality_metrics']['anonymous_participant_slots']}",
        f"- **Games with Draft Data (pick/ban order):** {audit['quality_metrics']['games_with_draft_data']}",
        f"- **Games with Lineups (heroes + players):** {audit['quality_metrics']['games_with_lineups']}",
        "",
        "## Patch Confidence Distribution",
    ]
    for k, v in audit["patch_confidence"].items():
        lines.append(f"- **{k}:** {v}")

    md_path = out_dir / "coverage.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    logger.info("Generated quality reports", json_path=str(json_path), md_path=str(md_path))
    return audit
