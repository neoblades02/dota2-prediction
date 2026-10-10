from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

from dota_predictor.logging_config import get_logger
from dota_predictor.modeling.baseline_elo import chronological_split, elo_logit, elo_win_prob, evaluate_elo_predictions
from dota_predictor.settings import get_settings

logger = get_logger("modeling.lightgbm")

DEFAULT_FEATURE_COLUMNS = [
    "elo_diff",
    "elo_radiant",
    "elo_dire",
    "team_games_radiant",
    "team_games_dire",
    "player_elo_diff",
    "player_games_radiant",
    "player_games_dire",
    "form_diff_5",
    "rest_days_diff",
    "lineup_5man_shared_radiant",
    "lineup_5man_shared_dire",
    "pairwise_coplay_diff",
    "hero_breadth_diff",
    "player_hero_comfort_diff",
    "player_hero_exp_diff",
    "draft_hero_winrate_diff",
    "draft_patch_hero_winrate_diff",
    "draft_counter_score",
    "draft_patch_counter_score",
    "draft_synergy_diff",
    "draft_meta_pick_diff",
    "radiant_side_bias",
    "days_since_patch",
    "patch_games_diff",
    "h2h_games",
    "h2h_diff",
]

# Before the draft only team, roster and context signals exist (hero pool breadth needs players, not picks).
PRE_DRAFT_FEATURE_COLUMNS = [
    c for c in DEFAULT_FEATURE_COLUMNS if not (c.startswith("draft_") or c.startswith("player_hero_"))
]

DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "learning_rate": 0.02,
    "num_leaves": 15,
    "min_child_samples": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 5.0,
    "verbosity": -1,
    "seed": 42,
}

# Columns whose sign flips when radiant and dire are swapped (everything named *_diff flips too).
_SIGNED = {"draft_counter_score", "draft_patch_counter_score"}


def mirror(df: pl.DataFrame) -> pl.DataFrame:
    """Swap radiant and dire. Doubling data this way teaches the model that only side bias is side-specific."""
    exprs = []
    for c in df.columns:
        if c.endswith("_radiant") and c[:-8] + "_dire" in df.columns:
            exprs.append(pl.col(c[:-8] + "_dire").alias(c))
        elif c.endswith("_dire") and c[:-5] + "_radiant" in df.columns:
            exprs.append(pl.col(c[:-5] + "_radiant").alias(c))
        elif c.endswith("_diff") or c in _SIGNED:
            exprs.append((-pl.col(c)).alias(c))
        elif c == "target_radiant_win":
            exprs.append((1 - pl.col(c)).alias(c))
        else:
            exprs.append(pl.col(c))
    return df.select(exprs)


def fit_booster(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    init_train: np.ndarray | None = None,
    init_val: np.ndarray | None = None,
    params: dict[str, Any] | None = None,
    feature_names: list[str] | None = None,
    num_boost_round: int = 3000,
    early_stopping_rounds: int = 150,
) -> lgb.Booster:
    """Gradient boosting with early stopping on a validation block that is never the test set."""
    kw = {"feature_name": feature_names} if feature_names else {}
    train_set = lgb.Dataset(X_train, y_train, init_score=init_train, free_raw_data=False, **kw)
    val_set = lgb.Dataset(X_val, y_val, init_score=init_val, reference=train_set, free_raw_data=False)
    return lgb.train(
        {**DEFAULT_PARAMS, **(params or {})},
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )


def predict_booster(booster: lgb.Booster, X: np.ndarray, init: np.ndarray | None = None) -> np.ndarray:
    raw = booster.predict(X, raw_score=True)
    if init is not None:
        raw = raw + init
    return 1.0 / (1.0 + np.exp(-raw))


def predict_artifact(artifact: dict[str, Any], X: np.ndarray, elo_diff: np.ndarray | None = None) -> np.ndarray:
    """Probability of a radiant win from a saved artifact (handles Elo init score and calibration)."""
    init = elo_logit(elo_diff) if artifact.get("elo_init") else None
    p = predict_booster(artifact["model"], X, init)
    calibrator = artifact.get("calibrator")
    return calibrator.predict(p) if calibrator is not None else p


def train_lightgbm_model(
    df: pl.DataFrame,
    feature_cols: list[str] | None = None,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    elo_init: bool = False,
    calibrate: bool = True,
    use_mirror: bool = True,
    pre_draft: bool = False,
    params: dict[str, Any] | None = None,
    output_model_path: Path | None = None,
) -> dict[str, Any]:
    """Train on a chronological split, early-stop and calibrate on validation, report on the test block.

    ``pre_draft`` trains the variant used for upcoming matches (no hero information).
    """
    settings = get_settings()
    default_name = "lightgbm_predraft_v1.joblib" if pre_draft else "lightgbm_pregame_v1.joblib"
    out_path = output_model_path or (settings.models_dir / default_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    default_cols = PRE_DRAFT_FEATURE_COLUMNS if pre_draft else DEFAULT_FEATURE_COLUMNS
    cols = [c for c in (feature_cols or default_cols) if c in df.columns]
    train_df, val_df, test_df = chronological_split(df, test_ratio=test_ratio, val_ratio=val_ratio)
    if use_mirror:
        train_df = pl.concat([train_df, mirror(train_df)])

    def xy(d: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return d.select(cols).to_numpy(), d["target_radiant_win"].to_numpy(), d["elo_diff"].to_numpy()

    X_tr, y_tr, elo_tr = xy(train_df)
    X_val, y_val, elo_val = xy(val_df)
    X_te, y_te, elo_te = xy(test_df)
    init = elo_logit if elo_init else (lambda e: None)

    logger.info("Training LightGBM", train=len(train_df), val=len(val_df), test=len(test_df), features=cols)
    booster = fit_booster(X_tr, y_tr, X_val, y_val, init(elo_tr), init(elo_val), params=params, feature_names=cols)

    calibrator = None
    if calibrate:
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
        calibrator.fit(predict_booster(booster, X_val, init(elo_val)), y_val)

    artifact = {
        "model": booster,
        "features": cols,
        "elo_init": elo_init,
        "pre_draft": pre_draft,
        "calibrator": calibrator,
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "train_rows": len(train_df),
        "best_iteration": booster.best_iteration,
    }
    p_test = predict_artifact(artifact, X_te, elo_te)
    metrics = evaluate_elo_predictions(y_te, p_test)
    metrics_elo = evaluate_elo_predictions(y_te, elo_win_prob(elo_te))
    artifact["metrics_test"] = metrics
    artifact["metrics_elo_baseline"] = metrics_elo
    joblib.dump(artifact, out_path)
    logger.info("Saved trained LightGBM model", path=str(out_path), metrics=metrics)

    importances = dict(zip(cols, booster.feature_importance(importance_type="gain").tolist(), strict=True))
    return {
        "features": cols,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "best_iteration": booster.best_iteration,
        "lightgbm_metrics": metrics,
        "elo_baseline_metrics": metrics_elo,
        "feature_importances": importances,
        "artifact_path": str(out_path),
        "p_test": p_test,
        "y_test": y_te,
    }
