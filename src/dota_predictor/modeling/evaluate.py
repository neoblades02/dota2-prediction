from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

from dota_predictor.logging_config import get_logger
from dota_predictor.modeling.baseline_elo import elo_logit, elo_win_prob, evaluate_elo_predictions
from dota_predictor.modeling.lightgbm_model import DEFAULT_FEATURE_COLUMNS, fit_booster, mirror, predict_booster
from dota_predictor.settings import get_settings

logger = get_logger("modeling.evaluate")

MODELS = ("elo", "lgbm", "lgbm_elo_init")


def _fit_predict(
    name: str,
    train: pl.DataFrame,
    val: pl.DataFrame,
    test: pl.DataFrame,
    cols: list[str],
    use_mirror: bool,
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    if name == "elo":
        return elo_win_prob(test["elo_diff"].to_numpy())
    if use_mirror:
        train = pl.concat([train, mirror(train)])
    init = elo_logit if name.endswith("elo_init") else (lambda e: None)
    booster = fit_booster(
        train.select(cols).to_numpy(),
        train["target_radiant_win"].to_numpy(),
        val.select(cols).to_numpy(),
        val["target_radiant_win"].to_numpy(),
        init(train["elo_diff"].to_numpy()),
        init(val["elo_diff"].to_numpy()),
        params=params,
        feature_names=cols,
    )
    cal = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
    cal.fit(
        predict_booster(booster, val.select(cols).to_numpy(), init(val["elo_diff"].to_numpy())),
        val["target_radiant_win"],
    )
    return cal.predict(predict_booster(booster, test.select(cols).to_numpy(), init(test["elo_diff"].to_numpy())))


def walk_forward_evaluation(
    df: pl.DataFrame,
    n_splits: int = 4,
    feature_cols: list[str] | None = None,
    models: tuple[str, ...] = MODELS,
    val_ratio: float = 0.1,
    use_mirror: bool = True,
    params: dict[str, Any] | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Expanding-window walk-forward: each fold trains on everything before its test block.

    The last ``val_ratio`` of each training window is held out for early stopping and
    calibration, so test metrics are never used to pick the model.
    """
    settings = get_settings()
    out_dir = output_dir or settings.reports_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cols = [c for c in (feature_cols or DEFAULT_FEATURE_COLUMNS) if c in df.columns]
    df_sorted = df.sort("cutoff_at_utc")
    n_total = len(df_sorted)
    if n_total < 6:
        raise ValueError(f"Need at least 6 samples for walk-forward evaluation, got {n_total}")

    split_size = n_total // (n_splits + 1)
    results = []
    for fold in range(n_splits):
        train_end = split_size * (fold + 1)
        test_end = n_total if fold == n_splits - 1 else split_size * (fold + 2)
        n_val = max(1, int(train_end * val_ratio))
        train = df_sorted[: train_end - n_val]
        val = df_sorted[train_end - n_val : train_end]
        test = df_sorted[train_end:test_end]
        y_test = test["target_radiant_win"].to_numpy()
        premium = (test["league_tier"] == "premium").to_numpy() if "league_tier" in test.columns else None

        fold_res: dict[str, Any] = {
            "fold": fold + 1,
            "train_start": str(train["cutoff_at_utc"][0]),
            "test_start": str(test["cutoff_at_utc"][0]),
            "test_end": str(test["cutoff_at_utc"][-1]),
            "train_size": len(train),
            "val_size": len(val),
            "test_size": len(test),
        }
        for name in models:
            p = _fit_predict(name, train, val, test, cols, use_mirror, params)
            fold_res[name] = evaluate_elo_predictions(y_test, p)
            if premium is not None and premium.sum() > 0:
                fold_res[name]["premium"] = evaluate_elo_predictions(y_test[premium], p[premium])
        results.append(fold_res)
        logger.info("Walk-forward fold complete", fold=fold + 1, **{m: fold_res[m]["log_loss"] for m in models})

    def mean_metric(name: str, key: str, premium: bool = False) -> float:
        vals = [
            (r[name]["premium"][key] if premium else r[name][key])
            for r in results
            if not premium or "premium" in r[name]
        ]
        return round(float(np.mean(vals)), 4) if vals else float("nan")

    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "total_samples": n_total,
        "n_folds": len(results),
        "features": cols,
        "mirror_augmentation": use_mirror,
        "folds": results,
        "mean": {
            name: {
                "log_loss": mean_metric(name, "log_loss"),
                "brier_score": mean_metric(name, "brier_score"),
                "accuracy": mean_metric(name, "accuracy"),
                "roc_auc": mean_metric(name, "roc_auc"),
                "ece": mean_metric(name, "ece"),
                "premium_log_loss": mean_metric(name, "log_loss", premium=True),
                "premium_accuracy": mean_metric(name, "accuracy", premium=True),
            }
            for name in models
        },
    }
    (out_dir / "model_evaluation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Dota 2 Competitive Prediction Engine — Walk-Forward Model Evaluation",
        f"**Generated:** {summary['generated_at_utc']}",
        f"**Total Samples:** {n_total} | **Folds:** {len(results)} | **Mirror augmentation:** {use_mirror}",
        "",
        "## Mean over folds",
        "",
        "| Model | Log Loss | Brier | Accuracy | ROC AUC | ECE | Premium Log Loss | Premium Acc |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, m in summary["mean"].items():
        lines.append(
            f"| {name} | {m['log_loss']:.4f} | {m['brier_score']:.4f} | {m['accuracy'] * 100:.1f}% | "
            f"{m['roc_auc']:.4f} | {m['ece']:.4f} | {m['premium_log_loss']:.4f} | {m['premium_accuracy'] * 100:.1f}% |"
        )
    lines += [
        "",
        "## Per fold (log loss / accuracy)",
        "",
        "| Fold | Test Window | Test N | " + " | ".join(models) + " |",
        "|---|---|---|" + "---|" * len(models),
    ]
    for r in results:
        cells = " | ".join(f"{r[m]['log_loss']:.4f} / {r[m]['accuracy'] * 100:.1f}%" for m in models)
        lines.append(f"| {r['fold']} | {r['test_start'][:10]} to {r['test_end'][:10]} | {r['test_size']} | {cells} |")
    (out_dir / "model_evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Walk-forward evaluation complete", report=str(out_dir / "model_evaluation.md"))
    return summary
