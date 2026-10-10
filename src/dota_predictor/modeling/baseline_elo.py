from __future__ import annotations

import math

import numpy as np
import polars as pl
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ELO_SCALE = 400.0


def elo_win_prob(elo_diff: np.ndarray | float) -> np.ndarray | float:
    """Classic Elo expectation: P(radiant) from rating difference."""
    return 1.0 / (1.0 + np.power(10.0, -np.asarray(elo_diff, dtype=float) / ELO_SCALE))


def elo_logit(elo_diff: np.ndarray | float) -> np.ndarray:
    """log-odds of ``elo_win_prob`` — used as a boosting init score so trees learn the residual."""
    return (math.log(10.0) / ELO_SCALE) * np.asarray(elo_diff, dtype=float)


def expected_calibration_error(y_true: np.ndarray, p_pred: np.ndarray, n_bins: int = 10) -> float:
    """Mean |observed rate - predicted probability| over equal-width bins, weighted by bin size."""
    bins = np.minimum((p_pred * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(y_true[mask].mean() - p_pred[mask].mean())
    return float(ece)


def evaluate_elo_predictions(y_true: np.ndarray, p_pred: np.ndarray) -> dict[str, float]:
    """Proper scoring metrics for probability forecasts (works for any model, not just Elo)."""
    y_true = np.asarray(y_true)
    p_pred = np.asarray(p_pred, dtype=float)
    p_clipped = np.clip(p_pred, 1e-6, 1.0 - 1e-6)
    auc = float(roc_auc_score(y_true, p_pred)) if len(np.unique(y_true)) > 1 else 0.5
    return {
        "log_loss": round(float(log_loss(y_true, p_clipped, labels=[0, 1])), 4),
        "brier_score": round(float(brier_score_loss(y_true, p_pred)), 4),
        "accuracy": round(float(np.mean((p_pred >= 0.5) == (y_true == 1))), 4),
        "roc_auc": round(auc, 4),
        "ece": round(expected_calibration_error(y_true, p_pred), 4),
        "samples": int(len(y_true)),
    }


def chronological_split(
    df: pl.DataFrame,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Sort by cutoff and cut train / validation / test as contiguous time blocks (oldest first)."""
    df = df.sort("cutoff_at_utc")
    n = len(df)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_test - n_val
    if n_train < 2 or n_val < 1 or n_test < 1:
        raise ValueError(f"Insufficient data to split: {n} rows")
    return df[:n_train], df[n_train : n_train + n_val], df[n_train + n_val :]
