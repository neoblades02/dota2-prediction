from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from dota_predictor.logging_config import get_logger

logger = get_logger("modeling.abstention")


def compute_selective_prediction_tiers(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    thresholds: list[float] | None = None,
    output_report_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Evaluate accuracy, precision, recall, and F1 across confidence tiers (abstention).

    Confidence metric: tau = |p - 0.5|
    """
    taus = thresholds or [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    results = []

    for tau in taus:
        mask = np.abs(p_pred - 0.5) >= tau
        coverage = float(np.mean(mask) * 100.0)
        n_samples = int(np.sum(mask))

        if n_samples > 0:
            y_sub = y_true[mask]
            p_sub = (p_pred[mask] >= 0.5).astype(int)

            acc = float(accuracy_score(y_sub, p_sub) * 100.0)
            prec = float(precision_score(y_sub, p_sub, zero_division=0) * 100.0)
            rec = float(recall_score(y_sub, p_sub, zero_division=0) * 100.0)
            f1 = float(f1_score(y_sub, p_sub, zero_division=0) * 100.0)
        else:
            acc, prec, rec, f1 = 0.0, 0.0, 0.0, 0.0

        p_min = round(0.5 + tau, 2)
        p_max = round(0.5 - tau, 2)
        tier_label = f"P >= {p_min:.2f} or P <= {p_max:.2f}" if tau > 0 else "All Matches (tau=0)"

        results.append(
            {
                "tau": tau,
                "tier_rule": tier_label,
                "coverage_pct": round(coverage, 1),
                "samples": n_samples,
                "accuracy_pct": round(acc, 2),
                "precision_pct": round(prec, 2),
                "recall_pct": round(rec, 2),
                "f1_score_pct": round(f1, 2),
            }
        )

    if output_report_path:
        output_report_path.parent.mkdir(parents=True, exist_ok=True)
        output_report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        logger.info("Saved selective prediction report", path=str(output_report_path))

    return results
