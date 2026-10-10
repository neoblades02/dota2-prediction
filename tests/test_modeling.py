import numpy as np
import polars as pl

from dota_predictor.features.draft import DraftAdvantageEngine
from dota_predictor.modeling.abstention import compute_selective_prediction_tiers
from dota_predictor.modeling.baseline_elo import chronological_split, elo_win_prob, evaluate_elo_predictions
from dota_predictor.modeling.evaluate import walk_forward_evaluation
from dota_predictor.modeling.lightgbm_model import mirror, predict_artifact, train_lightgbm_model


def synthetic_features(n: int = 400, seed: int = 42) -> pl.DataFrame:
    """Outcome driven by elo_diff plus a draft signal the trees must pick up."""
    rng = np.random.default_rng(seed)
    elo = rng.normal(0, 120, n)
    draft = rng.normal(0, 0.05, n)
    p = 1 / (1 + np.exp(-(elo / 150 + draft * 20)))
    y = rng.binomial(1, p)
    return pl.DataFrame(
        {
            "cutoff_at_utc": [
                f"2024-01-01T00:00:{i:02d}" if i < 60 else f"2024-01-01T{i // 60:02d}:{i % 60:02d}:00" for i in range(n)
            ],
            "league_tier": ["premium" if i % 3 == 0 else "professional" for i in range(n)],
            "elo_diff": elo,
            "elo_radiant": 1500 + elo / 2,
            "elo_dire": 1500 - elo / 2,
            "team_games_radiant": rng.integers(0, 50, n),
            "team_games_dire": rng.integers(0, 50, n),
            "form_diff_5": rng.uniform(-0.5, 0.5, n),
            "rest_days_diff": rng.uniform(-5, 5, n),
            "draft_hero_winrate_diff": draft,
            "draft_counter_score": draft * 0.5,
            "days_since_patch": rng.uniform(0, 90, n),
            "target_radiant_win": y,
        }
    )


def test_elo_metrics_evaluation():
    y_true = np.array([1, 0, 1, 1, 0])
    p_pred = np.array([0.8, 0.2, 0.9, 0.7, 0.1])
    res = evaluate_elo_predictions(y_true, p_pred)
    assert res["accuracy"] == 1.0
    assert res["brier_score"] < 0.1
    assert res["log_loss"] < 0.5
    assert 0.0 <= res["ece"] <= 1.0


def test_elo_win_prob_symmetry():
    assert elo_win_prob(0.0) == 0.5
    assert abs(elo_win_prob(200.0) + elo_win_prob(-200.0) - 1.0) < 1e-12


def test_chronological_split_is_ordered():
    df = synthetic_features(100)
    train, val, test = chronological_split(df, test_ratio=0.2, val_ratio=0.1)
    assert (len(train), len(val), len(test)) == (70, 10, 20)
    assert train["cutoff_at_utc"][-1] <= val["cutoff_at_utc"][0] <= test["cutoff_at_utc"][0]


def test_mirror_swaps_sides_and_is_an_involution():
    df = synthetic_features(10)
    m = mirror(df)
    assert m["elo_diff"][0] == -df["elo_diff"][0]
    assert m["elo_radiant"][0] == df["elo_dire"][0]
    assert m["draft_counter_score"][0] == -df["draft_counter_score"][0]
    assert m["target_radiant_win"][0] == 1 - df["target_radiant_win"][0]
    assert m["days_since_patch"][0] == df["days_since_patch"][0]
    assert mirror(m).equals(df)


def test_draft_advantage_engine():
    engine = DraftAdvantageEngine()
    engine.update([1, 2, 3, 4, 5], [6, 7, 8, 9, 10], radiant_win=True)
    feats = engine.compute_draft_features([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
    assert feats["draft_hero_winrate_diff"] > 0
    assert "draft_counter_score" in feats


def test_selective_prediction_tiers():
    y_true = np.array([1, 1, 0, 0, 1, 0])
    p_pred = np.array([0.9, 0.85, 0.1, 0.15, 0.52, 0.48])
    tiers = compute_selective_prediction_tiers(y_true, p_pred, thresholds=[0.0, 0.3])
    assert len(tiers) == 2
    assert tiers[1]["accuracy_pct"] == 100.0


def test_train_lightgbm_beats_elo_on_synthetic_draft_signal(tmp_path):
    df = synthetic_features()
    model_path = tmp_path / "model.joblib"
    res = train_lightgbm_model(df, output_model_path=model_path, params={"min_child_samples": 20})
    assert model_path.exists()
    assert res["lightgbm_metrics"]["log_loss"] < res["elo_baseline_metrics"]["log_loss"]
    assert len(res["p_test"]) == res["test_samples"]

    import joblib

    art = joblib.load(model_path)
    X = df.select(art["features"]).to_numpy()
    p = predict_artifact(art, X, df["elo_diff"].to_numpy())
    assert p.shape == (len(df),)
    assert np.all((p >= 0) & (p <= 1))


def test_walk_forward_writes_reports(tmp_path):
    df = synthetic_features(300)
    summary = walk_forward_evaluation(df, n_splits=2, output_dir=tmp_path / "reports")
    assert summary["n_folds"] == 2
    assert set(summary["mean"]) == {"elo", "lgbm", "lgbm_elo_init"}
    assert "premium" in summary["folds"][0]["elo"]
    assert (tmp_path / "reports" / "model_evaluation.md").exists()
