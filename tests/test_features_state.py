from datetime import datetime, timedelta

from sqlalchemy import create_engine

from dota_predictor.features.asof import PregameFeatureState
from dota_predictor.features.ratings import EloTracker
from dota_predictor.normalize.patches import resolve_patch_for_match
from dota_predictor.storage.models import patches
from dota_predictor.storage.repository import init_db

T0 = datetime(2026, 1, 1, 12, 0, 0)


def test_player_elo_updates_against_opposing_lineup():
    elo = EloTracker(k_factor=24.0)
    elo.update_groups([1, 2], [3, 4], outcome_a=1.0)
    assert elo.get_rating(1) == elo.get_rating(2) > 1500 > elo.get_rating(3) == elo.get_rating(4)
    assert elo.get_games_count(1) == 1
    # Ratings are zero-sum across the two groups
    assert abs(sum(elo.get_rating(i) for i in (1, 2, 3, 4)) - 6000.0) < 1e-9


def test_features_before_update_are_cold_then_warm():
    state = PregameFeatureState(patch_dates={60: T0 - timedelta(days=10)})
    r = [(1, 10), (2, 11), (3, 12), (4, 13), (5, 14)]
    d = [(6, 20), (7, 21), (8, 22), (9, 23), (10, 24)]
    cold = state.features(T0, 100, 200, r, d, 60, "hr", "hd")
    assert cold["elo_diff"] == 0.0
    assert cold["team_games_radiant"] == 0
    assert cold["hero_breadth_radiant"] == 0.0
    assert cold["days_since_patch"] == 10.0

    state.update(T0, 100, 200, r, d, 60, radiant_win=True, r_lineup_hash="hr", d_lineup_hash="hd")
    warm = state.features(T0 + timedelta(days=1), 100, 200, r, d, 60, "hr", "hd")
    assert warm["elo_diff"] > 0
    assert warm["player_elo_diff"] > 0
    assert warm["team_games_radiant"] == 1
    assert warm["lineup_5man_shared_radiant"] == 1
    assert warm["pairwise_coplay_radiant"] == 1.0
    assert warm["hero_breadth_radiant"] == 1.0
    assert warm["form_diff_5"] == 1.0
    assert warm["patch_games_diff"] == 0


def test_hero_breadth_window_expires_old_heroes():
    state = PregameFeatureState()
    r = [(1, 10), (2, 11), (3, 12), (4, 13), (5, 14)]
    d = [(6, 20), (7, 21), (8, 22), (9, 23), (10, 24)]
    state.update(T0, 100, 200, r, d, None, radiant_win=True)
    assert state.features(T0 + timedelta(days=30), 100, 200, r, d, None)["hero_breadth_radiant"] == 1.0
    assert state.features(T0 + timedelta(days=120), 100, 200, r, d, None)["hero_breadth_radiant"] == 0.0


def test_patch_resolution_prefers_time_over_raw_id(tmp_path):
    engine = create_engine(f"sqlite:///{(tmp_path / 'p.db').as_posix()}")
    init_db(engine)
    with engine.begin() as conn:
        conn.execute(
            patches.insert(),
            [
                {"patch_id": 22, "version": "7.03", "effective_at_utc": datetime(2017, 3, 16)},
                {"patch_id": 60, "version": "7.41", "effective_at_utc": datetime(2026, 3, 24)},
                {"patch_id": 1113, "version": "7.41b", "effective_at_utc": datetime(2026, 4, 7)},
            ],
        )
        # OpenDota's `version` (22) is a parser version, not a patch: time wins.
        assert resolve_patch_for_match(conn, datetime(2026, 4, 10), raw_patch_id=22) == (1113, "inferred")
        assert resolve_patch_for_match(conn, datetime(2026, 3, 30)) == (60, "inferred")
        assert resolve_patch_for_match(conn, None, raw_patch_id=60) == (60, "confirmed")
        assert resolve_patch_for_match(conn, None, raw_patch_id=999) == (None, "unknown")


def _match(i: int, r: int, d: int, radiant_win: bool) -> dict:
    accounts = [r * 10 + k for k in range(5)] + [d * 10 + k for k in range(5)]
    return {
        "match_id": 50000 + i,
        "start_time": 1720000000 + i * 3600,
        "duration": 2000,
        "radiant_win": radiant_win,
        "radiant_team_id": r,
        "dire_team_id": d,
        "players": [
            {"account_id": a, "hero_id": (i + k) % 20 + 1, "player_slot": k, "isRadiant": k < 5}
            for k, a in enumerate(accounts)
        ],
    }


def test_folding_new_games_matches_a_full_replay(tmp_path):
    from dota_predictor.features.asof import fold_new_games, replay_state
    from dota_predictor.normalize.games import normalize_match_detail

    engine = create_engine(f"sqlite:///{(tmp_path / 'fold.db').as_posix()}")
    init_db(engine)
    pairs = [(1, 2), (3, 4), (1, 3), (2, 4), (1, 4), (2, 3)] * 2
    ms = [_match(i, r, d, (i * 7) % 3 != 0) for i, (r, d) in enumerate(pairs)]
    with engine.begin() as conn:
        for m in ms[:6]:
            normalize_match_detail(conn, m)
    live = replay_state(engine)
    assert live.last_game_id > 0
    with engine.begin() as conn:
        for m in ms[6:]:
            normalize_match_detail(conn, m)
    assert fold_new_games(engine, live) == 6
    assert fold_new_games(engine, live) == 0
    full = replay_state(engine)

    now = datetime(2024, 8, 1)
    parts_1 = [(None, None)] * 5
    for r, d in [(1, 2), (3, 4), (1, 4)]:
        a = live.features(now, r, d, parts_1, parts_1, None, None, None)
        b = full.features(now, r, d, parts_1, parts_1, None, None, None)
        assert a == b
