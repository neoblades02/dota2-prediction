import pytest

from dota_predictor.predict.markets import series_markets


@pytest.mark.parametrize("best_of", [1, 2, 3, 5])
@pytest.mark.parametrize("p", [0.5, 0.762, 0.1])
def test_every_format_is_a_distribution(best_of, p):
    m = series_markets(p, best_of)
    assert sum(m["correct_score"].values()) == pytest.approx(1.0)
    assert sum(m["match_winner"].values()) == pytest.approx(1.0)
    for line in m["total_maps"].values():
        assert line["over"] + line["under"] == pytest.approx(1.0)
    if "map1_match" in m:
        assert sum(m["map1_match"].values()) == pytest.approx(1.0)
        assert m["map1_match"]["a/a"] + m["map1_match"]["b/a"] == pytest.approx(m["match_winner"]["a"])


@pytest.mark.parametrize("p", [0.5, 0.762, 0.1])
def test_bo3_closed_forms(p):
    m, q = series_markets(p, 3), 1 - p
    assert m["correct_score"] == pytest.approx({"2-0": p * p, "2-1": 2 * p * p * q, "1-2": 2 * p * q * q, "0-2": q * q})
    assert m["match_winner"]["a"] == pytest.approx(3 * p**2 - 2 * p**3)
    assert m["total_maps"]["2.5"]["over"] == pytest.approx(2 * p * q)
    assert m["handicap"]["a_-1.5"] == pytest.approx(p * p)
    assert m["handicap"]["a_+1.5"] == pytest.approx(1 - q * q)


def test_bo3_even_map1_match():
    m = series_markets(0.5, 3)
    assert m["map1_match"]["a/a"] == pytest.approx(0.375)
    assert m["map1_match"]["a/b"] == pytest.approx(0.125)


def test_bo5_and_bo2():
    p, q = 0.6, 0.4
    bo5 = series_markets(p, 5)
    assert bo5["correct_score"]["3-0"] == pytest.approx(p**3)
    assert bo5["match_winner"]["a"] == pytest.approx(p**3 * (10 - 15 * p + 6 * p**2))
    assert set(bo5["total_maps"]) == {"3.5", "4.5"}
    assert bo5["handicap"]["a_-2.5"] == pytest.approx(p**3)
    bo2 = series_markets(p, 2)
    assert bo2["match_winner"] == pytest.approx({"a": p * p, "draw": 2 * p * q, "b": q * q})
    assert bo2["total_maps"] == {}


def test_conditional_on_live_score():
    p, q = 0.6, 0.4
    assert series_markets(p, 3, 1, 0)["match_winner"]["a"] == pytest.approx(p + q * p)
    assert series_markets(p, 3, 1, 1)["match_winner"]["a"] == pytest.approx(p)
    done = series_markets(p, 3, 2, 0)
    assert done["match_winner"]["a"] == pytest.approx(1.0)
    assert "map1_match" not in series_markets(p, 3, 1, 0)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        series_markets(0.5, 4)
    with pytest.raises(ValueError):
        series_markets(0.5, 3, 3, 0)
