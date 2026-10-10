"""Series betting markets from a single per-map win probability.

Scores are from team A's side ("2-1" = A won two maps). Works for Bo1/Bo2/Bo3/Bo5
and from any live score, by enumerating the remaining maps.

ponytail: maps are modeled as independent with a constant per-map p; a
map-level model (side pick, momentum, draft) would replace the enumeration.
"""

from __future__ import annotations

FORMATS = (1, 2, 3, 5)


def _finished(a: int, b: int, best_of: int) -> bool:
    return a + b == 2 if best_of == 2 else max(a, b) == best_of // 2 + 1


def final_scores(p: float, best_of: int, score_a: int = 0, score_b: int = 0) -> dict[tuple[int, int], float]:
    if best_of not in FORMATS:
        raise ValueError(f"Unsupported series length: Bo{best_of}")
    limit = 2 if best_of == 2 else best_of // 2 + 1
    if not (0 <= score_a <= limit and 0 <= score_b <= limit) or (best_of == 2 and score_a + score_b > 2):
        raise ValueError(f"Impossible score {score_a}-{score_b} for Bo{best_of}")
    if best_of != 2 and score_a == limit and score_b == limit:
        raise ValueError(f"Impossible score {score_a}-{score_b} for Bo{best_of}")
    dist: dict[tuple[int, int], float] = {}
    stack = [(score_a, score_b, 1.0)]
    while stack:
        a, b, prob = stack.pop()
        if _finished(a, b, best_of):
            dist[(a, b)] = dist.get((a, b), 0.0) + prob
        else:
            stack += [(a + 1, b, prob * p), (a, b + 1, prob * (1.0 - p))]
    return dist


def series_markets(p: float, best_of: int, score_a: int = 0, score_b: int = 0) -> dict[str, dict]:
    """Market -> outcome -> probability for team A with per-map win probability ``p``."""
    dist = final_scores(p, best_of, score_a, score_b)
    by_score = sorted(dist.items(), key=lambda kv: (kv[0][1] - kv[0][0], -kv[0][0]))
    out: dict[str, dict] = {
        "match_winner": {
            "a": sum(v for (a, b), v in dist.items() if a > b),
            "b": sum(v for (a, b), v in dist.items() if b > a),
        },
        "correct_score": {f"{a}-{b}": v for (a, b), v in by_score},
        "total_maps": {},
        "handicap": {},
    }
    if best_of == 2:
        out["match_winner"]["draw"] = sum(v for (a, b), v in dist.items() if a == b)
    if best_of >= 3:
        totals = range(best_of // 2 + 1, best_of + 1)
        for t in list(totals)[:-1]:
            over = sum(v for (a, b), v in dist.items() if a + b > t)
            out["total_maps"][f"{t}.5"] = {"over": over, "under": 1.0 - over}
        for h in range(1, best_of // 2 + 1):
            a_cover = sum(v for (a, b), v in dist.items() if a - b > h)
            b_cover = sum(v for (a, b), v in dist.items() if b - a > h)
            out["handicap"] |= {
                f"a_-{h}.5": a_cover,
                f"b_+{h}.5": 1.0 - a_cover,
                f"b_-{h}.5": b_cover,
                f"a_+{h}.5": 1.0 - b_cover,
            }
        if score_a == score_b == 0:
            after_a = series_markets(p, best_of, 1, 0)["match_winner"]["a"]
            after_b = series_markets(p, best_of, 0, 1)["match_winner"]["a"]
            out["map1_match"] = {
                "a/a": p * after_a,
                "a/b": p * (1.0 - after_a),
                "b/a": (1.0 - p) * after_b,
                "b/b": (1.0 - p) * (1.0 - after_b),
            }
    return out


def fair_odds(prob: float) -> float:
    return round(1.0 / prob, 2) if prob > 0 else float("inf")


def market_rows(mk: dict[str, dict], a_name: str, b_name: str) -> list[tuple[str, list[tuple[str, float]]]]:
    """Human-readable (label, [(outcome, probability)]) rows, in bookmaker order."""
    names = {"a": a_name, "b": b_name, "draw": "Draw"}
    rows = [("Match winner:", [(names[k], v) for k, v in mk["match_winner"].items()])]
    for line, ou in mk["total_maps"].items():
        rows.append((f"Total maps {line}:", [("Over", ou["over"]), ("Under", ou["under"])]))
    rows.append(("Correct score:", [(_score_label(s, a_name, b_name), v) for s, v in mk["correct_score"].items()]))
    if mk["handicap"]:
        rows.append(("Map handicap:", [(f"{names[k[0]]} {k[2:]}", v) for k, v in mk["handicap"].items()]))
    if "map1_match" in mk:
        rows.append(("Map 1 / Match:", [(f"{names[k[0]]}/{names[k[2]]}", v) for k, v in mk["map1_match"].items()]))
    return rows


def _score_label(score: str, a_name: str, b_name: str) -> str:
    a, b = map(int, score.split("-"))
    if a == b:
        return f"Draw {a}-{b}"
    return f"{a_name} {a}-{b}" if a > b else f"{b_name} {b}-{a}"
