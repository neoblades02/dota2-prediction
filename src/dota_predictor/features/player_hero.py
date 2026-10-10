from __future__ import annotations


class PlayerHeroMasteryEngine:
    """Tracks historical player signature hero mastery, comfort win rates, and pro experience.

    Measures the 'Collapse on Magnus' or 'SumaiL on Pangolier' effect:
    individual mastery that transcends global hero statistics.
    """

    def __init__(self, prior_weight: float = 4.0):
        self.prior_weight = prior_weight
        # (player_id, hero_id) -> [wins, total_games]
        self.player_hero_stats: dict[tuple[int, int], list[int]] = {}
        # (hero_id, facet_variant) -> [wins, total_games]
        self.facet_stats: dict[tuple[int, int], list[int]] = {}

    def get_player_hero_experience(self, player_id: int | None, hero_id: int) -> int:
        if player_id is None:
            return 0
        stats = self.player_hero_stats.get((player_id, hero_id))
        return stats[1] if stats else 0

    def get_player_hero_winrate(self, player_id: int | None, hero_id: int, base_hero_wr: float = 0.5) -> float:
        if player_id is None:
            return base_hero_wr

        stats = self.player_hero_stats.get((player_id, hero_id))
        if not stats or stats[1] == 0:
            return base_hero_wr

        wins, total = stats
        # Bayesian smoothing toward the hero's base win rate
        return (wins + self.prior_weight * base_hero_wr) / (total + self.prior_weight)

    def get_facet_winrate(self, hero_id: int, variant: int | None, base_hero_wr: float = 0.5) -> float:
        if variant is None:
            return base_hero_wr
        stats = self.facet_stats.get((hero_id, variant))
        if not stats or stats[1] == 0:
            return base_hero_wr
        wins, total = stats
        return (wins + self.prior_weight * base_hero_wr) / (total + self.prior_weight)

    def compute_lineup_comfort(
        self,
        radiant_player_heroes: list[tuple[int | None, int]],
        dire_player_heroes: list[tuple[int | None, int]],
        base_hero_winrates: dict[int, float] | None = None,
    ) -> dict[str, float]:
        """Compute aggregate player signature comfort and pro hero experience differentials."""
        base_wrs = base_hero_winrates or {}

        def calc_comfort(pairs: list[tuple[int | None, int]]) -> tuple[float, float]:
            if not pairs:
                return 0.0, 0.0
            comfort_deltas = []
            exp_counts = []
            for pid, hid in pairs:
                base_wr = base_wrs.get(hid, 0.5)
                p_wr = self.get_player_hero_winrate(pid, hid, base_hero_wr=base_wr)
                comfort_deltas.append(p_wr - base_wr)
                exp_counts.append(self.get_player_hero_experience(pid, hid))

            avg_delta = sum(comfort_deltas) / len(comfort_deltas)
            avg_exp = sum(exp_counts) / len(exp_counts)
            return avg_delta, avg_exp

        r_comfort, r_exp = calc_comfort(radiant_player_heroes)
        d_comfort, d_exp = calc_comfort(dire_player_heroes)

        return {
            "player_hero_comfort_radiant": round(r_comfort, 4),
            "player_hero_comfort_dire": round(d_comfort, 4),
            "player_hero_comfort_diff": round(r_comfort - d_comfort, 4),
            "player_hero_exp_radiant": round(r_exp, 1),
            "player_hero_exp_dire": round(d_exp, 1),
            "player_hero_exp_diff": round(r_exp - d_exp, 1),
        }

    def update(
        self,
        radiant_player_heroes: list[tuple[int | None, int]],
        dire_player_heroes: list[tuple[int | None, int]],
        radiant_win: bool,
    ) -> None:
        """Update player signature hero records strictly AFTER match outcome is known."""
        r_outcome = 1 if radiant_win else 0
        d_outcome = 0 if radiant_win else 1

        for pid, hid in radiant_player_heroes:
            if pid is not None and hid:
                st = self.player_hero_stats.setdefault((pid, hid), [0, 0])
                st[0] += r_outcome
                st[1] += 1

        for pid, hid in dire_player_heroes:
            if pid is not None and hid:
                st = self.player_hero_stats.setdefault((pid, hid), [0, 0])
                st[0] += d_outcome
                st[1] += 1
