from __future__ import annotations

import itertools


class DraftAdvantageEngine:
    """Computes point-in-time, patch-aware hero counter, synergy, and meta advantages.

    Tracks hero performance both globally and conditionally per patch, with Bayesian
    shrinkage across patch transitions to handle cold-starts gracefully.
    """

    def __init__(self, smoothing: float = 10.0, patch_prior_weight: float = 5.0):
        self.smoothing = smoothing
        self.patch_prior_weight = patch_prior_weight

        # Global stats
        self.hero_stats: dict[int, list[int]] = {}  # hero_id -> [wins, total]
        self.counter_stats: dict[tuple[int, int], list[int]] = {}  # (hero_a, hero_b) -> [wins_for_a, total]
        self.synergy_stats: dict[tuple[int, int], list[int]] = {}  # (min(h1,h2), max(h1,h2)) -> [wins, total]

        # Patch-specific stats
        self.patch_hero_stats: dict[tuple[int, int], list[int]] = {}  # (patch_id, hero_id) -> [wins, total]
        self.patch_counter_stats: dict[tuple[int, int, int], list[int]] = {}  # (patch_id, h_a, h_b) -> [wins, total]
        self.patch_synergy_stats: dict[tuple[int, int, int], list[int]] = {}  # (patch_id, h1, h2) -> [wins, total]
        self.patch_total_games: dict[int, int] = {}

    def get_global_hero_winrate(self, hero_id: int) -> float:
        stats = self.hero_stats.get(hero_id)
        if not stats or stats[1] == 0:
            return 0.5
        wins, total = stats
        return (wins + self.smoothing * 0.5) / (total + self.smoothing)

    def get_patch_hero_winrate(self, hero_id: int, patch_id: int | None) -> float:
        global_wr = self.get_global_hero_winrate(hero_id)
        if patch_id is None:
            return global_wr

        p_stats = self.patch_hero_stats.get((patch_id, hero_id))
        if not p_stats or p_stats[1] == 0:
            return global_wr

        wins, total = p_stats
        # Shrink patch sample toward global prior
        return (wins + self.patch_prior_weight * global_wr) / (total + self.patch_prior_weight)

    def get_counter_rate(self, hero_a: int, hero_b: int, patch_id: int | None = None) -> float:
        # 1. Global counter rate
        g_stats = self.counter_stats.get((hero_a, hero_b))
        if not g_stats or g_stats[1] == 0:
            global_rate = 0.5 + (self.get_global_hero_winrate(hero_a) - self.get_global_hero_winrate(hero_b)) * 0.5
        else:
            global_rate = (g_stats[0] + self.smoothing * 0.5) / (g_stats[1] + self.smoothing)

        if patch_id is None:
            return global_rate

        p_stats = self.patch_counter_stats.get((patch_id, hero_a, hero_b))
        if not p_stats or p_stats[1] == 0:
            return global_rate

        # Shrink toward global counter rate
        return (p_stats[0] + self.patch_prior_weight * global_rate) / (p_stats[1] + self.patch_prior_weight)

    def get_synergy_rate(self, hero_1: int, hero_2: int, patch_id: int | None = None) -> float:
        pair = (min(hero_1, hero_2), max(hero_1, hero_2))
        g_stats = self.synergy_stats.get(pair)
        if g_stats and g_stats[1] > 0:
            global_syn = (g_stats[0] + self.smoothing * 0.5) / (g_stats[1] + self.smoothing)
        else:
            global_syn = 0.5

        if patch_id is None:
            return global_syn

        p_stats = self.patch_synergy_stats.get((patch_id, pair[0], pair[1]))
        if not p_stats or p_stats[1] == 0:
            return global_syn

        return (p_stats[0] + self.patch_prior_weight * global_syn) / (p_stats[1] + self.patch_prior_weight)

    def compute_draft_features(
        self,
        radiant_heroes: list[int],
        dire_heroes: list[int],
        patch_id: int | None = None,
    ) -> dict[str, float]:
        """Compute patch-aware draft features before observing game outcome."""
        if len(radiant_heroes) < 5 or len(dire_heroes) < 5:
            return {
                "draft_hero_winrate_diff": 0.0,
                "draft_patch_hero_winrate_diff": 0.0,
                "draft_counter_score": 0.0,
                "draft_patch_counter_score": 0.0,
                "draft_synergy_diff": 0.0,
                "draft_meta_pick_diff": 0.0,
            }

        # 1. Global & Patch-specific hero win rate diffs
        r_gwr = sum(self.get_global_hero_winrate(h) for h in radiant_heroes) / 5.0
        d_gwr = sum(self.get_global_hero_winrate(h) for h in dire_heroes) / 5.0
        gwr_diff = r_gwr - d_gwr

        r_pwr = sum(self.get_patch_hero_winrate(h, patch_id) for h in radiant_heroes) / 5.0
        d_pwr = sum(self.get_patch_hero_winrate(h, patch_id) for h in dire_heroes) / 5.0
        pwr_diff = r_pwr - d_pwr

        # 2. Global & Patch-specific counter matchup score across 25 pairings
        g_counter = 0.0
        p_counter = 0.0
        for r_h in radiant_heroes:
            for d_h in dire_heroes:
                g_counter += self.get_counter_rate(r_h, d_h, patch_id=None) - 0.5
                p_counter += self.get_counter_rate(r_h, d_h, patch_id=patch_id) - 0.5

        # 3. Synergy across 10 teammate pairs
        def calc_synergy(heroes: list[int], pid: int | None) -> float:
            scores = [self.get_synergy_rate(h1, h2, patch_id=pid) - 0.5 for h1, h2 in itertools.combinations(heroes, 2)]
            return sum(scores) / len(scores) if scores else 0.0

        syn_diff = calc_synergy(radiant_heroes, patch_id) - calc_synergy(dire_heroes, patch_id)

        # 4. Meta pick volume on this patch
        def calc_meta_volume(heroes: list[int]) -> float:
            if patch_id is None:
                return 0.0
            return sum(self.patch_hero_stats.get((patch_id, h), [0, 0])[1] for h in heroes) / 5.0

        r_meta = calc_meta_volume(radiant_heroes)
        d_meta = calc_meta_volume(dire_heroes)
        meta_diff = r_meta - d_meta

        return {
            "draft_hero_winrate_diff": round(gwr_diff, 4),
            "draft_patch_hero_winrate_diff": round(pwr_diff, 4),
            "draft_counter_score": round(g_counter / 25.0, 4),
            "draft_patch_counter_score": round(p_counter / 25.0, 4),
            "draft_synergy_diff": round(syn_diff, 4),
            "draft_meta_pick_diff": round(meta_diff, 2),
        }

    def update(
        self,
        radiant_heroes: list[int],
        dire_heroes: list[int],
        radiant_win: bool,
        patch_id: int | None = None,
    ) -> None:
        """Update historical global and patch-conditioned statistics strictly AFTER match concludes."""
        r_outcome = 1 if radiant_win else 0
        d_outcome = 0 if radiant_win else 1

        if patch_id is not None:
            self.patch_total_games[patch_id] = self.patch_total_games.get(patch_id, 0) + 1

        # Global hero updates
        for h in radiant_heroes:
            st = self.hero_stats.setdefault(h, [0, 0])
            st[0] += r_outcome
            st[1] += 1
            if patch_id is not None:
                pst = self.patch_hero_stats.setdefault((patch_id, h), [0, 0])
                pst[0] += r_outcome
                pst[1] += 1

        for h in dire_heroes:
            st = self.hero_stats.setdefault(h, [0, 0])
            st[0] += d_outcome
            st[1] += 1
            if patch_id is not None:
                pst = self.patch_hero_stats.setdefault((patch_id, h), [0, 0])
                pst[0] += d_outcome
                pst[1] += 1

        # Counter pairings (25)
        for r_h in radiant_heroes:
            for d_h in dire_heroes:
                st_rd = self.counter_stats.setdefault((r_h, d_h), [0, 0])
                st_rd[0] += r_outcome
                st_rd[1] += 1

                st_dr = self.counter_stats.setdefault((d_h, r_h), [0, 0])
                st_dr[0] += d_outcome
                st_dr[1] += 1

                if patch_id is not None:
                    p_rd = self.patch_counter_stats.setdefault((patch_id, r_h, d_h), [0, 0])
                    p_rd[0] += r_outcome
                    p_rd[1] += 1

                    p_dr = self.patch_counter_stats.setdefault((patch_id, d_h, r_h), [0, 0])
                    p_dr[0] += d_outcome
                    p_dr[1] += 1

        # Synergy pairings (10 per side)
        for h1, h2 in itertools.combinations(radiant_heroes, 2):
            pair = (min(h1, h2), max(h1, h2))
            st = self.synergy_stats.setdefault(pair, [0, 0])
            st[0] += r_outcome
            st[1] += 1
            if patch_id is not None:
                pst = self.patch_synergy_stats.setdefault((patch_id, pair[0], pair[1]), [0, 0])
                pst[0] += r_outcome
                pst[1] += 1

        for h1, h2 in itertools.combinations(dire_heroes, 2):
            pair = (min(h1, h2), max(h1, h2))
            st = self.synergy_stats.setdefault(pair, [0, 0])
            st[0] += d_outcome
            st[1] += 1
            if patch_id is not None:
                pst = self.patch_synergy_stats.setdefault((patch_id, pair[0], pair[1]), [0, 0])
                pst[0] += d_outcome
                pst[1] += 1
