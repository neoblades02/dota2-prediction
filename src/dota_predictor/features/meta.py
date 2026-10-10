from __future__ import annotations


class PatchMetaTracker:
    """Tracks map side bias (Radiant vs Dire win rates) and drafting turn order advantage per patch."""

    def __init__(self, smoothing: float = 20.0):
        self.smoothing = smoothing
        # patch_id -> [radiant_wins, total_games]
        self.patch_side_stats: dict[int, list[int]] = {}

    def get_radiant_side_advantage(self, patch_id: int | None) -> float:
        """Returns the deviation of Radiant side win rate from 50% on this patch."""
        if patch_id is None:
            return 0.0
        stats = self.patch_side_stats.get(patch_id)
        if not stats or stats[1] == 0:
            return 0.0
        r_wins, total = stats
        # Smoothed radiant win rate
        r_wr = (r_wins + self.smoothing * 0.5) / (total + self.smoothing)
        return round(r_wr - 0.5, 4)

    def update(self, patch_id: int | None, radiant_win: bool) -> None:
        if patch_id is not None:
            st = self.patch_side_stats.setdefault(patch_id, [0, 0])
            st[0] += 1 if radiant_win else 0
            st[1] += 1
