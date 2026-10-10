from __future__ import annotations

import math


class EloTracker:
    """Chronological sequential Elo tracker.

    Golden rule: Rating updates occur strictly AFTER each game.
    """

    def __init__(self, initial_rating: float = 1500.0, k_factor: float = 32.0, scale: float = 400.0):
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.scale = scale
        self.ratings: dict[int, float] = {}
        self.games_played: dict[int, int] = {}

    def get_rating(self, entity_id: int | None) -> float:
        if entity_id is None:
            return self.initial_rating
        return self.ratings.get(entity_id, self.initial_rating)

    def get_games_count(self, entity_id: int | None) -> int:
        if entity_id is None:
            return 0
        return self.games_played.get(entity_id, 0)

    def expected_prob(self, rating_a: float, rating_b: float) -> float:
        diff = (rating_b - rating_a) / self.scale
        return 1.0 / (1.0 + math.pow(10.0, diff))

    def update(self, entity_a: int | None, entity_b: int | None, outcome_a: float) -> tuple[float, float]:
        """Update ratings after observing outcome (1.0 = A won, 0.0 = B won, 0.5 = draw).

        Returns new (rating_a, rating_b).
        """
        r_a = self.get_rating(entity_a)
        r_b = self.get_rating(entity_b)

        exp_a = self.expected_prob(r_a, r_b)
        exp_b = 1.0 - exp_a

        outcome_b = 1.0 - outcome_a

        new_a = r_a + self.k_factor * (outcome_a - exp_a)
        new_b = r_b + self.k_factor * (outcome_b - exp_b)

        if entity_a is not None:
            self.ratings[entity_a] = new_a
            self.games_played[entity_a] = self.games_played.get(entity_a, 0) + 1

        if entity_b is not None:
            self.ratings[entity_b] = new_b
            self.games_played[entity_b] = self.games_played.get(entity_b, 0) + 1

        return new_a, new_b

    def update_groups(self, ids_a: list[int], ids_b: list[int], outcome_a: float) -> None:
        """Update every member of two groups (e.g. the ten players) against the opposing group's mean rating."""
        if not ids_a or not ids_b:
            return
        mean_a = sum(self.get_rating(i) for i in ids_a) / len(ids_a)
        mean_b = sum(self.get_rating(i) for i in ids_b) / len(ids_b)
        exp_a = self.expected_prob(mean_a, mean_b)
        for i in ids_a:
            self.ratings[i] = self.get_rating(i) + self.k_factor * (outcome_a - exp_a)
            self.games_played[i] = self.games_played.get(i, 0) + 1
        for i in ids_b:
            self.ratings[i] = self.get_rating(i) + self.k_factor * ((1.0 - outcome_a) - (1.0 - exp_a))
            self.games_played[i] = self.games_played.get(i, 0) + 1
