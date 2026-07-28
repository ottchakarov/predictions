"""L4 - The "dumb" model: an Elo rating system.

Elo is computationally cheap and needs no player-level data, yet it produces a
calibrated baseline win probability. That makes it the perfect placeholder for
the skeleton: once the chassis works end-to-end, this single class can be swapped
for a Soccermatics-style xG model or an ML ensemble without touching any other
layer.

Design notes:
* We follow the World Football Elo conventions: a home-advantage offset, a
  tournament-importance weight ``K``, and an optional goal-difference multiplier
  so that a 4-0 win moves ratings more than a 1-0 win.
* Match results are international football, which has frequent draws, so we model
  a full 1X2 (home / draw / away) distribution rather than a two-way coin flip.
* ``fit`` walks matches in date order and records each prediction using only
  ratings that existed *before* that match — no leakage (López de Prado).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class EloConfig:
    base_rating: float = 1500.0
    k_factor: float = 40.0          # base learning rate (World Cup importance is high)
    home_advantage: float = 65.0    # rating points added to the home side
    goal_diff_scaling: bool = True  # weight updates by margin of victory
    # Draw model: p_draw = draw_base * exp(-(rating_diff / draw_scale)^2)
    draw_base: float = 0.29
    draw_scale: float = 350.0


@dataclass
class Elo:
    config: EloConfig = field(default_factory=EloConfig)
    ratings: Dict[str, float] = field(default_factory=dict)
    games_played: Dict[str, int] = field(default_factory=dict)

    # Unified model interface: identifies the algorithm that produced a trade.
    model_version = "elo-1.0"
    # Leakage-free backtest log populated by fit(record=True); None otherwise.
    fit_log_ = None

    # ------------------------------------------------------------------ core
    def rating(self, team: str) -> float:
        return self.ratings.get(team, self.config.base_rating)

    def knows_team(self, team: str) -> bool:
        """Interface parity: has this team been observed during fit?"""
        return team in self.ratings

    def expected_score(self, home: str, away: str, *, neutral: bool = False) -> float:
        """Expected score for the home team in [0, 1] (win=1, draw=0.5)."""
        hfa = 0.0 if neutral else self.config.home_advantage
        diff = self.rating(home) + hfa - self.rating(away)
        return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

    def match_probabilities(
        self, home: str, away: str, *, neutral: bool = False
    ) -> Dict[str, float]:
        """Return calibrated {'home', 'draw', 'away'} probabilities summing to 1.

        We split the Elo expected score ``W`` into the three outcomes using a
        draw model that peaks for evenly matched sides and decays as the rating
        gap widens, with::

            home_win + 0.5 * draw = W   (consistency with expected score)
        """
        w = self.expected_score(home, away, neutral=neutral)
        hfa = 0.0 if neutral else self.config.home_advantage
        diff = self.rating(home) + hfa - self.rating(away)

        p_draw = self.config.draw_base * np.exp(-((diff / self.config.draw_scale) ** 2))
        p_home = w - 0.5 * p_draw
        p_away = 1.0 - w - 0.5 * p_draw

        # Clamp away tiny negatives from the parametric draw model, then renormalise.
        probs = np.array([max(p_home, 1e-6), max(p_draw, 1e-6), max(p_away, 1e-6)])
        probs = probs / probs.sum()
        return {"home": float(probs[0]), "draw": float(probs[1]), "away": float(probs[2])}

    def update(
        self,
        home: str,
        away: str,
        home_score: int,
        away_score: int,
        *,
        neutral: bool = False,
    ) -> None:
        """Apply a single observed result to the ratings (mutates state)."""
        expected = self.expected_score(home, away, neutral=neutral)
        if home_score > away_score:
            actual = 1.0
        elif home_score < away_score:
            actual = 0.0
        else:
            actual = 0.5

        k = self.config.k_factor
        if self.config.goal_diff_scaling:
            k *= _goal_diff_multiplier(abs(home_score - away_score))

        delta = k * (actual - expected)
        self.ratings[home] = self.rating(home) + delta
        self.ratings[away] = self.rating(away) - delta
        self.games_played[home] = self.games_played.get(home, 0) + 1
        self.games_played[away] = self.games_played.get(away, 0) + 1

    # ------------------------------------------------------------------ fit
    def fit(self, matches: pd.DataFrame, *, record: bool = False) -> "Elo":
        """Train ratings by walking matches forward in time. Returns ``self``.

        Returning ``self`` gives every L4 model a uniform, chainable contract
        (matches ``DixonColesModel.fit``). If ``record`` is True, a frame of
        *pre-match* predictions and realised outcomes is stored on ``self.fit_log_``
        — a leakage-free backtest log, because each row's probabilities are
        computed strictly before that match updates the ratings.
        """
        rows = []
        for m in matches.itertuples(index=False):
            home, away = m.home_team, m.away_team
            neutral = bool(m.neutral)

            if record:
                probs = self.match_probabilities(home, away, neutral=neutral)
                rows.append(
                    {
                        "date": m.date,
                        "home_team": home,
                        "away_team": away,
                        "neutral": neutral,
                        "p_home": probs["home"],
                        "p_draw": probs["draw"],
                        "p_away": probs["away"],
                        "result": m.result,
                    }
                )

            self.update(
                home, away, int(m.home_score), int(m.away_score), neutral=neutral
            )

        self.fit_log_ = pd.DataFrame(rows) if record else None
        return self

    def top(self, n: int = 20) -> pd.DataFrame:
        data = sorted(self.ratings.items(), key=lambda kv: kv[1], reverse=True)[:n]
        return pd.DataFrame(data, columns=["team", "rating"])


def _goal_diff_multiplier(goal_diff: int) -> float:
    """World Football Elo margin-of-victory multiplier."""
    if goal_diff <= 1:
        return 1.0
    if goal_diff == 2:
        return 1.5
    return (11.0 + goal_diff) / 8.0
