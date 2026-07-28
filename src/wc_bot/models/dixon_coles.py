"""L4 - Dixon-Coles (bivariate Poisson) goal model.

This upgrades the Elo baseline to a generative scoring model. Following
Dixon & Coles (1997), each team carries an **attack** strength and a **defence**
weakness, and a match's goals are modelled as

    HomeGoals ~ Poisson(lambda),  log lambda = c + home + att[home] + def[away]
    AwayGoals ~ Poisson(mu),      log mu     = c + att[away] + def[home]

with a low-score dependence correction ``tau(.,.;rho)`` that fixes the well-known
underestimation of 0-0 / 1-0 / 0-1 / 1-1 outcomes that an independent double
Poisson produces.

Estimation choices (rigour notes):
* **Weighted MLE.** Parameters are fit by maximum likelihood. Each fixture is
  weighted by a time-decay factor ``0.5 ** (age_days / half_life)`` so recent
  internationals dominate — important because national-team form drifts and the
  raw dataset spans 150 years.
* **Analytic gradient.** The attack/defence/home/intercept block is the standard
  double-Poisson log-likelihood; we supply its exact gradient and solve with
  L-BFGS-B, which is fast and stable even with ~300 teams (~600 params).
* **Profiled rho.** ``rho`` is a small low-score *correction*, not the time
  decay. We fit the Poisson block first, then estimate ``rho`` by 1-D MLE on the
  full Dixon-Coles likelihood holding the rates fixed (block coordinate ascent).
  This keeps the gradient clean and matches how rho behaves in practice.
* **Ridge / identifiability.** The model is invariant to shifting all attacks by
  a constant (absorbed by the intercept), so we add a small L2 penalty and
  re-center attack/defence to mean zero, which also shrinks thin-sample teams.

Interface parity with the Elo model: ``fit`` / ``predict`` /
``match_probabilities`` so it is a drop-in replacement in the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson


@dataclass
class DixonColesConfig:
    half_life_days: float = 540.0     # time-decay: ~18 months to half weight
    max_goals: int = 10               # truncation of the score matrix
    ridge: float = 1e-2               # L2 shrinkage on attack/defence
    max_iter: int = 250
    rho_bounds: Tuple[float, float] = (-0.2, 0.2)
    # L2 RAG fusion: a sentiment of +/-1.0 shifts a team's log-attack by
    # +/- this amount (and its log-defence-weakness by the opposite), i.e. ~5%
    # change to the multiplicative goal rate. Conservative on purpose: an
    # unstructured-text signal must never dominate the fitted strengths.
    sentiment_max_adjustment: float = 0.05


class DixonColesModel:
    # Unified model interface: identifies the algorithm that produced a trade.
    model_version = "dixon_coles-1.0"

    def __init__(self, config: Optional[DixonColesConfig] = None) -> None:
        self.config = config or DixonColesConfig()
        self.teams_: list[str] = []
        self.team_index_: Dict[str, int] = {}
        self.attack_: Optional[np.ndarray] = None
        self.defense_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.home_adv_: float = 0.0
        self.rho_: float = 0.0
        self.reference_date_: Optional[pd.Timestamp] = None
        self._fitted = False

    # ================================================================== fit
    def fit(
        self,
        matches: pd.DataFrame,
        *,
        reference_date: Optional[datetime] = None,
    ) -> "DixonColesModel":
        """Estimate parameters by weighted maximum likelihood.

        ``matches`` needs columns: ``date``, ``home_team``, ``away_team`` and goal
        columns (``home_goals``/``away_goals`` from the FeatureStore, or
        ``home_score``/``away_score`` from raw ingestion). ``neutral`` is optional.
        """
        home_idx, away_idx, hg, ag, neutral, weights = self._prepare(
            matches, reference_date
        )
        n_teams = len(self.teams_)

        # Parameter vector: [intercept, home_adv, attack(n), defence(n)].
        def unpack(p: np.ndarray):
            intercept = p[0]
            home_adv = p[1]
            attack = p[2 : 2 + n_teams]
            defense = p[2 + n_teams :]
            return intercept, home_adv, attack, defense

        not_neutral = (~neutral).astype(float)

        def rates(p: np.ndarray):
            intercept, home_adv, attack, defense = unpack(p)
            log_lambda = (
                intercept + home_adv * not_neutral + attack[home_idx] + defense[away_idx]
            )
            log_mu = intercept + attack[away_idx] + defense[home_idx]
            return np.exp(log_lambda), np.exp(log_mu)

        def neg_loglik(p: np.ndarray) -> float:
            lam, mu = rates(p)
            # Weighted double-Poisson NLL (constant log-factorials dropped).
            nll = np.sum(weights * (lam - hg * np.log(lam) + mu - ag * np.log(mu)))
            _, _, attack, defense = unpack(p)
            nll += 0.5 * self.config.ridge * (attack @ attack + defense @ defense)
            return nll

        def grad(p: np.ndarray) -> np.ndarray:
            lam, mu = rates(p)
            intercept, home_adv, attack, defense = unpack(p)
            r_home = weights * (lam - hg)   # d nll / d log_lambda
            r_away = weights * (mu - ag)    # d nll / d log_mu

            g = np.zeros_like(p)
            g[0] = r_home.sum() + r_away.sum()              # intercept
            g[1] = (r_home * not_neutral).sum()             # home advantage

            g_att = np.zeros(n_teams)
            np.add.at(g_att, home_idx, r_home)   # home team attacks in lambda
            np.add.at(g_att, away_idx, r_away)   # away team attacks in mu
            g_def = np.zeros(n_teams)
            np.add.at(g_def, away_idx, r_home)   # away team defence in lambda
            np.add.at(g_def, home_idx, r_away)   # home team defence in mu

            g[2 : 2 + n_teams] = g_att + self.config.ridge * attack
            g[2 + n_teams :] = g_def + self.config.ridge * defense
            return g

        p0 = np.zeros(2 + 2 * n_teams)
        p0[0] = np.log(max(hg.mean(), 0.1))  # intercept ~ log mean goals
        p0[1] = 0.2                          # mild home-advantage prior

        res = minimize(
            neg_loglik,
            p0,
            jac=grad,
            method="L-BFGS-B",
            options={"maxiter": self.config.max_iter},
        )

        intercept, home_adv, attack, defense = unpack(res.x)
        # Re-center for identifiability; fold the means into the intercept so the
        # implied rates are unchanged.
        intercept += attack.mean() + defense.mean()
        attack = attack - attack.mean()
        defense = defense - defense.mean()

        self.intercept_ = float(intercept)
        self.home_adv_ = float(home_adv)
        self.attack_ = attack
        self.defense_ = defense

        # Profiled rho: 1-D MLE on the full DC likelihood, rates held fixed.
        lam, mu = self._rates_from_params(
            home_idx, away_idx, not_neutral, intercept, home_adv, attack, defense
        )
        self.rho_ = self._fit_rho(hg, ag, lam, mu, weights)
        self._fitted = True
        return self

    # ============================================================== predict
    def predict(
        self,
        home: str,
        away: str,
        *,
        neutral: bool = False,
        home_sentiment: float = 0.0,
        away_sentiment: float = 0.0,
    ) -> Dict[str, float]:
        """1X2 probabilities from the collapsed exact-score matrix.

        ``home_sentiment`` / ``away_sentiment`` are L2 RAG signals in [-1, 1] that
        nudge each team's attack/defence (see :meth:`expected_goals`). Both default
        to 0.0, so the call is identical to the pure baseline when no signal flows.
        """
        matrix = self.score_matrix(
            home, away, neutral=neutral,
            home_sentiment=home_sentiment, away_sentiment=away_sentiment,
        )
        return self.matrix_to_1x2(matrix)

    def match_probabilities(
        self,
        home: str,
        away: str,
        *,
        neutral: bool = False,
        home_sentiment: float = 0.0,
        away_sentiment: float = 0.0,
    ) -> Dict[str, float]:
        """Alias of :meth:`predict` for interface parity with the Elo model."""
        return self.predict(
            home, away, neutral=neutral,
            home_sentiment=home_sentiment, away_sentiment=away_sentiment,
        )

    def knows_team(self, team: str) -> bool:
        """Interface parity: was this team observed during fit?"""
        return team in self.team_index_

    def expected_goals(
        self,
        home: str,
        away: str,
        *,
        neutral: bool = False,
        home_sentiment: float = 0.0,
        away_sentiment: float = 0.0,
    ) -> Tuple[float, float]:
        """Expected goals (lambda_home, mu_away) for a fixture.

        L2 RAG fusion: a team's sentiment in [-1, 1] shifts its fitted log-attack
        by ``sentiment * max_adjustment`` and its log-defence-*weakness* by the
        opposite sign — so positive sentiment (healthy/high-morale squad) both
        raises that team's scoring rate and lowers the rate it concedes, and
        negative sentiment does the reverse. The shift stays in the model's
        log-linear rate link (modeling.mdc), and ``max_adjustment`` keeps the
        unstructured-text signal conservative relative to the fitted strengths.
        """
        self._check_fitted()
        i = self.team_index_.get(home)
        j = self.team_index_.get(away)
        att_h = self.attack_[i] if i is not None else 0.0
        def_h = self.defense_[i] if i is not None else 0.0
        att_a = self.attack_[j] if j is not None else 0.0
        def_a = self.defense_[j] if j is not None else 0.0

        adj = self.config.sentiment_max_adjustment
        att_h += home_sentiment * adj      # better morale -> more attack
        def_h -= home_sentiment * adj      # better morale -> less defensive weakness
        att_a += away_sentiment * adj
        def_a -= away_sentiment * adj

        home_term = 0.0 if neutral else self.home_adv_
        lam = np.exp(self.intercept_ + home_term + att_h + def_a)
        mu = np.exp(self.intercept_ + att_a + def_h)
        return float(lam), float(mu)

    def score_matrix(
        self,
        home: str,
        away: str,
        *,
        neutral: bool = False,
        max_goals: Optional[int] = None,
        home_sentiment: float = 0.0,
        away_sentiment: float = 0.0,
    ) -> np.ndarray:
        """Bivariate probability matrix ``P[x, y]`` for home x, away y goals.

        Independent Poisson outer product with the Dixon-Coles low-score
        correction applied to the 2x2 corner, then renormalised. Optional
        ``home_sentiment`` / ``away_sentiment`` (L2 RAG) feed into the rates via
        :meth:`expected_goals`.
        """
        self._check_fitted()
        k = max_goals or self.config.max_goals
        lam, mu = self.expected_goals(
            home, away, neutral=neutral,
            home_sentiment=home_sentiment, away_sentiment=away_sentiment,
        )

        goals = np.arange(k + 1)
        p_home = poisson.pmf(goals, lam)
        p_away = poisson.pmf(goals, mu)
        matrix = np.outer(p_home, p_away)

        rho = self.rho_
        matrix[0, 0] *= 1.0 - lam * mu * rho
        matrix[0, 1] *= 1.0 + lam * rho
        matrix[1, 0] *= 1.0 + mu * rho
        matrix[1, 1] *= 1.0 - rho

        matrix = np.clip(matrix, 0.0, None)
        total = matrix.sum()
        return matrix / total if total > 0 else matrix

    @staticmethod
    def matrix_to_1x2(matrix: np.ndarray) -> Dict[str, float]:
        """Collapse an exact-score matrix into Home / Draw / Away probabilities."""
        home = float(np.tril(matrix, -1).sum())   # x > y
        draw = float(np.trace(matrix))            # x == y
        away = float(np.triu(matrix, 1).sum())    # x < y
        return {"home": home, "draw": draw, "away": away}

    def most_likely_score(
        self, home: str, away: str, *, neutral: bool = False
    ) -> Tuple[int, int]:
        matrix = self.score_matrix(home, away, neutral=neutral)
        x, y = np.unravel_index(int(matrix.argmax()), matrix.shape)
        return int(x), int(y)

    def team_strengths(self) -> pd.DataFrame:
        """Fitted attack/defence vectors for inspection/audit."""
        self._check_fitted()
        return (
            pd.DataFrame(
                {"team": self.teams_, "attack": self.attack_, "defense": self.defense_}
            )
            .sort_values("attack", ascending=False)
            .reset_index(drop=True)
        )

    # ============================================================ internals
    def _prepare(self, matches: pd.DataFrame, reference_date):
        home_col, away_col = self._goal_columns(matches)
        df = matches.dropna(subset=["home_team", "away_team", home_col, away_col]).copy()

        teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        self.teams_ = teams
        self.team_index_ = {t: i for i, t in enumerate(teams)}

        home_idx = df["home_team"].map(self.team_index_).to_numpy()
        away_idx = df["away_team"].map(self.team_index_).to_numpy()
        hg = df[home_col].astype(float).to_numpy()
        ag = df[away_col].astype(float).to_numpy()
        neutral = (
            df["neutral"].astype(bool).to_numpy()
            if "neutral" in df.columns
            else np.zeros(len(df), dtype=bool)
        )

        dates = pd.to_datetime(df["date"])
        ref = pd.Timestamp(reference_date) if reference_date else dates.max()
        self.reference_date_ = ref
        age_days = (ref - dates).dt.days.clip(lower=0).to_numpy()
        weights = 0.5 ** (age_days / self.config.half_life_days)

        return home_idx, away_idx, hg, ag, neutral, weights

    @staticmethod
    def _goal_columns(matches: pd.DataFrame) -> Tuple[str, str]:
        if {"home_goals", "away_goals"} <= set(matches.columns):
            return "home_goals", "away_goals"
        if {"home_score", "away_score"} <= set(matches.columns):
            return "home_score", "away_score"
        raise ValueError(
            "matches must contain home_goals/away_goals or home_score/away_score"
        )

    @staticmethod
    def _rates_from_params(
        home_idx, away_idx, not_neutral, intercept, home_adv, attack, defense
    ):
        log_lambda = intercept + home_adv * not_neutral + attack[home_idx] + defense[away_idx]
        log_mu = intercept + attack[away_idx] + defense[home_idx]
        return np.exp(log_lambda), np.exp(log_mu)

    def _fit_rho(self, hg, ag, lam, mu, weights) -> float:
        def neg_loglik_rho(rho: float) -> float:
            tau = _tau(hg, ag, lam, mu, rho)
            tau = np.clip(tau, 1e-9, None)  # guard against negative corrections
            return -np.sum(weights * np.log(tau))

        res = minimize_scalar(
            neg_loglik_rho, bounds=self.config.rho_bounds, method="bounded"
        )
        return float(res.x)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("DixonColesModel must be fit() before prediction.")


def _tau(hg: np.ndarray, ag: np.ndarray, lam: np.ndarray, mu: np.ndarray, rho: float):
    """Dixon-Coles low-score dependence correction (vectorised)."""
    tau = np.ones_like(lam, dtype=float)
    m00 = (hg == 0) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m10 = (hg == 1) & (ag == 0)
    m11 = (hg == 1) & (ag == 1)
    tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
    tau[m01] = 1.0 + lam[m01] * rho
    tau[m10] = 1.0 + mu[m10] * rho
    tau[m11] = 1.0 - rho
    return tau
