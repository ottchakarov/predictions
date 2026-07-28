"""L3 - Feature Store.

A thin, leakage-safe access layer between raw match data (L1) and the models
(L4). It is deliberately decoupled from any model: the store knows nothing about
Dixon-Coles, Elo, or Poisson math — it only produces clean, structured,
time-stamped feature frames and enforces point-in-time retrieval. Models consume
those frames and own all the statistics.

The single rule this layer guarantees is **no look-ahead bias**: every retrieval
that targets a moment ``as_of`` returns *strictly* prior fixtures (``date <
as_of``), so a backtest loop training "as of" a kickoff can never see that match
or anything after it.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

from ..ingest import load_matches

# The structured schema the store exposes to models. Goal columns are named
# generically (not 'score') because they feed a goals-based generative model.
FEATURE_COLUMNS = ["date", "home_team", "away_team", "home_goals", "away_goals", "neutral"]


class FeatureStore:
    def __init__(self, data_path: Optional[str | Path] = None) -> None:
        self._data_path = data_path
        self._features: Optional[pd.DataFrame] = None  # lazy cache

    # ------------------------------------------------------------- loading
    def load(self, *, force_reload: bool = False) -> pd.DataFrame:
        """Parse the international results CSV into the structured feature frame."""
        if self._features is None or force_reload:
            raw = load_matches(self._data_path)
            self._features = self._build_features(raw)
        return self._features

    @staticmethod
    def _build_features(raw: pd.DataFrame) -> pd.DataFrame:
        """Extract exactly the parameters Dixon-Coles needs, cleanly typed."""
        features = pd.DataFrame(
            {
                "date": pd.to_datetime(raw["date"]),
                "home_team": raw["home_team"].astype(str),
                "away_team": raw["away_team"].astype(str),
                "home_goals": raw["home_score"].astype(int),
                "away_goals": raw["away_score"].astype(int),
                "neutral": raw["neutral"].astype(bool),
            }
        )
        return features.sort_values("date", kind="mergesort").reset_index(drop=True)

    # ----------------------------------------------- point-in-time retrieval
    def training_frame(
        self,
        as_of: Optional[datetime] = None,
        *,
        lookback_days: Optional[int] = None,
        tournament_only_wc: bool = False,
    ) -> pd.DataFrame:
        """Return training rows usable at time ``as_of`` (leakage-free).

        * ``as_of`` (exclusive): keep only fixtures with ``date < as_of``. The
          strict ``<`` is the anti-leakage guarantee — the match being predicted,
          and anything after it, is never visible.
        * ``lookback_days``: optionally restrict to a recent window before
          ``as_of`` (the model also time-decays, so this is mainly a speed knob).
        """
        df = self.load()
        if as_of is not None:
            cutoff = pd.Timestamp(as_of)
            df = df[df["date"] < cutoff]
            if lookback_days is not None:
                df = df[df["date"] >= cutoff - pd.Timedelta(days=lookback_days)]
        return df.reset_index(drop=True)

    def teams(self, as_of: Optional[datetime] = None) -> List[str]:
        """Distinct teams observed strictly before ``as_of``."""
        df = self.training_frame(as_of)
        return sorted(set(df["home_team"]) | set(df["away_team"]))

    def latest_date(self) -> pd.Timestamp:
        return self.load()["date"].max()
