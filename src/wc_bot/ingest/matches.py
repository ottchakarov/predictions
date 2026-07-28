"""L1 - Historical match-data ingestion.

Loads international football results and exposes them as a clean, chronologically
sorted ``pandas.DataFrame``. The dataset (martj42/international_results) covers
every men's international from 1872 to the present, which is exactly what we need
to *seed* Elo ratings before evaluating World Cup matches.

POINT-IN-TIME CORRECTNESS (López de Prado):
The single most important invariant in this whole project is that no computation
ever sees information from the future. We enforce the first half of that here by
guaranteeing the frame is sorted by ``date`` ascending, so any downstream
consumer (the Elo fitter, a backtest) can walk it forward and only ever update
state *after* a match has been observed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

DATASET_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)

# This module lives at src/wc_bot/ingest/matches.py, so the repo root is parents[3].
DEFAULT_DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "international_results.csv"

REQUIRED_COLUMNS = {
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "neutral",
}


def load_matches(
    path: Optional[Path | str] = None,
    *,
    download_if_missing: bool = True,
) -> pd.DataFrame:
    """Load and clean the historical results.

    Returns a frame sorted ascending by date with parsed types and a derived
    ``result`` column ('H', 'D', 'A') from the *home* team's perspective.
    """
    path = Path(path) if path else DEFAULT_DATA_PATH

    if not path.exists():
        if download_if_missing:
            path = _download(path)
        else:
            raise FileNotFoundError(
                f"Dataset not found at {path}. Run with download_if_missing=True "
                f"or fetch {DATASET_URL} manually."
            )

    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # Normalise the boolean that arrives as the strings 'TRUE'/'FALSE'.
    df["neutral"] = (
        df["neutral"].astype(str).str.strip().str.upper().map({"TRUE": True, "FALSE": False})
    )
    df["neutral"] = df["neutral"].fillna(False)

    df["result"] = df.apply(_result_from_scores, axis=1)

    # The core point-in-time guarantee for everything downstream.
    df = df.sort_values("date", kind="mergesort").reset_index(drop=True)
    return df


def world_cup_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to FIFA World Cup matches (finals tournament, not qualifiers)."""
    mask = df["tournament"].str.strip().str.lower() == "fifa world cup"
    return df.loc[mask].reset_index(drop=True)


def _result_from_scores(row: pd.Series) -> str:
    if row["home_score"] > row["away_score"]:
        return "H"
    if row["home_score"] < row["away_score"]:
        return "A"
    return "D"


def _download(path: Path) -> Path:
    import requests

    path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(DATASET_URL, timeout=30)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return path
