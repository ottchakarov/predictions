import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wc_bot.features import FeatureStore  # noqa: E402
from wc_bot.features.store import FEATURE_COLUMNS  # noqa: E402


def test_load_exposes_structured_schema():
    df = FeatureStore().load()
    assert list(df.columns) == FEATURE_COLUMNS
    assert pd.api.types.is_integer_dtype(df["home_goals"])
    assert pd.api.types.is_integer_dtype(df["away_goals"])
    assert pd.api.types.is_bool_dtype(df["neutral"])


def test_features_are_chronologically_sorted():
    df = FeatureStore().load()
    assert df["date"].is_monotonic_increasing


def test_training_frame_prevents_lookahead():
    fs = FeatureStore()
    as_of = pd.Timestamp("2018-06-14")  # 2018 World Cup opener
    train = fs.training_frame(as_of)
    # STRICT anti-leakage guarantee: nothing on or after the cutoff is visible.
    assert (train["date"] < as_of).all()
    assert train["date"].max() < as_of


def test_lookback_window_restricts_history():
    fs = FeatureStore()
    as_of = pd.Timestamp("2018-06-14")
    windowed = fs.training_frame(as_of, lookback_days=365)
    assert windowed["date"].min() >= as_of - pd.Timedelta(days=365)
    assert windowed["date"].max() < as_of


def test_teams_as_of_is_subset_of_all():
    fs = FeatureStore()
    early = set(fs.teams(pd.Timestamp("1950-01-01")))
    everyone = set(fs.teams())
    assert early.issubset(everyone)
    assert len(early) < len(everyone)
