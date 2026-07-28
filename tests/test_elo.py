import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wc_bot.elo import Elo, EloConfig  # noqa: E402


def test_equal_ratings_neutral_is_fair():
    elo = Elo()
    probs = elo.match_probabilities("A", "B", neutral=True)
    assert abs(probs["home"] - probs["away"]) < 1e-9
    assert abs(sum(probs.values()) - 1.0) < 1e-9
    assert probs["draw"] > 0.2  # draws are common at parity


def test_home_advantage_helps_home():
    elo = Elo()
    home_probs = elo.match_probabilities("A", "B", neutral=False)
    neutral_probs = elo.match_probabilities("A", "B", neutral=True)
    assert home_probs["home"] > neutral_probs["home"]


def test_higher_rating_more_likely_to_win():
    elo = Elo()
    elo.ratings["Strong"] = 2000.0
    elo.ratings["Weak"] = 1200.0
    probs = elo.match_probabilities("Strong", "Weak", neutral=True)
    assert probs["home"] > 0.8
    assert probs["away"] < probs["home"]


def test_probabilities_always_normalised():
    elo = Elo()
    elo.ratings["X"] = 2500.0
    elo.ratings["Y"] = 800.0
    probs = elo.match_probabilities("X", "Y", neutral=True)
    assert all(p >= 0 for p in probs.values())
    assert abs(sum(probs.values()) - 1.0) < 1e-9


def test_winning_raises_rating():
    elo = Elo(config=EloConfig(goal_diff_scaling=False))
    before = elo.rating("A")
    elo.update("A", "B", 2, 0, neutral=True)
    assert elo.rating("A") > before
    assert elo.rating("B") < before  # zero-sum update


def test_goal_diff_scaling_amplifies_blowout():
    blowout = Elo()
    narrow = Elo()
    blowout.update("A", "B", 5, 0, neutral=True)
    narrow.update("A", "B", 1, 0, neutral=True)
    assert blowout.rating("A") > narrow.rating("A")


def test_fit_records_leakage_free_log():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2000-01-01", "2000-02-01"]),
            "home_team": ["A", "A"],
            "away_team": ["B", "B"],
            "home_score": [1, 1],
            "away_score": [0, 0],
            "neutral": [True, True],
            "result": ["H", "H"],
        }
    )
    elo = Elo()
    returned = elo.fit(df, record=True)
    assert returned is elo  # uniform contract: fit() returns self
    log = elo.fit_log_
    # First prediction is made at parity (no prior info); after A wins game 1, the
    # second prediction must favour A more strongly.
    assert log.iloc[1]["p_home"] > log.iloc[0]["p_home"]
