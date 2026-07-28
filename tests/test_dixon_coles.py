import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wc_bot.models import DixonColesConfig, DixonColesModel  # noqa: E402


def _synthetic_matches() -> pd.DataFrame:
    """A small league where 'Strong' >> 'Mid' >> 'Weak' in scoring."""
    rng = np.random.default_rng(0)
    base = {"Strong": (2.4, 0.4), "Mid": (1.2, 1.2), "Weak": (0.4, 2.4)}
    teams = list(base)
    rows = []
    date = pd.Timestamp("2024-01-01")
    for _ in range(80):
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                hg = rng.poisson(base[h][0])
                ag = rng.poisson(base[a][0])
                rows.append(
                    {
                        "date": date,
                        "home_team": h,
                        "away_team": a,
                        "home_goals": int(hg),
                        "away_goals": int(ag),
                        "neutral": True,
                    }
                )
                date += pd.Timedelta(days=1)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def model():
    return DixonColesModel(DixonColesConfig(half_life_days=100000)).fit(
        _synthetic_matches()
    )


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        DixonColesModel().predict("A", "B")


def test_probabilities_sum_to_one(model):
    p = model.predict("Strong", "Weak", neutral=True)
    assert abs(sum(p.values()) - 1.0) < 1e-9
    assert all(v >= 0 for v in p.values())


def test_stronger_team_is_favoured(model):
    p = model.predict("Strong", "Weak", neutral=True)
    assert p["home"] > 0.6
    assert p["home"] > p["away"]


def test_expected_goals_reflect_strength(model):
    lam, mu = model.expected_goals("Strong", "Weak", neutral=True)
    assert lam > mu  # strong home side should out-score the weak away side


def test_score_matrix_is_a_distribution(model):
    m = model.score_matrix("Strong", "Mid", neutral=True)
    assert m.shape == (model.config.max_goals + 1, model.config.max_goals + 1)
    assert abs(m.sum() - 1.0) < 1e-9
    assert (m >= 0).all()


def test_matrix_collapse_matches_predict(model):
    m = model.score_matrix("Mid", "Strong", neutral=True)
    collapsed = model.matrix_to_1x2(m)
    p = model.predict("Mid", "Strong", neutral=True)
    for k in ("home", "draw", "away"):
        assert abs(collapsed[k] - p[k]) < 1e-9


def test_match_probabilities_is_alias(model):
    assert model.match_probabilities("Strong", "Mid", neutral=True) == model.predict(
        "Strong", "Mid", neutral=True
    )


def test_unknown_team_is_handled(model):
    p = model.predict("Strong", "NeverSeenFC", neutral=True)
    assert abs(sum(p.values()) - 1.0) < 1e-9


def test_home_advantage_helps_home(model):
    home = model.predict("Mid", "Mid", neutral=False)
    neutral = model.predict("Mid", "Mid", neutral=True)
    assert home["home"] > neutral["home"]


# --------------------------------------------------------- L2 RAG sentiment fusion
def test_zero_sentiment_equals_baseline(model):
    base = model.predict("Mid", "Mid", neutral=True)
    fused = model.predict("Mid", "Mid", neutral=True, home_sentiment=0.0, away_sentiment=0.0)
    assert base == fused  # default 0.0 must be a pure no-op


def test_positive_home_sentiment_raises_home_attack_and_defense(model):
    lam0, mu0 = model.expected_goals("Mid", "Mid", neutral=True)
    lam1, mu1 = model.expected_goals("Mid", "Mid", neutral=True, home_sentiment=1.0)
    assert lam1 > lam0   # healthy home squad scores more
    assert mu1 < mu0     # ...and concedes fewer (better defence)


def test_positive_home_sentiment_increases_home_win_prob(model):
    base = model.predict("Mid", "Mid", neutral=True)
    boosted = model.predict("Mid", "Mid", neutral=True, home_sentiment=1.0)
    assert boosted["home"] > base["home"]


def test_negative_home_sentiment_decreases_home_win_prob(model):
    base = model.predict("Mid", "Mid", neutral=True)
    crisis = model.predict("Mid", "Mid", neutral=True, home_sentiment=-1.0)
    assert crisis["home"] < base["home"]


def test_sentiment_fusion_keeps_valid_distribution(model):
    p = model.predict("Strong", "Weak", neutral=True, home_sentiment=-1.0, away_sentiment=1.0)
    assert abs(sum(p.values()) - 1.0) < 1e-9
    assert all(v >= 0 for v in p.values())


def test_adjustment_magnitude_matches_config():
    import numpy as np
    m = DixonColesModel(DixonColesConfig(half_life_days=100000, sentiment_max_adjustment=0.05))
    m.fit(_synthetic_matches())
    lam0, _ = m.expected_goals("Mid", "Mid", neutral=True)
    lam1, _ = m.expected_goals("Mid", "Mid", neutral=True, home_sentiment=1.0)
    # +1.0 sentiment shifts log-attack by +0.05 -> lambda scales by exp(0.05).
    assert abs(lam1 / lam0 - np.exp(0.05)) < 1e-9
