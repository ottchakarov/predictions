import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wc_bot.decision import (  # noqa: E402
    DecisionConfig,
    evaluate_market,
    kelly_fraction,
    remove_vig,
)


def test_remove_vig_normalises_to_one():
    fair = remove_vig([0.55, 0.50])  # sums to 1.05 (5% vig)
    assert abs(sum(fair) - 1.0) < 1e-9
    assert fair[0] > fair[1]


def test_kelly_zero_when_no_edge():
    assert kelly_fraction(0.50, 0.50) == 0.0
    assert kelly_fraction(0.40, 0.50) == 0.0  # negative edge -> no bet


def test_kelly_positive_with_edge():
    # q=0.6, p=0.5 -> f* = (0.6-0.5)/(1-0.5) = 0.2
    assert abs(kelly_fraction(0.60, 0.50) - 0.20) < 1e-9


def test_edge_flags_bet_only_above_target():
    cfg = DecisionConfig(target_edge=0.04, kelly_fraction=0.25, max_stake_fraction=1.0)
    # Model 0.60 vs fair 0.50 -> 10pt edge, should bet.
    signals = evaluate_market(
        {"Yes": 0.60, "No": 0.40},
        {"Yes": 0.50, "No": 0.50},
        bankroll=1000.0,
        config=cfg,
    )
    yes = next(s for s in signals if s.outcome == "Yes")
    assert yes.is_bet
    assert yes.stake > 0


def test_small_edge_below_target_does_not_bet():
    cfg = DecisionConfig(target_edge=0.05)
    signals = evaluate_market(
        {"Yes": 0.52, "No": 0.48},
        {"Yes": 0.50, "No": 0.50},
        bankroll=1000.0,
        config=cfg,
    )
    yes = next(s for s in signals if s.outcome == "Yes")
    assert not yes.is_bet  # 2pt edge < 5pt target


def test_incomplete_market_prices_nothing():
    # Required 3-way book missing the Draw quote: evaluate_market must NOT de-vig
    # the survivors (which would manufacture a fake edge) and return no signals.
    cfg = DecisionConfig(target_edge=0.01, kelly_fraction=0.25)
    signals = evaluate_market(
        {"Home": 0.60, "Draw": 0.25, "Away": 0.15},
        {"Home": 0.40, "Away": 0.20},  # Draw price absent
        bankroll=1000.0,
        config=cfg,
        required_outcomes=["Home", "Draw", "Away"],
    )
    assert signals == []


def test_complete_market_still_prices():
    # With every required outcome quoted, pricing proceeds normally.
    cfg = DecisionConfig(target_edge=0.01, kelly_fraction=0.25)
    signals = evaluate_market(
        {"Home": 0.60, "Draw": 0.25, "Away": 0.15},
        {"Home": 0.40, "Draw": 0.30, "Away": 0.30},
        bankroll=1000.0,
        config=cfg,
        required_outcomes=["Home", "Draw", "Away"],
    )
    assert len(signals) == 3
    assert any(s.is_bet for s in signals)


def test_max_stake_cap_respected():
    cfg = DecisionConfig(target_edge=0.01, kelly_fraction=1.0, max_stake_fraction=0.05)
    signals = evaluate_market(
        {"Yes": 0.95, "No": 0.05},
        {"Yes": 0.50, "No": 0.50},
        bankroll=1000.0,
        config=cfg,
    )
    yes = next(s for s in signals if s.outcome == "Yes")
    assert yes.kelly_fraction <= 0.05 + 1e-9
    assert yes.stake <= 50.0 + 1e-6
