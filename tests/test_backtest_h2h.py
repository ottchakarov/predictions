import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from wc_bot.features import FeatureStore  # noqa: E402

import backtest_h2h as bt  # noqa: E402


# ------------------------------------------------------------------- metrics
def test_truth_from_goals_maps_outcomes():
    hg = np.array([2, 1, 0])
    ag = np.array([1, 1, 2])
    assert list(bt.truth_from_goals(hg, ag)) == [0, 1, 2]  # home, draw, away


def test_log_loss_is_zero_for_perfect_confident_predictions():
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    truth = np.array([0, 1])
    assert bt.log_loss(probs, truth) < 1e-9


def test_log_loss_penalises_wrong_confident_prediction():
    probs = np.array([[1.0, 0.0, 0.0]])
    truth = np.array([2])  # confidently wrong
    assert bt.log_loss(probs, truth) > 10  # clipped -log(eps)


def test_brier_zero_for_perfect_and_two_for_worst():
    perfect = bt.brier_score(np.array([[1.0, 0.0, 0.0]]), np.array([0]))
    worst = bt.brier_score(np.array([[1.0, 0.0, 0.0]]), np.array([2]))
    assert perfect < 1e-9
    assert abs(worst - 2.0) < 1e-9


def test_calibration_error_zero_when_confidence_matches_accuracy():
    # 10 predictions at 100% confidence, all correct -> perfectly calibrated.
    probs = np.tile([0.99, 0.005, 0.005], (10, 1))
    truth = np.zeros(10, dtype=int)
    assert bt.calibration_error(probs, truth) < 0.02


def test_base_rates_sum_to_one():
    truth = np.array([0, 0, 1, 2])
    rates = bt.base_rates(truth)
    assert abs(rates.sum() - 1.0) < 1e-9
    assert rates[0] == 0.5


# ----------------------------------------------------------------- flat PnL
def test_flat_bet_pnl_profits_when_model_beats_priced_base_rate():
    # base rate for home = 0.25 (odds 4.0). Model says 0.90 (>0.25+0.05) -> bet,
    # and home actually wins -> profit 100*(4-1)=300 per bet.
    rates = np.array([0.25, 0.25, 0.50])
    probs = np.array([[0.90, 0.05, 0.05]])
    truth = np.array([0])
    res = bt.flat_bet_pnl(probs, truth, rates, edge_threshold=0.05, stake=100.0)
    assert res.bets == 1
    assert abs(res.profit - 300.0) < 1e-9
    assert res.roi == 3.0


def test_flat_bet_pnl_no_bet_below_edge_threshold():
    rates = np.array([0.33, 0.33, 0.34])
    probs = np.array([[0.34, 0.33, 0.33]])  # only 0.01 over base rate -> no bet
    truth = np.array([0])
    res = bt.flat_bet_pnl(probs, truth, rates, edge_threshold=0.05)
    assert res.bets == 0 and res.staked == 0.0 and res.roi == 0.0


def test_flat_bet_pnl_loss_recorded():
    rates = np.array([0.25, 0.25, 0.50])
    probs = np.array([[0.90, 0.05, 0.05]])
    truth = np.array([2])  # bet home, away wins -> lose stake
    res = bt.flat_bet_pnl(probs, truth, rates, stake=100.0)
    assert res.profit == -100.0 and res.hit_rate == 0.0


# ------------------------------------------------- end-to-end on injected store
def _synthetic_features(n_per_year=60, years=(2018, 2019, 2020, 2021)) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    teams = [f"Team{i}" for i in range(8)]
    strength = {t: rng.normal(0, 0.4) for t in teams}
    rows = []
    for y in years:
        for g in range(n_per_year):
            h, a = rng.choice(teams, size=2, replace=False)
            lam = np.exp(0.2 + strength[h] - strength[a])
            mu = np.exp(0.1 + strength[a] - strength[h])
            rows.append({
                "date": pd.Timestamp(f"{y}-{1 + g % 12:02d}-15"),
                "home_team": h,
                "away_team": a,
                "home_goals": int(rng.poisson(lam)),
                "away_goals": int(rng.poisson(mu)),
                "neutral": False,
            })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _store_with(features: pd.DataFrame) -> FeatureStore:
    store = FeatureStore()
    store._features = features  # inject to bypass CSV load
    return store


def test_run_backtest_is_point_in_time_and_produces_aligned_outputs():
    store = _store_with(_synthetic_features())
    config = bt.BacktestConfig(
        start_year=2019, step_months=12, dc_lookback_days=None,
        min_train_matches=30,
    )
    elo_run, dc_run, dc_rag_run, truth, meta = bt.run_backtest(store, config)
    elo, dc = elo_run.array(), dc_run.array()

    assert dc_rag_run is None  # no RAG column unless --use-rag
    assert len(truth) == len(elo) == len(dc) > 0
    # Probabilities are valid distributions.
    assert np.allclose(elo.sum(axis=1), 1.0, atol=1e-6)
    assert np.allclose(dc.sum(axis=1), 1.0, atol=1e-6)
    # Both models beat (or match) a uniform guess on log-loss on this signal.
    uniform = np.full_like(elo, 1 / 3)
    assert bt.log_loss(elo, truth) <= bt.log_loss(uniform, truth) + 1e-9
    assert bt.log_loss(dc, truth) <= bt.log_loss(uniform, truth) + 1e-9


def test_report_renders_without_error():
    store = _store_with(_synthetic_features())
    config = bt.BacktestConfig(start_year=2019, dc_lookback_days=None, min_train_matches=30)
    elo_run, dc_run, _dc_rag, truth, meta = bt.run_backtest(store, config)
    report = bt.format_report(elo_run.array(), dc_run.array(), truth, config, meta)
    assert "HEAD-TO-HEAD" in report and "log-loss" in report and "ROI" in report


def test_pricing_uses_historical_rates_not_eval_outcomes():
    # Train year (2019): every match is a HOME win. Eval year (2020): every match
    # is an AWAY win. If the flat-stake pricing leaked the evaluation outcome
    # distribution, the priced base rate would be away-heavy. Leakage-free pricing
    # must instead reflect the home-heavy TRAINING history.
    teams = [f"T{i}" for i in range(6)]
    rng = np.random.default_rng(1)
    rows = []
    for g in range(40):  # 2019: home wins (both sides score -> DC stays numeric)
        h, a = rng.choice(teams, size=2, replace=False)
        rows.append({"date": pd.Timestamp(f"2019-{1 + g % 12:02d}-10"),
                     "home_team": h, "away_team": a,
                     "home_goals": 2, "away_goals": 1, "neutral": False})
    for g in range(40):  # 2020: away wins
        h, a = rng.choice(teams, size=2, replace=False)
        rows.append({"date": pd.Timestamp(f"2020-{1 + g % 12:02d}-10"),
                     "home_team": h, "away_team": a,
                     "home_goals": 1, "away_goals": 2, "neutral": False})
    features = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    store = _store_with(features)

    config = bt.BacktestConfig(
        start_year=2020, step_months=12, dc_lookback_days=None, min_train_matches=30
    )
    _elo, _dc, _rag, truth, meta = bt.run_backtest(store, config)

    assert len(truth) > 0
    priced = meta["priced_rates"]
    assert priced.shape == (len(truth), 3)
    # Pricing reflects the training history (home base rate = 1.0), NOT the eval
    # truth (which is entirely away wins).
    assert np.all(priced[:, 0] == 1.0)
    assert np.all(priced[:, 2] == 0.0)
    eval_rates = bt.base_rates(truth)  # away-heavy: the leaked distribution
    assert eval_rates[2] > eval_rates[0]
    assert not np.allclose(priced.mean(axis=0), eval_rates)


def test_use_rag_produces_aligned_rag_column():
    store = _store_with(_synthetic_features())
    config = bt.BacktestConfig(
        start_year=2019, dc_lookback_days=None, min_train_matches=30, use_rag=True,
    )
    elo_run, dc_run, dc_rag_run, truth, meta = bt.run_backtest(store, config)
    assert meta["use_rag"] is True
    assert dc_rag_run is not None
    rag = dc_rag_run.array()
    assert len(rag) == len(truth) > 0
    assert np.allclose(rag.sum(axis=1), 1.0, atol=1e-6)
    # The mock signal is non-trivial, so DC+RAG must diverge from pure DC somewhere.
    assert not np.allclose(rag, dc_run.array())

    report = bt.format_report(
        elo_run.array(), dc_run.array(), truth, config, meta, dc_rag=rag
    )
    assert "DC+RAG" in report and "RAG verdict" in report
