#!/usr/bin/env python3
"""Head-to-Head L4 backtester: Elo vs Dixon-Coles, point-in-time and leakage-free.

This is the formal quantitative baseline we must clear before adding the L2 RAG
(unstructured text) layer. It replaces the old ``backtest_elo.py`` single-model
sanity check.

Protocol (strictly point-in-time; data_hygiene.mdc)
---------------------------------------------------
History is stepped through in fixed periods (default: one year). For each period
``[p_start, p_end)`` BOTH models are (re)fit ONLY on fixtures with
``date < p_start`` via the L3 ``FeatureStore`` — the strict ``<`` is the
anti-leakage guarantee — and then used to predict every match that kicks off
inside the period. No match is ever scored by a model that has seen it or
anything after it.

* Elo trains on the full prior history (cheap; needs long memory for stable
  ratings).
* Dixon-Coles trains on a bounded recent window (``--dc-lookback-days``); it
  time-decays internally so older fixtures are ~weightless anyway, and bounding
  the window keeps the sequential MLE fast.

Outputs
-------
1. Comparative predictive metrics (log-loss, Brier, calibration error) side by
   side, vs an empirical base-rate baseline.
2. A flat-stake theoretical PnL/ROI (sizing held constant to isolate predictive
   accuracy from the live Kelly/Sharp execution layer).

Run::

    python scripts/backtest_h2h.py                       # full default protocol
    python scripts/backtest_h2h.py --start-year 2010 --wc-only
    python scripts/backtest_h2h.py --step-months 6 --dc-lookback-days 1500
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wc_bot.elo import Elo, EloConfig  # noqa: E402
from wc_bot.features import FeatureStore  # noqa: E402
from wc_bot.ingest import load_matches, world_cup_matches  # noqa: E402
from wc_bot.models import DixonColesConfig, DixonColesModel  # noqa: E402
from wc_bot.rag import NewsFetcher, SentimentAgent  # noqa: E402

logger = logging.getLogger("backtest_h2h")

_EPS = 1e-12
_OUTCOME_LABELS = ("home", "draw", "away")


# ============================================================ scoring metrics
def truth_from_goals(home_goals: np.ndarray, away_goals: np.ndarray) -> np.ndarray:
    """Map (home_goals, away_goals) to outcome index: 0=home, 1=draw, 2=away."""
    out = np.full(len(home_goals), 1, dtype=int)  # draw
    out[home_goals > away_goals] = 0
    out[home_goals < away_goals] = 2
    return out


def log_loss(probs: np.ndarray, truth: np.ndarray) -> float:
    """Multiclass cross-entropy: mean -log(p assigned to the realised outcome)."""
    picked = probs[np.arange(len(truth)), truth]
    return float(-np.log(np.clip(picked, _EPS, 1.0)).mean())


def brier_score(probs: np.ndarray, truth: np.ndarray) -> float:
    """Multiclass Brier: mean squared error vs the one-hot realised outcome."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(truth)), truth] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def calibration_error(probs: np.ndarray, truth: np.ndarray, *, n_bins: int = 10) -> float:
    """Expected Calibration Error on the top-class confidence.

    Bins predictions by their max probability (confidence). Within each bin we
    compare mean confidence to realised accuracy (argmax == truth); ECE is the
    sample-weighted mean absolute gap. Low ECE => the stated probability can be
    trusted for sizing (Kelly is only as good as the calibration of ``p``).
    """
    confidence = probs.max(axis=1)
    predicted = probs.argmax(axis=1)
    correct = (predicted == truth).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(truth)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        # Bins are (lo, hi]; the top bin's hi == 1.0 includes confidence == 1.0.
        in_bin = (confidence > lo) & (confidence <= hi)
        count = int(in_bin.sum())
        if count == 0:
            continue
        ece += (count / n) * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


def base_rates(truth: np.ndarray) -> np.ndarray:
    """Empirical [home, draw, away] frequencies over the evaluated sample."""
    counts = np.bincount(truth, minlength=3).astype(float)
    return counts / counts.sum() if counts.sum() > 0 else counts


# ===================================================== flat-betting PnL model
@dataclass
class PnLResult:
    bets: int
    staked: float
    profit: float
    roi: float          # profit / staked
    hit_rate: float     # fraction of placed bets that won


def flat_bet_pnl(
    probs: np.ndarray,
    truth: np.ndarray,
    rates: np.ndarray,
    *,
    edge_threshold: float = 0.05,
    stake: float = 100.0,
) -> PnLResult:
    """Flat-stake theoretical PnL priced at the HISTORICAL base rate.

    ``rates`` is the "market" implied probability used to price each outcome
    (fair decimal odds = 1 / rate). It may be either a single ``[home, draw,
    away]`` vector (applied to every match) or a per-match ``(n, 3)`` array — the
    latter is REQUIRED in the backtester so each bet is priced at the base rate
    derived strictly from data prior to the match (no look-ahead; data_hygiene.mdc).

    We place a flat ``stake`` whenever the model's probability beats the priced
    rate by more than ``edge_threshold``. Win pays ``stake * (1/rate - 1)``; loss
    pays ``-stake``. Flat sizing isolates predictive accuracy from the live
    Kelly/Sharp sizing layer.
    """
    rates = np.asarray(rates, dtype=float)
    if rates.ndim == 1:
        rates = np.tile(rates, (len(truth), 1))
    if rates.shape != (len(truth), 3):
        raise ValueError(
            f"rates must be shape (3,) or ({len(truth)}, 3); got {rates.shape}"
        )

    bets = 0
    staked = 0.0
    profit = 0.0
    wins = 0
    for i in range(len(truth)):
        for o in range(3):
            rate = rates[i, o]
            if rate <= 0:
                continue
            if probs[i, o] > rate + edge_threshold:
                bets += 1
                staked += stake
                if truth[i] == o:
                    profit += stake * (1.0 / rate - 1.0)
                    wins += 1
                else:
                    profit -= stake
    roi = (profit / staked) if staked > 0 else 0.0
    hit_rate = (wins / bets) if bets > 0 else 0.0
    return PnLResult(bets=bets, staked=staked, profit=profit, roi=roi, hit_rate=hit_rate)


# ================================================================ backtester
@dataclass
class BacktestConfig:
    start_year: int = 2006
    step_months: int = 12
    dc_lookback_days: Optional[int] = 2000   # bounded window for DC speed/decay
    min_train_matches: int = 500             # warmup before a period is scored
    wc_only: bool = False                    # score only World Cup fixtures
    edge_threshold: float = 0.05
    flat_stake: float = 100.0
    data_path: Optional[str] = None
    use_rag: bool = False                     # fuse L2 sentiment into a DC+RAG column
    rag_lookback_days: int = 7               # point-in-time news window per match


@dataclass
class ModelRun:
    name: str
    probs: List[List[float]]

    def array(self) -> np.ndarray:
        return np.asarray(self.probs, dtype=float)


def _period_starts(start: pd.Timestamp, end: pd.Timestamp, step_months: int) -> List[pd.Timestamp]:
    return list(pd.date_range(start=start, end=end, freq=pd.DateOffset(months=step_months)))


def _fit_elo(train: pd.DataFrame) -> Elo:
    # Elo consumes home_score/away_score; FeatureStore exposes home_goals/away_goals.
    elo_frame = train.rename(columns={"home_goals": "home_score", "away_goals": "away_score"})
    return Elo(config=EloConfig()).fit(elo_frame, record=False)


def _fit_dixon_coles(train: pd.DataFrame, reference_date: pd.Timestamp) -> DixonColesModel:
    return DixonColesModel(DixonColesConfig()).fit(train, reference_date=reference_date)


def run_backtest(store: FeatureStore, config: BacktestConfig):
    """Walk history period-by-period, returning per-model probs + realised truth.

    Returns (elo_run, dc_run, truth, meta) where meta carries diagnostics.
    """
    features = store.load()
    if config.wc_only:
        # The feature frame drops the 'tournament' column, so derive the WC key set
        # from the raw loader and match on (date, home_team, away_team).
        raw = load_matches(config.data_path)
        wc = world_cup_matches(raw)
        wc_keys = set(
            zip(pd.to_datetime(wc["date"]), wc["home_team"].astype(str), wc["away_team"].astype(str))
        )
    else:
        wc_keys = None

    data_start = features["date"].min()
    data_end = features["date"].max()
    first = pd.Timestamp(year=config.start_year, month=1, day=1)
    first = max(first, pd.Timestamp(data_start))
    starts = _period_starts(first, pd.Timestamp(data_end), config.step_months)

    elo_probs: List[List[float]] = []
    dc_probs: List[List[float]] = []
    dc_rag_probs: List[List[float]] = []
    truth: List[int] = []
    # Per-match "market" price for the flat-stake PnL: the empirical [home, draw,
    # away] base rate computed STRICTLY from the period's training frame
    # (date < p_start). Pricing must never see the evaluation outcomes it scores.
    priced_rates: List[List[float]] = []

    # L2 RAG collaborators (stubbed/deterministic). Sentiment is cached per
    # (team, kickoff-date) so the home/away lookups are computed once per fixture.
    fetcher = SentimentAgent_news_fetcher = None
    agent = None
    sent_cache: Dict[tuple, float] = {}
    if config.use_rag:
        SentimentAgent_news_fetcher = NewsFetcher(mock=True)
        agent = SentimentAgent(mock=True)

    def _sentiment(team: str, as_of) -> float:
        key = (team, as_of)
        if key not in sent_cache:
            sent_cache[key] = agent.team_sentiment(
                team, as_of, SentimentAgent_news_fetcher,
                lookback_days=config.rag_lookback_days,
            )
        return sent_cache[key]

    n_periods = len(starts)
    scored = 0
    skipped_unknown = 0
    t0 = time.time()
    logger.info(
        "Backtest %d period(s) from %s to %s (step=%dmo, dc_lookback=%s, wc_only=%s)",
        n_periods, first.date(), pd.Timestamp(data_end).date(),
        config.step_months, config.dc_lookback_days, config.wc_only,
    )

    for k, p_start in enumerate(starts):
        p_end = starts[k + 1] if k + 1 < n_periods else (pd.Timestamp(data_end) + pd.Timedelta(days=1))

        elo_train = store.training_frame(as_of=p_start)
        if len(elo_train) < config.min_train_matches:
            logger.info("[%d/%d] %s: warmup (%d<%d train) — skipped",
                        k + 1, n_periods, p_start.date(), len(elo_train), config.min_train_matches)
            continue

        dc_train = store.training_frame(as_of=p_start, lookback_days=config.dc_lookback_days)

        # Historical base rate priced into this period's bets — derived from the
        # training frame only (date < p_start), never the matches being scored.
        train_truth = truth_from_goals(
            elo_train["home_goals"].to_numpy(), elo_train["away_goals"].to_numpy()
        )
        period_rates = base_rates(train_truth).tolist()

        elo = _fit_elo(elo_train)
        dc = _fit_dixon_coles(dc_train, reference_date=p_start)

        period_eval = features[(features["date"] >= p_start) & (features["date"] < p_end)]
        period_scored = 0
        for m in period_eval.itertuples(index=False):
            h, a, neutral = m.home_team, m.away_team, bool(m.neutral)
            if wc_keys is not None and (m.date, h, a) not in wc_keys:
                continue
            if not (elo.knows_team(h) and elo.knows_team(a)
                    and dc.knows_team(h) and dc.knows_team(a)):
                skipped_unknown += 1
                continue

            ep = elo.match_probabilities(h, a, neutral=neutral)
            dp = dc.match_probabilities(h, a, neutral=neutral)
            elo_probs.append([ep["home"], ep["draw"], ep["away"]])
            dc_probs.append([dp["home"], dp["draw"], dp["away"]])

            if config.use_rag:
                # Strict point-in-time: news as-of the kickoff only (date < as_of).
                hs = _sentiment(h, m.date)
                as_ = _sentiment(a, m.date)
                rp = dc.match_probabilities(
                    h, a, neutral=neutral, home_sentiment=hs, away_sentiment=as_
                )
                dc_rag_probs.append([rp["home"], rp["draw"], rp["away"]])

            truth.append(_outcome_idx(m.home_goals, m.away_goals))
            priced_rates.append(period_rates)
            period_scored += 1

        scored += period_scored
        logger.info(
            "[%d/%d] %s: elo_train=%d dc_train=%d scored=%d (cum=%d) elapsed=%.1fs",
            k + 1, n_periods, p_start.date(), len(elo_train), len(dc_train),
            period_scored, scored, time.time() - t0,
        )

    meta = {
        "scored": scored,
        "skipped_unknown": skipped_unknown,
        "periods": n_periods,
        "elapsed_s": time.time() - t0,
        "use_rag": config.use_rag,
        # Leakage-free per-match pricing (historical base rates) for the PnL model.
        "priced_rates": np.asarray(priced_rates, dtype=float),
    }
    logger.info("Done: scored %d matches (%d skipped: unknown team) in %.1fs",
                scored, skipped_unknown, meta["elapsed_s"])
    dc_rag_run = ModelRun("Dixon-Coles+RAG", dc_rag_probs) if config.use_rag else None
    return (
        ModelRun("Elo", elo_probs),
        ModelRun("Dixon-Coles", dc_probs),
        dc_rag_run,
        np.asarray(truth, dtype=int),
        meta,
    )


def _outcome_idx(home_goals: int, away_goals: int) -> int:
    if home_goals > away_goals:
        return 0
    if home_goals < away_goals:
        return 2
    return 1


# ================================================================== reporting
def score_model(probs: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    preds = probs.argmax(axis=1)
    return {
        "accuracy": float((preds == truth).mean()),
        "log_loss": log_loss(probs, truth),
        "brier": brier_score(probs, truth),
        "ece": calibration_error(probs, truth),
    }


def format_report(
    elo: np.ndarray,
    dc: np.ndarray,
    truth: np.ndarray,
    config: BacktestConfig,
    meta: dict,
    dc_rag: Optional[np.ndarray] = None,
) -> str:
    # Pricing is the HISTORICAL base rate (computed in run_backtest from each
    # period's training frame, date < p_start) — NOT base_rates(truth), which
    # would leak the evaluation outcome distribution into the bet pricing.
    priced = meta.get("priced_rates")
    if priced is None or len(priced) != len(truth) or len(truth) == 0:
        # Fallback only for ad-hoc calls without a backtest meta; flagged as such.
        priced = np.tile(base_rates(truth), (len(truth), 1)) if len(truth) else np.empty((0, 3))
    base_probs = np.asarray(priced, dtype=float)
    # Headline "base rate" line: the average historical price actually used.
    rates = base_probs.mean(axis=0) if len(base_probs) else np.zeros(3)

    # Columns: always Elo + Dixon-Coles + base-rate; add DC+RAG when present.
    columns: List[Tuple[str, np.ndarray]] = [
        ("Elo", elo),
        ("Dixon-Coles", dc),
    ]
    if dc_rag is not None and len(dc_rag) == len(truth) and len(truth) > 0:
        columns.append(("DC+RAG", dc_rag))
    columns.append(("base-rate", base_probs))

    metrics = {name: score_model(arr, truth) for name, arr in columns}
    pnls = {
        name: flat_bet_pnl(arr, truth, base_probs,
                           edge_threshold=config.edge_threshold, stake=config.flat_stake)
        for name, arr in columns if name != "base-rate"
    }

    def row(label: str, fmt, getter) -> str:
        cells = "".join(f"{fmt(getter(metrics[name])):>16}" for name, _ in columns)
        return f"  {label:<16}{cells}"

    lines: List[str] = []
    w = lines.append
    w("=" * (24 + 16 * len(columns)))
    title = "  HEAD-TO-HEAD L4 BACKTEST  —  Elo vs Dixon-Coles"
    if dc_rag is not None:
        title += " vs DC+RAG"
    w(title)
    w("=" * (24 + 16 * len(columns)))
    w(f"  Matches scored : {len(truth):,}   (periods={meta['periods']}, "
      f"skipped-unknown={meta['skipped_unknown']})")
    w(f"  Base rates     : home {rates[0]:.3f} | draw {rates[1]:.3f} | away {rates[2]:.3f}")
    w(f"  Runtime        : {meta['elapsed_s']:.1f}s")
    w("")
    w("  PREDICTIVE ACCURACY (lower log-loss / Brier / ECE = better)")
    w("  " + "-" * (16 * len(columns) + 6))
    w("  " + f"{'metric':<16}" + "".join(f"{name:>16}" for name, _ in columns))
    w(row("accuracy", lambda v: f"{v:.4f}", lambda m: m["accuracy"]))
    w(row("log-loss", lambda v: f"{v:.4f}", lambda m: m["log_loss"]))
    w(row("Brier", lambda v: f"{v:.4f}", lambda m: m["brier"]))
    w(row("calibration ECE", lambda v: f"{v:.4f}", lambda m: m["ece"]))
    w("")
    w(f"  FLAT-STAKE PnL  (${config.flat_stake:.0f}/bet, edge > base-rate + "
      f"{config.edge_threshold:.0%}, priced at HISTORICAL base-rate odds)")
    w("  " + "-" * (16 * len(columns) + 6))
    bet_cols = [name for name, _ in columns if name != "base-rate"]
    w("  " + f"{'metric':<16}" + "".join(f"{name:>16}" for name in bet_cols))
    w("  " + f"{'bets placed':<16}" + "".join(f"{pnls[n].bets:>16,}" for n in bet_cols))
    w("  " + f"{'total staked':<16}" + "".join(f"{('$'+format(pnls[n].staked, ',.0f')):>16}" for n in bet_cols))
    w("  " + f"{'net profit':<16}" + "".join(f"{('$'+format(pnls[n].profit, ',.0f')):>16}" for n in bet_cols))
    w("  " + f"{'ROI':<16}" + "".join(f"{pnls[n].roi:>16.2%}" for n in bet_cols))
    w("  " + f"{'hit rate':<16}" + "".join(f"{pnls[n].hit_rate:>16.2%}" for n in bet_cols))
    w("=" * (24 + 16 * len(columns)))
    if dc_rag is not None:
        d_ll = metrics["Dixon-Coles"]["log_loss"] - metrics["DC+RAG"]["log_loss"]
        d_br = metrics["Dixon-Coles"]["brier"] - metrics["DC+RAG"]["brier"]
        verdict = "IMPROVES" if (d_ll > 0 and d_br > 0) else "does NOT improve"
        w(f"  RAG verdict: signal {verdict} the DC baseline "
          f"(Δlog-loss={d_ll:+.4f}, ΔBrier={d_br:+.4f}; positive = better).")
    w("  NOTE: base-rate-priced PnL is a predictive-accuracy proxy, NOT a live")
    w("  edge — real markets price sharper than the unconditional base rate.")
    w("=" * (24 + 16 * len(columns)))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start-year", type=int, default=2006, help="first evaluation year")
    parser.add_argument("--step-months", type=int, default=12, help="refit cadence in months")
    parser.add_argument("--dc-lookback-days", type=int, default=2000,
                        help="Dixon-Coles training window (None-like 0 = full history)")
    parser.add_argument("--min-train-matches", type=int, default=500, help="warmup gate")
    parser.add_argument("--wc-only", action="store_true", help="score only World Cup fixtures")
    parser.add_argument("--edge-threshold", type=float, default=0.05, help="flat-bet edge vs base rate")
    parser.add_argument("--flat-stake", type=float, default=100.0, help="flat bet size")
    parser.add_argument("--data-path", default=None, help="override international results CSV path")
    parser.add_argument("--use-rag", action="store_true",
                        help="add a Dixon-Coles+RAG column fusing L2 news sentiment (mock)")
    parser.add_argument("--rag-lookback-days", type=int, default=7,
                        help="point-in-time news window per match for --use-rag")
    parser.add_argument("--verbose", action="store_true", help="DEBUG logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    config = BacktestConfig(
        start_year=args.start_year,
        step_months=args.step_months,
        dc_lookback_days=(args.dc_lookback_days or None),
        min_train_matches=args.min_train_matches,
        wc_only=args.wc_only,
        edge_threshold=args.edge_threshold,
        flat_stake=args.flat_stake,
        data_path=args.data_path,
        use_rag=args.use_rag,
        rag_lookback_days=args.rag_lookback_days,
    )

    store = FeatureStore(config.data_path)
    elo_run, dc_run, dc_rag_run, truth, meta = run_backtest(store, config)

    if len(truth) == 0:
        logger.error("No matches scored — try a lower --start-year or --min-train-matches.")
        return 1

    dc_rag_arr = dc_rag_run.array() if dc_rag_run is not None else None
    report = format_report(elo_run.array(), dc_run.array(), truth, config, meta, dc_rag=dc_rag_arr)
    print("\n" + report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
