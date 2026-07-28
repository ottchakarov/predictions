"""L6 - Decision engine + L7 sizing: edge detection and Kelly staking.

This is where the project either has a business or doesn't. Following Miller &
Davidow (*The Logic of Sports Betting*), a prediction is only *tradable* when our
estimated probability beats the market's price by more than the friction (vig,
spread, fees). The model being "right" is necessary but not sufficient.

Steps:
1. Remove the vig from the raw market prices so we compare against a *fair*
   implied probability (multiplicative / "Shin-lite" normalisation).  # execution.mdc: de-vig before EV
2. Compute the edge = model_prob - fair_market_prob.
3. Require edge > target_edge (a buffer that absorbs fees + model error).
4. Size with fractional Kelly (Ernest Chan / risk management): full Kelly is
   growth-optimal but far too volatile in practice.
5. Apply the on-chain SharpTracker as a *sizing multiplier only* (never a trigger):
   scale the stake by sharp confirmation, and VETO (stake -> 0) when sharp capital
   heavily contradicts the model. All sharp data is read strictly as-of the
   decision timestamp (point-in-time; core.mdc / data_hygiene.mdc).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Dict, Optional, Sequence

if TYPE_CHECKING:  # avoid import cost / cycles; only needed for type hints
    from .ingest import SharpTracker, WalletStat


@dataclass
class DecisionConfig:
    target_edge: float = 0.04       # min model-vs-market edge to act on (4 pts)
    kelly_fraction: float = 0.25    # fractional Kelly multiplier
    max_stake_fraction: float = 0.05  # hard cap: never risk >5% of bankroll on one bet
    min_price: float = 0.02         # ignore dust / illiquid extremes
    max_price: float = 0.98
    # Sharp-money sizing modifier (applied only to bets that already cleared the
    # edge filter). Full confirmation scales the stake up to sharp_max_boost;
    # if contradicting sharp capital share exceeds sharp_veto_threshold -> veto.
    sharp_max_boost: float = 1.5
    sharp_veto_threshold: float = 0.60


@dataclass
class TradeSignal:
    outcome: str
    model_prob: float
    market_price: float        # the price you would pay (the ask / quote)
    fair_market_prob: float    # vig-removed market probability
    edge: float                # model_prob - fair_market_prob
    ev_per_dollar: float       # expected profit per $1 staked
    kelly_fraction: float      # APPLIED fractional-Kelly size (post sharp multiplier)
    stake: float               # kelly_fraction * bankroll
    is_bet: bool               # passed the target_edge filter (and not vetoed)
    sharp_multiplier: float = 1.0      # L6 sizing factor applied (1.0 = neutral)
    net_sharp_alignment: float = 0.0   # (confirming - contradicting) / total in [-1, 1]


def remove_vig(prices: Sequence[float]) -> list[float]:
    """Normalise raw outcome prices to fair probabilities summing to 1.

    Polymarket's complementary YES/NO tokens already sum to ~1, but order-book
    midpoints and multi-outcome markets generally don't, so we always normalise.
    """
    total = sum(p for p in prices if p is not None)
    if total <= 0:
        raise ValueError("Cannot remove vig from non-positive price total")
    return [(p / total) if p is not None else 0.0 for p in prices]


def kelly_fraction(model_prob: float, price: float) -> float:
    """Optimal Kelly fraction for a binary bet bought at ``price``.

    For a contract paying $1 on win, bought at ``price`` p with win prob q::

        f* = (q - p) / (1 - p)

    Negative values mean "no bet" (the market price is above our probability).
    """
    if not 0.0 < price < 1.0:
        return 0.0
    f = (model_prob - price) / (1.0 - price)
    return max(f, 0.0)


def sharp_sizing(
    tracker: Optional["SharpTracker"],
    wallets: "Sequence[WalletStat]",
    token_id: str,
    *,
    as_of: Optional[datetime],
    config: DecisionConfig,
) -> tuple[float, float, bool]:
    """Compute the (multiplier, net_alignment, vetoed) sharp-money sizing signal.

    Sharp activity is a *sizing modifier only*. Capital long the target token
    confirms our model; capital short it (i.e. on the opposing side) contradicts.
    Returns the SAFE DEFAULT (1.0, 0.0, False) — i.e. raw Kelly — whenever there
    is no tracker, no sharp wallets/positions, or the tracker errors out.

    Point-in-time: positions are read strictly as-of ``as_of`` (core.mdc).
    """
    if tracker is None or not wallets or not token_id:
        return 1.0, 0.0, False

    try:
        positions = tracker.get_recent_positions(
            [w.address for w in wallets], token_id, as_of=as_of
        )
    except Exception:  # noqa: BLE001 - any tracker failure must not size a bet up
        return 1.0, 0.0, False

    if not positions:
        return 1.0, 0.0, False

    # Capital = |shares| * entry price, split by side of the target outcome.
    confirming = sum(p.size * p.avg_price for p in positions.values() if p.size > 0)
    contradicting = sum(-p.size * p.avg_price for p in positions.values() if p.size < 0)
    total = confirming + contradicting
    if total <= 0:
        return 1.0, 0.0, False

    net_alignment = (confirming - contradicting) / total      # in [-1, 1]
    contradicting_share = contradicting / total

    # Strict veto: heavy contradicting consensus kills the bet outright.
    if contradicting_share > config.sharp_veto_threshold:
        return 0.0, net_alignment, True

    # Linear scaling: net=+1 -> sharp_max_boost, net=0 -> 1.0, net<0 -> shrink.
    multiplier = 1.0 + net_alignment * (config.sharp_max_boost - 1.0)
    multiplier = max(0.0, min(multiplier, config.sharp_max_boost))
    return multiplier, net_alignment, False


def evaluate_outcome(
    outcome: str,
    model_prob: float,
    market_price: float,
    fair_market_prob: float,
    bankroll: float,
    config: DecisionConfig,
    *,
    sharp: Optional["SharpTracker"] = None,
    sharp_wallets: "Optional[Sequence[WalletStat]]" = None,
    token_id: str = "",
    as_of: Optional[datetime] = None,
) -> TradeSignal:
    """Build a (possibly non-acting) TradeSignal for one outcome token."""
    edge = model_prob - fair_market_prob
    # EV per $1 staked buying at market_price: q*(1-p) - (1-q)*p == q - p.
    ev_per_dollar = (model_prob - market_price) / market_price if market_price > 0 else 0.0

    raw_kelly = kelly_fraction(model_prob, market_price) * config.kelly_fraction
    capped_kelly = min(raw_kelly, config.max_stake_fraction)

    tradable_price = config.min_price <= market_price <= config.max_price
    base_is_bet = bool(edge > config.target_edge and capped_kelly > 0 and tradable_price)

    # Sharp-money sizing only applies to bets that already cleared the edge filter
    # (it is a modifier, never a trigger).
    multiplier, net_alignment = 1.0, 0.0
    if base_is_bet and sharp is not None:
        multiplier, net_alignment, _vetoed = sharp_sizing(
            sharp, sharp_wallets or [], token_id, as_of=as_of, config=config
        )

    # The hard per-bet bankroll cap is an absolute ceiling: a sharp boost may
    # scale UP toward it but never through it (execution.mdc: hard cap).
    applied_kelly = min(capped_kelly * multiplier, config.max_stake_fraction)
    is_bet = bool(base_is_bet and applied_kelly > 0)

    return TradeSignal(
        outcome=outcome,
        model_prob=model_prob,
        market_price=market_price,
        fair_market_prob=fair_market_prob,
        edge=edge,
        ev_per_dollar=ev_per_dollar,
        kelly_fraction=applied_kelly,
        stake=round(applied_kelly * bankroll, 2) if is_bet else 0.0,
        is_bet=is_bet,
        sharp_multiplier=multiplier,
        net_sharp_alignment=net_alignment,
    )


def evaluate_market(
    model_probs: Dict[str, float],
    market_prices: Dict[str, float],
    *,
    bankroll: float,
    config: Optional[DecisionConfig] = None,
    sharp: Optional["SharpTracker"] = None,
    as_of: Optional[datetime] = None,
    token_ids: Optional[Dict[str, str]] = None,
    required_outcomes: Optional[Sequence[str]] = None,
) -> list[TradeSignal]:
    """Evaluate every outcome of a market and return the signals.

    ``model_probs`` and ``market_prices`` are keyed by the same outcome labels.

    ALL-OR-NOTHING PRICING (execution.mdc): vig is removed across the outcomes in
    ``market_prices``, so that set MUST be the complete book. Renormalizing a
    PARTIAL book (e.g. a 1X2 market missing the draw quote) would rescale the
    survivors to sum to 1 and manufacture a fake edge. When ``required_outcomes``
    is given and any of them is absent from ``market_prices``, we therefore price
    NOTHING and return ``[]`` (the market is incomplete). Callers must pass the
    full required outcome set for multi-leg markets.

    If ``sharp`` is provided, the top sharp wallets are fetched ONCE (as-of
    ``as_of``) and used to scale each bet's stake; ``token_ids`` maps outcome
    labels to their CLOB token ids so positions can be looked up per side.
    """
    config = config or DecisionConfig()
    if required_outcomes is not None:
        missing = [o for o in required_outcomes if o not in market_prices]
        if missing:
            # Incomplete book: never renormalize the remainder into a fake 100%.
            return []
    outcomes = list(market_prices.keys())
    fair = dict(zip(outcomes, remove_vig([market_prices[o] for o in outcomes])))
    token_ids = token_ids or {}

    # Fetch the sharp wallet set once per market; any failure -> no sharp wallets
    # -> safe default raw Kelly downstream.
    sharp_wallets: "Sequence[WalletStat]" = []
    if sharp is not None:
        try:
            sharp_wallets = sharp.get_top_wallets(as_of=as_of)
        except Exception:  # noqa: BLE001
            sharp_wallets = []

    signals = []
    for outcome in outcomes:
        if outcome not in model_probs:
            continue
        signals.append(
            evaluate_outcome(
                outcome=outcome,
                model_prob=model_probs[outcome],
                market_price=market_prices[outcome],
                fair_market_prob=fair[outcome],
                bankroll=bankroll,
                config=config,
                sharp=sharp,
                sharp_wallets=sharp_wallets,
                token_id=token_ids.get(outcome, ""),
                as_of=as_of,
            )
        )
    return signals
