import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wc_bot.decision import DecisionConfig, evaluate_market  # noqa: E402
from wc_bot.ingest import WalletPosition, WalletStat  # noqa: E402

AS_OF = datetime(2026, 6, 1, tzinfo=timezone.utc)
TOKEN = "TOK_YES"
# Generous cap so the sharp multiplier (not the cap) drives the stake in tests.
CFG = DecisionConfig(
    target_edge=0.04,
    kelly_fraction=0.25,
    max_stake_fraction=1.0,
    sharp_max_boost=1.5,
    sharp_veto_threshold=0.60,
)
# model 0.60 vs fair 0.50 -> kelly (0.6-0.5)/(1-0.5)=0.2 *0.25 = 0.05 -> $50 raw.
MODEL = {"Yes": 0.60}
PRICES = {"Yes": 0.50, "No": 0.50}
TOKENS = {"Yes": TOKEN}
BANKROLL = 1000.0
RAW_STAKE = 50.0


class FakeTracker:
    def __init__(self, positions=None, *, raise_on=None, wallets=None):
        self._positions = positions or {}
        self._raise_on = raise_on
        self._wallets = wallets if wallets is not None else [
            WalletStat("0xsharp1", 0.5, 1000.0, 20, AS_OF),
            WalletStat("0xsharp2", 0.4, 800.0, 15, AS_OF),
        ]

    def get_top_wallets(self, market_filter="Sports", min_roi=0.10, *, as_of=None):
        if self._raise_on == "wallets":
            raise RuntimeError("indexer down")
        return self._wallets

    def get_recent_positions(self, wallet_addresses, token_id, *, as_of=None):
        if self._raise_on == "positions":
            raise RuntimeError("rpc down")
        return self._positions


def _pos(wallet, size, price=0.5):
    return WalletPosition(wallet=wallet, token_id=TOKEN, size=size, avg_price=price, as_of=AS_OF)


def _yes(tracker):
    signals = evaluate_market(
        MODEL, PRICES, bankroll=BANKROLL, config=CFG,
        sharp=tracker, as_of=AS_OF, token_ids=TOKENS,
    )
    return next(s for s in signals if s.outcome == "Yes")


def test_baseline_without_tracker_is_raw_kelly():
    s = evaluate_market(MODEL, PRICES, bankroll=BANKROLL, config=CFG)[0]
    assert s.is_bet
    assert s.stake == RAW_STAKE
    assert s.sharp_multiplier == 1.0


def test_confirming_sharp_scales_up():
    # All sharp capital long the target outcome -> net=+1 -> boost to 1.5x.
    s = _yes(FakeTracker({"0xsharp1": _pos("0xsharp1", +100)}))
    assert s.is_bet
    assert s.sharp_multiplier > 1.0
    assert s.stake > RAW_STAKE
    assert abs(s.stake - RAW_STAKE * 1.5) < 1e-6


def test_contradicting_minority_scales_down():
    # 25 contra vs 20 confirming -> contra share 0.555 (< veto) -> shrink below raw.
    s = _yes(
        FakeTracker({"0xsharp1": _pos("0xsharp1", +40), "0xsharp2": _pos("0xsharp2", -50)})
    )
    assert s.is_bet
    assert 0.0 < s.stake < RAW_STAKE
    assert s.net_sharp_alignment < 0


def test_heavy_contradiction_triggers_veto():
    # contra share 0.70 (> 0.60) -> strict veto -> stake 0, not a bet.
    s = _yes(
        FakeTracker({"0xsharp1": _pos("0xsharp1", +30), "0xsharp2": _pos("0xsharp2", -70)})
    )
    assert not s.is_bet
    assert s.stake == 0.0
    assert s.sharp_multiplier == 0.0


def test_empty_positions_default_to_raw_kelly():
    s = _yes(FakeTracker({}))  # no positions
    assert s.is_bet
    assert s.stake == RAW_STAKE
    assert s.sharp_multiplier == 1.0


def test_no_sharp_wallets_default_to_raw_kelly():
    s = _yes(FakeTracker({"0xsharp1": _pos("0xsharp1", +100)}, wallets=[]))
    assert s.stake == RAW_STAKE
    assert s.sharp_multiplier == 1.0


def test_tracker_error_defaults_to_raw_kelly():
    assert _yes(FakeTracker(raise_on="wallets")).stake == RAW_STAKE
    assert _yes(FakeTracker(raise_on="positions")).stake == RAW_STAKE


def test_veto_only_applies_to_bets_that_cleared_edge():
    # No edge: model == fair -> not a bet regardless of sharp money.
    s = evaluate_market(
        {"Yes": 0.50}, PRICES, bankroll=BANKROLL, config=CFG,
        sharp=FakeTracker({"0xsharp1": _pos("0xsharp1", +100)}),
        as_of=AS_OF, token_ids=TOKENS,
    )[0]
    assert not s.is_bet
    assert s.sharp_multiplier == 1.0  # sharp logic skipped for non-bets
