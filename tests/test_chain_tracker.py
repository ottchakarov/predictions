import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wc_bot.ingest import SharpTracker  # noqa: E402
from wc_bot.ingest.chain_tracker import CTF_EXCHANGE  # noqa: E402

AS_OF = datetime(2026, 6, 1, tzinfo=timezone.utc)
TOK = "TOK"
M = 1_000_000  # 1e6 fixed point


def _buy(maker, usd, shares):
    """Maker pays USDC (asset 0) to receive `shares` of TOK."""
    return {
        "maker": maker,
        "makerAssetId": "0",
        "takerAssetId": TOK,
        "makerAmountFilled": str(int(usd * M)),
        "takerAmountFilled": str(int(shares * M)),
        "timestamp": "1700000000",
    }


def _sell(maker, shares, usd):
    """Maker gives `shares` of TOK to receive USDC."""
    return {
        "maker": maker,
        "makerAssetId": TOK,
        "takerAssetId": "0",
        "makerAmountFilled": str(int(shares * M)),
        "takerAmountFilled": str(int(usd * M)),
        "timestamp": "1700000000",
    }


def _tracker_returning(fills, *, capture=None):
    t = SharpTracker(mock=False)

    def fake_graphql(query, variables):
        if capture is not None:
            capture["query"] = query
            capture["variables"] = variables
        return {"orderFilledEvents": fills}

    t._graphql = fake_graphql  # type: ignore[assignment]
    return t


# --------------------------------------------------------------------- basics
def test_mock_mode_returns_empty_not_raises():
    tracker = SharpTracker(mock=True)
    assert tracker.get_top_wallets("Sports", min_roi=0.10, as_of=AS_OF) == []
    assert tracker.get_recent_positions(["0xabc"], TOK, as_of=AS_OF) == {}


def test_as_of_defaults_to_now_and_is_tz_aware():
    resolved = SharpTracker._resolve_as_of(None)
    assert resolved.tzinfo is not None
    naive = datetime(2026, 6, 1)
    assert SharpTracker._resolve_as_of(naive).tzinfo == timezone.utc


# ------------------------------------------------------------- positions parse
def test_get_recent_positions_nets_buys_and_sells():
    # Buy 1@0.30, buy 2@0.40 (0.80 total), sell 1 -> net +2 shares, avg from buys.
    fills = [_buy("0xWALLET", 0.30, 1), _buy("0xWALLET", 0.80, 2), _sell("0xWALLET", 1, 0.50)]
    pos = _tracker_returning(fills).get_recent_positions(["0xWALLET"], TOK, as_of=AS_OF)
    p = pos["0xwallet"]  # keyed lowercased
    assert abs(p.size - 2.0) < 1e-9
    assert abs(p.avg_price - (1.10 / 3.0)) < 1e-9  # cost-weighted over buys only


def test_get_recent_positions_short_is_negative_size():
    fills = [_buy("0xW", 0.30, 1), _sell("0xW", 3, 1.5)]  # net -2
    pos = _tracker_returning(fills).get_recent_positions(["0xW"], TOK, as_of=AS_OF)
    assert pos["0xw"].size < 0


def test_get_recent_positions_omits_zero_net():
    fills = [_buy("0xW", 0.30, 1), _sell("0xW", 1, 0.40)]  # net 0
    pos = _tracker_returning(fills).get_recent_positions(["0xW"], TOK, as_of=AS_OF)
    assert "0xw" not in pos


# ------------------------------------------------------------- top wallets rank
def test_get_top_wallets_ranks_and_filters_by_roi():
    fills = [
        _buy("0xA", 0.30, 1), _sell("0xA", 1, 0.50),   # +0.20 profit, roi 0.67
        _buy("0xB", 0.40, 1), _sell("0xB", 1, 0.45),   # +0.05 profit, roi 0.125
        _buy(CTF_EXCHANGE, 1.0, 1), _sell(CTF_EXCHANGE, 1, 5.0),  # exchange: excluded
    ]
    wallets = _tracker_returning(fills).get_top_wallets(min_roi=0.50, as_of=AS_OF)
    addrs = [w.address for w in wallets]
    assert addrs == ["0xa"]  # B filtered by min_roi; exchange excluded
    assert wallets[0].realized_pnl_usdc > 0


def test_live_top_wallets_warns_proxy_roi_is_unsafe(caplog):
    # The live ranking is a cashflow proxy, not resolved PnL. It must loudly warn
    # so it is never silently trusted for live sizing.
    fills = [_buy("0xA", 0.30, 1), _sell("0xA", 1, 0.50)]
    with caplog.at_level("WARNING"):
        _tracker_returning(fills).get_top_wallets(min_roi=0.0, as_of=AS_OF)
    assert any(
        "Cashflow-proxy ROI is unsafe for live sharp tracking" in r.message
        for r in caplog.records
    )


def test_mock_top_wallets_does_not_warn(caplog):
    # Mock mode short-circuits before the proxy ranking -> no false alarm.
    with caplog.at_level("WARNING"):
        assert SharpTracker(mock=True).get_top_wallets(as_of=AS_OF) == []
    assert not any("Cashflow-proxy ROI" in r.message for r in caplog.records)


def test_get_top_wallets_orders_by_profit_desc():
    fills = [
        _buy("0xA", 0.30, 1), _sell("0xA", 1, 0.50),   # +0.20
        _buy("0xB", 0.10, 1), _sell("0xB", 1, 1.00),   # +0.90
    ]
    wallets = _tracker_returning(fills).get_top_wallets(min_roi=0.0, as_of=AS_OF)
    assert [w.address for w in wallets] == ["0xb", "0xa"]


# ------------------------------------------------------ point-in-time + errors
def test_queries_pass_strict_as_of_filter():
    cap = {}
    t = _tracker_returning([], capture=cap)
    t.get_recent_positions(["0xW"], TOK, as_of=AS_OF)
    assert cap["variables"]["ts"] == str(int(AS_OF.timestamp()))
    assert "timestamp_lt" in cap["query"]  # anti-leakage: strictly before as_of


def test_subgraph_failure_returns_safe_default_not_raises():
    def boom(query, variables):
        raise RuntimeError("subgraph 429 rate limited")

    t = SharpTracker(mock=False)
    t._graphql = boom  # type: ignore[assignment]
    # Both must degrade to the safe default (-> raw Kelly downstream), not crash.
    assert t.get_top_wallets(as_of=AS_OF) == []
    assert t.get_recent_positions(["0xW"], TOK, as_of=AS_OF) == {}


def test_empty_inputs_short_circuit_without_network():
    # No wallets / no token must not even attempt a query.
    t = SharpTracker(mock=False)

    def boom(query, variables):  # pragma: no cover - must not be called
        raise AssertionError("should not query the subgraph")

    t._graphql = boom  # type: ignore[assignment]
    assert t.get_recent_positions([], TOK, as_of=AS_OF) == {}
    assert t.get_recent_positions(["0xW"], "", as_of=AS_OF) == {}
