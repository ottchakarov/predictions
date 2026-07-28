import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wc_bot.polymarket import OrderBook, PolymarketClient  # noqa: E402


# --------------------------------------------------------------- spread guard
def _client_with_book(bid, ask, *, max_spread=0.05):
    book = OrderBook("T", best_bid=bid, best_ask=ask, bid_size=1e3, ask_size=1e3)
    return PolymarketClient(mock=True, mock_books={"T": book}, max_spread=max_spread)


def test_live_price_midpoint_from_top_of_book():
    price = _client_with_book(0.40, 0.42).get_live_price("T")
    assert price.is_reliable
    assert abs(price.midpoint - 0.41) < 1e-9
    assert abs(price.spread - 0.02) < 1e-9


def test_wide_spread_returns_safe_default():
    # 8c spread > 5c default -> midpoint suppressed so no phantom edge.
    price = _client_with_book(0.30, 0.38).get_live_price("T")
    assert not price.is_reliable
    assert price.midpoint is None
    assert "spread" in price.reason


def test_one_sided_book_returns_safe_default():
    price = _client_with_book(0.30, None).get_live_price("T")
    assert not price.is_reliable and price.midpoint is None


def test_crossed_book_returns_safe_default():
    price = _client_with_book(0.55, 0.45).get_live_price("T")
    assert not price.is_reliable and price.midpoint is None
    assert price.reason == "crossed book"


def test_custom_max_spread_override():
    client = _client_with_book(0.30, 0.38)
    assert client.get_live_price("T", max_spread=0.10).is_reliable  # 8c <= 10c


def test_unregistered_mock_token_is_safe_default_not_error():
    # A live-only watchlist entry under --mock must not crash the demo.
    client = PolymarketClient(mock=True, mock_books={})
    price = client.get_live_price("UNKNOWN")
    assert not price.is_reliable and price.midpoint is None


def test_empty_live_book_returns_none_midpoint():
    # A concluded market whose /book returns empty bid/ask lists -> safe default.
    client = PolymarketClient()
    client._get_json = lambda url, params=None: {"bids": [], "asks": []}  # type: ignore[assignment]
    price = client.get_live_price("RESOLVED")
    assert price.midpoint is None and not price.is_reliable
    assert price.best_bid is None and price.best_ask is None


def test_clob_404_for_concluded_market_is_safe_default_not_raise():
    # A resolved fixture (e.g. Croatia vs Ghana after kickoff) often 404s on /book.
    # get_live_price must swallow it and return a None midpoint, never raise.
    def boom(url, params=None):
        raise RuntimeError("404 Client Error: Not Found for /book")

    client = PolymarketClient()
    client._get_json = boom  # type: ignore[assignment]
    price = client.get_live_price("RESOLVED")
    assert price.midpoint is None and not price.is_reliable
    assert price.best_bid is None and price.best_ask is None
    assert "book unavailable" in price.reason


# ----------------------------------------------------- live parsing (no network)
def test_get_order_book_parses_real_clob_shape():
    # Real /book: bids ascending, asks descending; best = max bid / min ask.
    payload = {
        "bids": [{"price": "0.01", "size": "100"}, {"price": "0.23", "size": "5000"}],
        "asks": [{"price": "0.99", "size": "100"}, {"price": "0.24", "size": "4000"}],
    }
    client = PolymarketClient()
    client._get_json = lambda url, params=None: payload  # type: ignore[assignment]
    book = client.get_order_book("X")
    assert book.best_bid == 0.23 and book.best_ask == 0.24
    assert abs(book.mid - 0.235) < 1e-9


def test_get_market_metadata_parses_gamma_shape():
    payload = [
        {
            "conditionId": "0xabc",
            "question": "Will Croatia win on 2026-06-27?",
            "slug": "fifwc-hrv-gha-2026-06-27-hrv",
            "outcomes": '["Yes", "No"]',
            "clobTokenIds": '["TOK_YES", "TOK_NO"]',
            "active": True,
            "closed": False,
            "enableOrderBook": True,
        }
    ]
    client = PolymarketClient()
    client._get_json = lambda url, params=None: payload  # type: ignore[assignment]
    meta = client.get_market_metadata("TOK_NO")
    assert meta is not None
    assert meta.condition_id == "0xabc"
    assert meta.outcome == "No"  # mapped token id -> outcome label
    assert meta.is_tradable


def test_get_market_metadata_flags_closed_market_not_tradable():
    payload = [
        {
            "conditionId": "0xabc",
            "outcomes": '["Yes", "No"]',
            "clobTokenIds": '["TOK_YES", "TOK_NO"]',
            "active": True,
            "closed": True,
            "enableOrderBook": True,
        }
    ]
    client = PolymarketClient()
    client._get_json = lambda url, params=None: payload  # type: ignore[assignment]
    assert not client.get_market_metadata("TOK_YES").is_tradable


def test_mock_mode_metadata_returns_none():
    assert PolymarketClient(mock=True).get_market_metadata("anything") is None
