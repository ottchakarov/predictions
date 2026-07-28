"""L5 - Market interface: read-only Polymarket client.

Polymarket exposes two relevant HTTP services (no wallet/auth needed for reads):

* Gamma API  (https://gamma-api.polymarket.com)  - market metadata & discovery.
  Each market carries ``outcomes``, ``outcomePrices`` and ``clobTokenIds`` as
  JSON-encoded strings, plus the ``conditionId`` used on-chain (Polygon/USDC) and
  ``active``/``closed``/``acceptingOrders`` status flags. Filter by token via
  ``/markets?clob_token_ids=<id>``.
* CLOB API   (https://clob.polymarket.com)        - the live central limit order
  book per outcome token. ``/book?token_id=...`` returns ``bids``/``asks`` (lists
  of ``{price, size}`` strings). We compute the market price as the midpoint of
  the best bid and best ask -- NEVER the last traded price (``last_trade_price``),
  which is stale and easily manipulated on thin books (execution.mdc).

EXECUTION DISCIPLINE (execution.mdc):
* The market price is the TOP-OF-BOOK MIDPOINT, not the last trade.
* If the bid/ask spread is wider than ``max_spread`` (default 5c) the book is too
  thin to trust: we return a "safe default" (midpoint=None, is_reliable=False) so
  the edge calculation skips it rather than hallucinating an edge off a stale quote.

This module is intentionally read-only: it never signs a transaction and has no
private-key path. Execution (L7) only writes to a CSV paper ledger.

A ``mock=True`` mode returns deterministic synthetic books so the pipeline and
tests run with no network access.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"

# Top-of-book spread (in probability/price units) beyond which a quote is deemed
# too thin to price an edge against. 5 cents per the execution rule.
DEFAULT_MAX_SPREAD = 0.05


@dataclass
class Market:
    """A Polymarket market, with its outcome tokens and last Gamma prices."""

    id: str
    question: str
    slug: str
    condition_id: str
    outcomes: List[str]
    token_ids: List[str]
    gamma_prices: List[float]
    active: bool
    closed: bool

    def token_for(self, outcome: str) -> Optional[str]:
        """Return the CLOB token id whose outcome label matches (case-insensitive)."""
        for label, token in zip(self.outcomes, self.token_ids):
            if label.strip().lower() == outcome.strip().lower():
                return token
        return None


@dataclass
class MarketMetadata:
    """Gamma status for a single outcome token (read-only, point-in-time 'now')."""

    token_id: str
    condition_id: str
    question: str
    slug: str
    outcome: str            # the outcome label this token represents (e.g. "Yes")
    active: bool
    closed: bool
    accepting_orders: bool

    @property
    def is_tradable(self) -> bool:
        """True only if the market is live and its book is accepting orders."""
        return self.active and (not self.closed) and self.accepting_orders


@dataclass
class OrderBook:
    token_id: str
    best_bid: Optional[float]
    best_ask: Optional[float]
    bid_size: float = 0.0
    ask_size: float = 0.0

    @property
    def mid(self) -> Optional[float]:
        """Midpoint price = implied probability for this outcome token."""
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2.0
        return self.best_bid if self.best_bid is not None else self.best_ask

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None


@dataclass
class LivePrice:
    """A spread-guarded market price for one token.

    ``midpoint`` is ``None`` (the safe default) whenever the quote is unreliable
    (empty/one-sided/crossed book, or spread wider than the configured maximum),
    so downstream edge logic skips it instead of trading off a bad number.
    """

    token_id: str
    best_bid: Optional[float]
    best_ask: Optional[float]
    spread: Optional[float]
    midpoint: Optional[float]
    is_reliable: bool
    reason: str = ""


class PolymarketClient:
    def __init__(
        self,
        *,
        gamma_base: str = GAMMA_BASE,
        clob_base: str = CLOB_BASE,
        timeout: float = 15.0,
        max_spread: float = DEFAULT_MAX_SPREAD,
        mock: bool = False,
        mock_books: Optional[Dict[str, OrderBook]] = None,
    ) -> None:
        self.gamma_base = gamma_base.rstrip("/")
        self.clob_base = clob_base.rstrip("/")
        self.timeout = timeout
        self.max_spread = max_spread
        self.mock = mock
        self._mock_books = mock_books or {}
        self._session = None  # lazily created so import works without `requests`

    # --------------------------------------------------------------- session
    def _get_session(self):
        if self._session is None:
            import requests

            self._session = requests.Session()
            self._session.headers.update({"User-Agent": "wc-bot/0.1 (read-only)"})
        return self._session

    def _get_json(self, url: str, params: Optional[dict] = None):
        resp = self._get_session().get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------ discovery
    def search_markets(self, query: str, *, limit: int = 20) -> List[Market]:
        """Find active markets whose question mentions ``query``."""
        if self.mock:
            return []
        # Gamma supports server-side text filtering; we also filter client-side to
        # be robust to API changes.
        raw = self._get_json(
            f"{self.gamma_base}/markets",
            params={"active": "true", "closed": "false", "limit": limit, "search": query},
        )
        markets = [self._parse_market(m) for m in _as_list(raw)]
        q = query.strip().lower()
        return [m for m in markets if m is not None and q in m.question.lower()]

    def get_market_by_slug(self, slug: str) -> Optional[Market]:
        if self.mock:
            return None
        raw = self._get_json(f"{self.gamma_base}/markets", params={"slug": slug})
        items = _as_list(raw)
        return self._parse_market(items[0]) if items else None

    def get_market_metadata(self, token_id: str) -> Optional[MarketMetadata]:
        """Resolve a CLOB token id to its Gamma market status (read-only).

        Used to confirm a watched token is still ``active`` / not ``closed`` and is
        accepting orders before we price it, and to surface the ``conditionId``
        (the on-chain resolution condition). Returns ``None`` in mock mode or when
        the token is unknown to Gamma.
        """
        if self.mock:
            return None
        raw = self._get_json(
            f"{self.gamma_base}/markets", params={"clob_token_ids": token_id}
        )
        items = _as_list(raw)
        if not items:
            return None
        item = items[0]
        market = self._parse_market(item)
        outcome = ""
        if market is not None:
            for label, tok in zip(market.outcomes, market.token_ids):
                if str(tok) == str(token_id):
                    outcome = label
                    break
        return MarketMetadata(
            token_id=token_id,
            condition_id=(market.condition_id if market else item.get("conditionId", "")),
            question=(market.question if market else item.get("question", "")),
            slug=(market.slug if market else item.get("slug", "")),
            outcome=outcome,
            active=bool(item.get("active", False)),
            closed=bool(item.get("closed", False)),
            accepting_orders=bool(
                item.get("acceptingOrders", item.get("enableOrderBook", False))
            ),
        )

    # -------------------------------------------------------------- pricing
    def get_order_book(self, token_id: str) -> OrderBook:
        if self.mock:
            # Unregistered tokens (e.g. a live-only watchlist entry under --mock)
            # return an empty book -> get_live_price treats it as an unreliable
            # quote and skips it, rather than crashing the offline demo.
            return self._mock_books.get(
                token_id, OrderBook(token_id=token_id, best_bid=None, best_ask=None)
            )

        raw = self._get_json(f"{self.clob_base}/book", params={"token_id": token_id})
        bids = raw.get("bids") or []
        asks = raw.get("asks") or []

        best_bid_lvl = max(bids, key=lambda lvl: float(lvl["price"]), default=None)
        best_ask_lvl = min(asks, key=lambda lvl: float(lvl["price"]), default=None)

        return OrderBook(
            token_id=token_id,
            best_bid=float(best_bid_lvl["price"]) if best_bid_lvl else None,
            best_ask=float(best_ask_lvl["price"]) if best_ask_lvl else None,
            bid_size=float(best_bid_lvl["size"]) if best_bid_lvl else 0.0,
            ask_size=float(best_ask_lvl["size"]) if best_ask_lvl else 0.0,
        )

    def implied_probability(self, token_id: str) -> Optional[float]:
        """Market-implied probability for an outcome token (order-book midpoint)."""
        return self.get_live_price(token_id).midpoint

    def get_live_price(
        self, token_id: str, *, max_spread: Optional[float] = None
    ) -> LivePrice:
        """Spread-guarded top-of-book midpoint for ``token_id``.

        Pulls the live CLOB book and prices off the BEST bid/ask midpoint (never
        the last trade). Returns a ``LivePrice`` whose ``midpoint`` is ``None`` when
        the quote can't be trusted:

        * empty or one-sided book (no bid or no ask),
        * crossed book (ask < bid), or
        * spread wider than ``max_spread`` (default ``self.max_spread``, 5c).

        This is the "safe default" that stops a thin/stale book from manufacturing
        a phantom edge (execution.mdc).

        A CONCLUDED market typically has no live CLOB book: ``/book`` may return an
        empty payload OR a 404 error. Both are handled here as an empty book
        (midpoint=None) so a settled fixture is skipped, never crashes the loop.
        """
        limit = self.max_spread if max_spread is None else max_spread
        try:
            book = self.get_order_book(token_id)
        except Exception as exc:  # noqa: BLE001 - resolved/closed market 404 or transient
            return LivePrice(
                token_id, None, None, None, None, False,
                f"book unavailable ({exc.__class__.__name__})",
            )
        bid, ask = book.best_bid, book.best_ask

        if bid is None or ask is None:
            return LivePrice(token_id, bid, ask, None, None, False, "empty/one-sided book")

        spread = ask - bid
        if spread < 0:
            return LivePrice(token_id, bid, ask, spread, None, False, "crossed book")
        if spread > limit:
            return LivePrice(
                token_id, bid, ask, spread, None, False,
                f"spread {spread:.3f} > max {limit:.3f}",
            )

        return LivePrice(token_id, bid, ask, spread, (bid + ask) / 2.0, True, "")

    # --------------------------------------------------------------- parsing
    @staticmethod
    def _parse_market(raw: dict) -> Optional[Market]:
        try:
            return Market(
                id=str(raw.get("id", "")),
                question=raw.get("question", ""),
                slug=raw.get("slug", ""),
                condition_id=raw.get("conditionId", ""),
                outcomes=_load_json_list(raw.get("outcomes")),
                token_ids=_load_json_list(raw.get("clobTokenIds")),
                gamma_prices=[float(x) for x in _load_json_list(raw.get("outcomePrices"))],
                active=bool(raw.get("active", False)),
                closed=bool(raw.get("closed", False)),
            )
        except (ValueError, TypeError):
            return None


def _as_list(raw) -> list:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and "data" in raw:
        return raw["data"]
    return [raw] if raw else []


def _load_json_list(value) -> list:
    """Gamma encodes list fields as JSON strings, e.g. '["Yes", "No"]'."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError):
        return []
