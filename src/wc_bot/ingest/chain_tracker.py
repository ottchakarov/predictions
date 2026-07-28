"""L1 - On-Chain Sharp Money Tracker (scaffold).

Tracks historically profitable Polymarket wallets on Polygon and exposes their
activity as a **position-sizing modifier** — NOT an execution trigger. Sharp
agreement may scale an already-+EV bet up (or disagreement scale it down); it
must never, on its own, open a position. The edge/Kelly logic in `decision.py`
remains the sole arbiter of whether a bet exists.

POINT-IN-TIME CONTRACT (core.mdc, data_hygiene.mdc):
This is an L1 feature source, so every method is `as_of`-bound and MUST only ever
consider on-chain events whose block timestamp is STRICTLY before `as_of`:
* `get_top_wallets` ROI MUST be computed from positions *resolved* before
  `as_of` — never from open positions whose outcome is not yet known, and never
  using a market's final resolution that post-dates `as_of` (that is look-ahead).
* `get_recent_positions` MUST return only fills with block time `< as_of`.
Backtests MUST pass the decision timestamp; live polling passes `now()`.

DATA SOURCES (in order of preference for the live implementation)
=================================================================
There are three layers we can pull from. Prefer (A) the subgraph for point-in-time
correctness (native `timestamp_lt` filtering), fall back to (B) raw RPC logs, and
use (C) Gamma/Data REST only for metadata and sanity checks.

(A) Goldsky subgraphs (RECOMMENDED — leakage-safe by construction)
    Polymarket publishes GraphQL subgraphs indexing the Exchange + CTF. Each entity
    carries a block `timestamp`, so the entire point-in-time contract reduces to a
    `where: { timestamp_lt: <as_of_unix> }` clause — no manual block bisection.
      * activity/orders subgraph -> `OrderFilled` / `enrichedOrderFilled` entities
        (maker, taker, makerAssetId, takerAssetId, makerAmountFilled,
        takerAmountFilled, fee, timestamp).
      * positions subgraph -> per-(user, tokenId) net `amount` + cost basis.
    Example (recent fills for a token, strictly before as_of):
        POST <SUBGRAPH_URL>
        {
          "query": "query($tok:String!,$ts:BigInt!){
            enrichedOrderFilleds(
              where:{ market:$tok, timestamp_lt:$ts }
              orderBy:timestamp orderDirection:desc first:1000
            ){ maker taker side size price timestamp }
          }",
          "variables": { "tok": "<token_id>", "ts": <as_of_unix> }
        }

(B) Polygon JSON-RPC `eth_getLogs` (fallback / verification)
    Decode events directly from the two settlement contracts. Requires mapping
    `as_of` -> block number first (see `_block_at`), then querying logs in
    [fromBlock, toBlock=block_at(as_of)] so nothing after the cutoff leaks in.
    Relevant events:
      * CTF ERC-1155 `TransferSingle(operator,from,to,id,value)`
        topic0 = 0xc3d58168c5ae7397731d063d5bbf3d657854427343f4c083240f7aacaa2d0f62
        topic0 `TransferBatch` = 0x4a39dc06d4c0dbc64b70af90fd698a233a518aa5d07e595d983b8c0526c8f7fb
        (from/to/operator are indexed topics; id + value live in `data`).
      * CTF Exchange `OrderFilled(orderHash,maker,taker,makerAssetId,
        takerAssetId,makerAmountFilled,takerAmountFilled,fee)` -> gives execution
        price (USDC assetId == 0) and is the source of cost basis.
    Example request:
        POST <rpc_url>  {"jsonrpc":"2.0","id":1,"method":"eth_getLogs","params":[{
          "address": CTF_EXCHANGE,
          "topics": [ORDER_FILLED_TOPIC],
          "fromBlock":"0x...", "toBlock": hex(block_at(as_of)) }]}
    `value`/amounts decode as uint256; CTF shares are 1e6-scaled like USDC.

(C) REST metadata (NOT point-in-time — never use its live state for backtests)
    * Gamma `GET /markets?closed=true&<category filter>` -> conditionId,
      clobTokenIds (the two ERC-1155 outcome ids), slug, endDate, umaResolution
      status. Used only to (i) scope `market_filter` and (ii) get each market's
      resolution time + winning outcome for realized-PnL accounting.
    * Data API `GET /holders?market=<conditionId>` and `GET /positions?user=` are
      "as-of-now" snapshots only — use for live monitoring, NEVER for historical
      ROI (they would inject look-ahead).

KEY: a wallet "confirms" our target outcome when it is net-LONG that outcome's
token; "contradicts" when net-short (i.e. long the complementary token). The
decision engine weights by capital = |size| * avg_price.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

GAMMA_BASE = "https://gamma-api.polymarket.com"
DATA_API_BASE = "https://data-api.polymarket.com"
DEFAULT_POLYGON_RPC = "https://polygon-rpc.com"

# Polymarket's live Goldsky orderbook subgraph (raw OrderFilledEvent fills, each
# carrying a `timestamp` so point-in-time filtering is a `timestamp_lt` clause).
# The legacy positions/pnl subgraphs were deprecated; raw fills are the source of
# truth and are aggregated client-side.
ORDERBOOK_SUBGRAPH = (
    "https://api.goldsky.com/api/public/"
    "project_cl6mb8i9h0003e201j6li0diw/subgraphs/orderbook-subgraph/prod/gn"
)

# Polymarket settlement contracts on Polygon (mainnet, chainId 137).
CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"            # ERC-1155 ConditionalTokens
USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"           # collateral
CTF_EXCHANGE = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"           # CTF Exchange
NEG_RISK_EXCHANGE = "0xC5d563A36AE78145C45a50134d48A1215220f80a"      # Neg-risk (multi-outcome)

# In OrderFilledEvent, USDC collateral is asset id "0"; the other side is the
# ERC-1155 outcome token id. Amounts are 1e6 fixed-point (like USDC).
USDC_ASSET_ID = "0"
FIXED_POINT = 1_000_000.0
# Order matching is routed through these operator addresses; they appear as the
# `taker` on most fills and must never be counted as a "sharp wallet".
_EXCHANGE_ADDRESSES = {CTF_EXCHANGE.lower(), NEG_RISK_EXCHANGE.lower()}
# The Graph caps `first` at 1000; bound how many recent fills we scan per call.
_MAX_FILLS = 1000

# Event topic0 hashes (keccak256 of the canonical event signature). The two
# ERC-1155 transfer topics below are the OpenZeppelin standard and are stable.
TRANSFER_SINGLE_TOPIC = "0xc3d58168c5ae7397731d063d5bbf3d657854427343f4c083240f7aacaa2d0f62"
TRANSFER_BATCH_TOPIC = "0x4a39dc06d4c0dbc64b70af90fd698a233a518aa5d07e595d983b8c0526c8f7fb"
# OrderFilled is a Polymarket-specific signature; compute & pin before use, e.g.
#   Web3.keccak(text="OrderFilled(bytes32,address,address,uint256,uint256,uint256,uint256,uint256)").hex()
# (left as None so a wrong magic constant can't silently mis-filter logs).
ORDER_FILLED_TOPIC: Optional[str] = None


@dataclass(frozen=True)
class WalletStat:
    """A profitable wallet's track record, computed strictly as-of a cutoff."""

    address: str
    realized_roi: float          # ROI over positions RESOLVED before as_of
    realized_pnl_usdc: float
    resolved_trades: int
    as_of: datetime


@dataclass(frozen=True)
class WalletPosition:
    """A wallet's holding in one outcome token, known as-of a cutoff."""

    wallet: str
    token_id: str
    size: float                  # net shares held (signed; >0 long the outcome)
    avg_price: float             # average entry price in USDC
    as_of: datetime


class SharpTracker:
    """Queries Polymarket's Goldsky subgraph for sharp-wallet activity (read-only).

    Live methods aggregate raw ``OrderFilledEvent`` fills from the orderbook
    subgraph, filtered ``timestamp_lt`` the decision cutoff so nothing after
    ``as_of`` ever leaks in (data_hygiene.mdc). Every network call is wrapped so a
    subgraph outage / rate-limit / timeout degrades to the SAFE DEFAULT (empty
    result -> raw Kelly), never an exception that could kill the polling loop.
    """

    def __init__(
        self,
        *,
        rpc_url: str = DEFAULT_POLYGON_RPC,
        gamma_base: str = GAMMA_BASE,
        data_api_base: str = DATA_API_BASE,
        subgraph_url: str = ORDERBOOK_SUBGRAPH,
        polygonscan_api_key: Optional[str] = None,
        top_n: int = 20,
        timeout: float = 15.0,
        mock: bool = False,
    ) -> None:
        self.rpc_url = rpc_url
        self.gamma_base = gamma_base.rstrip("/")
        self.data_api_base = data_api_base.rstrip("/")
        self.subgraph_url = subgraph_url  # Goldsky GraphQL endpoint (orderbook)
        self.polygonscan_api_key = polygonscan_api_key
        self.top_n = top_n
        self.timeout = timeout
        self.mock = mock
        self._session = None  # lazy requests.Session so import works without it

    # ------------------------------------------------------------- discovery
    def get_top_wallets(
        self,
        market_filter: str = "Sports",
        min_roi: float = 0.10,
        *,
        as_of: Optional[datetime] = None,
    ) -> List[WalletStat]:
        """Return up to ``top_n`` wallets ranked by realized profit, ROI ≥ ``min_roi``.

        Profit is aggregated only from fills with ``timestamp`` STRICTLY before
        ``as_of`` (point-in-time; data_hygiene.mdc). ``market_filter`` is accepted
        for interface stability but the orderbook subgraph has no category, so it
        is best-effort (ranking spans all recent fills). Returns ``[]`` in mock
        mode and on ANY subgraph failure (logged), so the caller falls back to raw
        Kelly rather than crashing.
        """
        as_of = self._resolve_as_of(as_of)
        if self.mock:
            return []

        # SAFETY (execution.mdc / data_hygiene.mdc): the live ranking below is a
        # recent-fill CASHFLOW PROXY, not fully-resolved PnL. It ignores open
        # positions whose markets have not yet settled and any redemption value
        # realized after as_of, so the implied "ROI" can badly misrank wallets.
        # We keep the proxy for mock/scaffold testing but loudly flag it so it is
        # never silently trusted for live position sizing.
        logger.warning(
            "Cashflow-proxy ROI is unsafe for live sharp tracking; fully resolved "
            "PnL indexer required. get_top_wallets is ranking on recent-fill "
            "cashflow (open positions and post-as_of redemptions are not settled)."
        )

        as_of_unix = int(as_of.timestamp())
        # Scan the most recent fills strictly before as_of and aggregate realized
        # USD cash-flow per maker (makers are real traders; takers are the exchange
        # operator). This is a leakage-safe realized-profit proxy: no fill at or
        # after as_of is ever read, so a backtest sees only what was known then.
        query = """
        query TopWallets($ts: BigInt!, $first: Int!) {
          orderFilledEvents(
            first: $first
            orderBy: timestamp
            orderDirection: desc
            where: { timestamp_lt: $ts }
          ) {
            maker
            makerAssetId
            takerAssetId
            makerAmountFilled
            takerAmountFilled
            timestamp
          }
        }
        """
        try:
            data = self._graphql(query, {"ts": str(as_of_unix), "first": _MAX_FILLS})
            fills = data.get("orderFilledEvents", [])
        except Exception as exc:  # noqa: BLE001 - degrade to safe default, never crash
            logger.warning("SharpTracker.get_top_wallets subgraph call failed: %s", exc)
            return []

        # wallet -> [net_usd_cashflow, gross_usd_spent]
        agg: Dict[str, List[float]] = {}
        for fill in fills:
            maker = str(fill.get("maker", "")).lower()
            if not maker or maker in _EXCHANGE_ADDRESSES:
                continue
            usd, spent = _fill_cashflow(fill)
            if usd is None:
                continue
            bucket = agg.setdefault(maker, [0.0, 0.0])
            bucket[0] += usd
            bucket[1] += spent

        stats: List[WalletStat] = []
        for addr, (net_usd, spent) in agg.items():
            if spent <= 0:
                continue
            roi = net_usd / spent
            if roi < min_roi:
                continue
            stats.append(
                WalletStat(
                    address=addr,
                    realized_roi=roi,
                    realized_pnl_usdc=net_usd,
                    resolved_trades=0,  # fill-level proxy; redemptions not counted
                    as_of=as_of,
                )
            )
        stats.sort(key=lambda w: w.realized_pnl_usdc, reverse=True)
        return stats[: self.top_n]

    # -------------------------------------------------------------- positions
    def get_recent_positions(
        self,
        wallet_addresses: Sequence[str],
        token_id: str,
        *,
        as_of: Optional[datetime] = None,
    ) -> Dict[str, WalletPosition]:
        """Net holding of each wallet in ``token_id`` as-of ``as_of``.

        Aggregates ``OrderFilledEvent`` fills where the wallet is the ``maker`` and
        ``token_id`` is one side of the trade, with ``timestamp`` STRICTLY before
        ``as_of`` (core.mdc) -- so a backtest can never see a position the wallet
        had not yet taken. ``size`` is signed (>0 long the outcome), ``avg_price``
        is the cost-weighted entry. Keyed by lowercased wallet address; wallets
        with a zero net are omitted.

        Returns ``{}`` in mock mode and on ANY subgraph failure (logged).
        """
        as_of = self._resolve_as_of(as_of)
        if self.mock or not wallet_addresses or not token_id:
            return {}

        as_of_unix = int(as_of.timestamp())
        makers = [a.lower() for a in wallet_addresses]
        query = """
        query Positions($ts: BigInt!, $tok: String!, $makers: [String!], $first: Int!) {
          orderFilledEvents(
            first: $first
            orderBy: timestamp
            orderDirection: desc
            where: {
              or: [
                { timestamp_lt: $ts, maker_in: $makers, makerAssetId: $tok }
                { timestamp_lt: $ts, maker_in: $makers, takerAssetId: $tok }
              ]
            }
          ) {
            maker
            makerAssetId
            takerAssetId
            makerAmountFilled
            takerAmountFilled
          }
        }
        """
        try:
            data = self._graphql(
                query,
                {
                    "ts": str(as_of_unix),
                    "tok": token_id,
                    "makers": makers,
                    "first": _MAX_FILLS,
                },
            )
            fills = data.get("orderFilledEvents", [])
        except Exception as exc:  # noqa: BLE001 - degrade to safe default, never crash
            logger.warning(
                "SharpTracker.get_recent_positions subgraph call failed: %s", exc
            )
            return {}

        # wallet -> [net_shares, usd_spent_buying, shares_bought]
        agg: Dict[str, List[float]] = {}
        for fill in fills:
            maker = str(fill.get("maker", "")).lower()
            shares, usd_spent_buy, shares_bought = _token_delta(fill, token_id)
            if shares == 0.0 and shares_bought == 0.0:
                continue
            bucket = agg.setdefault(maker, [0.0, 0.0, 0.0])
            bucket[0] += shares
            bucket[1] += usd_spent_buy
            bucket[2] += shares_bought

        positions: Dict[str, WalletPosition] = {}
        for addr, (net_shares, usd_spent, shares_bought) in agg.items():
            if abs(net_shares) < 1e-9:
                continue
            avg_price = (usd_spent / shares_bought) if shares_bought > 0 else 0.0
            positions[addr] = WalletPosition(
                wallet=addr,
                token_id=token_id,
                size=net_shares,
                avg_price=avg_price,
                as_of=as_of,
            )
        return positions

    # ---------------------------------------------------------------- helpers
    def _block_at(self, as_of: datetime) -> int:
        """Map a UTC cutoff to the latest Polygon block with timestamp < as_of.

        Two implementations, both point-in-time safe:
        * Polygonscan: `GET api?module=block&action=getblocknobytime&closest=before
          &timestamp={int(as_of.timestamp())}&apikey={polygonscan_api_key}`.
        * Pure RPC: binary-search `eth_getBlockByNumber` between a known lower bound
          and `eth_blockNumber`, choosing the greatest block whose `timestamp`
          (hex seconds) is strictly < as_of. This block is the inclusive toBlock
          for every `eth_getLogs` query so nothing after the cutoff can leak in.
        """
        raise NotImplementedError("Wire Polygonscan getblocknobytime or RPC bisection.")

    @staticmethod
    def _resolve_as_of(as_of: Optional[datetime]) -> datetime:
        """Default to now() for live use; enforce tz-aware UTC for comparisons."""
        if as_of is None:
            return datetime.now(timezone.utc)
        return as_of if as_of.tzinfo else as_of.replace(tzinfo=timezone.utc)

    # --------------------------------------------------------------- transport
    def _get_session(self):
        if self._session is None:
            import requests

            self._session = requests.Session()
            self._session.headers.update({"User-Agent": "wc-bot/0.1 (read-only)"})
        return self._session

    def _graphql(self, query: str, variables: dict) -> dict:
        """POST a GraphQL query to the subgraph; raise on transport/GraphQL errors.

        Callers wrap this so any failure (timeout, rate-limit, indexing error)
        degrades to the safe default. GraphQL returns HTTP 200 even for query
        errors, so we explicitly check the ``errors`` field.
        """
        resp = self._get_session().post(
            self.subgraph_url,
            json={"query": query, "variables": variables},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("errors"):
            raise RuntimeError(f"subgraph GraphQL errors: {payload['errors']}")
        return payload.get("data") or {}


def _to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _fill_cashflow(fill: dict):
    """Maker's signed USDC cash-flow for one fill: (net_usd, usd_spent_buying).

    Returns ``(None, 0.0)`` if neither side is USDC (token<->token swaps, which the
    cash-flow proxy can't price). Amounts are 1e6 fixed-point.
    """
    maker_asset = str(fill.get("makerAssetId", ""))
    taker_asset = str(fill.get("takerAssetId", ""))
    maker_amt = _to_float(fill.get("makerAmountFilled")) / FIXED_POINT
    taker_amt = _to_float(fill.get("takerAmountFilled")) / FIXED_POINT

    if maker_asset == USDC_ASSET_ID:
        # Maker gave USDC (bought the token): cash out.
        return -maker_amt, maker_amt
    if taker_asset == USDC_ASSET_ID:
        # Maker received USDC (sold the token): cash in.
        return taker_amt, 0.0
    return None, 0.0


def _token_delta(fill: dict, token_id: str):
    """Maker's (signed_shares, usd_spent_buying, shares_bought) for ``token_id``.

    >0 shares = maker accumulated the outcome token (bought); <0 = sold. Only the
    buy leg contributes to cost basis (usd_spent_buying / shares_bought).
    """
    maker_asset = str(fill.get("makerAssetId", ""))
    taker_asset = str(fill.get("takerAssetId", ""))
    maker_amt = _to_float(fill.get("makerAmountFilled")) / FIXED_POINT
    taker_amt = _to_float(fill.get("takerAmountFilled")) / FIXED_POINT

    if taker_asset == token_id and maker_asset == USDC_ASSET_ID:
        # Maker paid USDC (maker_amt) to receive token (taker_amt): BUY.
        return taker_amt, maker_amt, taker_amt
    if maker_asset == token_id and taker_asset == USDC_ASSET_ID:
        # Maker gave token (maker_amt) for USDC (taker_amt): SELL.
        return -maker_amt, 0.0, 0.0
    return 0.0, 0.0, 0.0
