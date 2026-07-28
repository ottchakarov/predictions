#!/usr/bin/env python3
"""Run the World Cup paper-trading skeleton end-to-end.

Examples
--------
Offline demo (no network, uses 'mock_price' fields in the watchlist)::

    python scripts/run_skeleton.py --mock

Live, single pass against Polymarket (read-only CLOB midpoints)::

    python scripts/run_skeleton.py --live-odds

Live autonomous loop, polling every 90s::

    python scripts/run_skeleton.py --live-odds --loop --interval 90

Default (no flags) runs offline on the watchlist's 'mock_price' fields.

Toggle the L4 model (Elo is default); Dixon-Coles enforces a point-in-time cutoff::

    python scripts/run_skeleton.py --mock --model dixon_coles --as-of 2026-06-01

Fuse L2 RAG sentiment into Dixon-Coles (live news API + LLM with --live-odds)::

    python scripts/run_skeleton.py --live-odds --model dixon_coles --use-rag
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make the src/ package importable when run as a plain script.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no dependency): KEY=VALUE lines into os.environ.

    Existing environment variables take precedence (we only setdefault), so a real
    shell export always wins over the file.
    """
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

from wc_bot.decision import DecisionConfig  # noqa: E402
from wc_bot.pipeline import (  # noqa: E402
    MODEL_DIXON_COLES,
    MODEL_ELO,
    PipelineConfig,
    TradingPipeline,
    default_watchlist_path,
    load_watchlist,
)
from wc_bot.polymarket import OrderBook, PolymarketClient  # noqa: E402


def build_mock_client(watchlist, spread: float = 0.02) -> PolymarketClient:
    """Build a PolymarketClient backed by synthetic books from 'mock_price'."""
    books = {}
    for wm in watchlist:
        for spec in wm.tokens.values():
            price = spec.get("mock_price")
            if price is None:
                continue
            price = float(price)
            books[spec["token_id"]] = OrderBook(
                token_id=spec["token_id"],
                best_bid=max(0.0, price - spread / 2),
                best_ask=min(1.0, price + spread / 2),
                bid_size=1000.0,
                ask_size=1000.0,
            )
    return PolymarketClient(mock=True, mock_books=books)


def _verify_live_tokens(client: PolymarketClient, watchlist) -> None:
    """Read-only startup check: confirm each active token is live on Gamma."""
    for wm in watchlist:
        if not wm.is_active:
            continue
        for outcome_label, spec in wm.tokens.items():
            token_id = spec["token_id"]
            try:
                meta = client.get_market_metadata(token_id)
            except Exception as exc:  # noqa: BLE001 - never block the run on metadata
                print(f"  [warn] {wm.label}/{outcome_label}: metadata error: {exc}")
                continue
            if meta is None:
                print(f"  [warn] {wm.label}/{outcome_label}: token unknown to Gamma.")
            elif not meta.is_tradable:
                print(
                    f"  [warn] {wm.label}/{outcome_label}: not tradable "
                    f"(active={meta.active} closed={meta.closed} "
                    f"accepting_orders={meta.accepting_orders})."
                )
            else:
                print(f"  [ok]   {wm.label}/{outcome_label}: live ({meta.slug}).")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--watchlist",
        default=None,
        help="watchlist JSON; defaults to watchlist.live.json with --live-odds, "
        "else watchlist.mock.json",
    )
    parser.add_argument("--ledger", default=str(ROOT / "data" / "paper_trades.csv"))
    parser.add_argument("--bankroll", type=float, default=1000.0)
    parser.add_argument("--target-edge", type=float, default=0.04)
    parser.add_argument("--kelly", type=float, default=0.25, help="fractional Kelly")
    parser.add_argument(
        "--model",
        choices=[MODEL_ELO, MODEL_DIXON_COLES],
        default=MODEL_ELO,
        help="L4 model to use",
    )
    parser.add_argument(
        "--as-of",
        default=None,
        help="ISO date; point-in-time training cutoff (Dixon-Coles enforces date < as_of)",
    )
    parser.add_argument(
        "--sharp",
        action="store_true",
        help="enable the on-chain SharpTracker sizing modifier (mock/stub for now)",
    )
    parser.add_argument(
        "--use-rag",
        action="store_true",
        help="enable the L2 RAG sentiment fusion (Dixon-Coles only); with "
        "--live-odds it makes live news API + LLM calls, else deterministic mock",
    )
    parser.add_argument(
        "--rag-lookback-days",
        type=int,
        default=7,
        help="point-in-time news window per team for --use-rag",
    )
    parser.add_argument(
        "--rag-provider",
        choices=["gemini", "openai", "anthropic"],
        default="gemini",
        help="LLM provider for live sentiment (key read from its env var, e.g. "
        "GEMINI_API_KEY); only used with --use-rag --live-odds",
    )
    parser.add_argument(
        "--rag-model",
        default=None,
        help="override the LLM model name (default depends on --rag-provider)",
    )
    parser.add_argument("--loop", action="store_true", help="poll forever")
    parser.add_argument("--interval", type=float, default=60.0, help="poll seconds")
    parser.add_argument(
        "--live-odds",
        action="store_true",
        help="read live Polymarket CLOB midpoints (read-only); default is offline mock",
    )
    parser.add_argument(
        "--max-spread",
        type=float,
        default=0.05,
        help="skip a token whose bid/ask spread exceeds this (thin-liquidity guard)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="(default) offline demo using mock_price fields; ignored if --live-odds",
    )
    args = parser.parse_args()

    # Load .env (NEWSAPI_KEY, GEMINI_API_KEY, ...) so live RAG can authenticate.
    _load_dotenv(ROOT / ".env")

    # The watchlist defaults to the file matching the run mode (live vs mock).
    watchlist_path = args.watchlist or default_watchlist_path(live=args.live_odds)
    watchlist = load_watchlist(watchlist_path)
    print(f"Watchlist: {watchlist_path}")

    config = PipelineConfig(
        ledger_path=args.ledger,
        bankroll=args.bankroll,
        poll_seconds=args.interval,
        model_kind=args.model,
        as_of=args.as_of,
        use_sharp_tracker=args.sharp,
        sharp_live=args.sharp and args.live_odds,  # real subgraph only when live
        use_rag=args.use_rag,
        rag_live=args.use_rag and args.live_odds,  # real news+LLM only when live
        rag_lookback_days=args.rag_lookback_days,
        rag_provider=args.rag_provider,
        rag_model=args.rag_model,
        decision=DecisionConfig(
            target_edge=args.target_edge, kelly_fraction=args.kelly
        ),
    )

    if args.live_odds:
        # Read-only live client: CLOB midpoints with the thin-liquidity spread guard.
        client = PolymarketClient(max_spread=args.max_spread)
        print(
            f"LIVE read-only mode: Polymarket CLOB midpoints "
            f"(max spread {args.max_spread:.2f}). No orders are ever signed."
        )
        _verify_live_tokens(client, watchlist)
    else:
        client = build_mock_client(watchlist)
        print("OFFLINE mock mode: pricing off watchlist 'mock_price' fields.")

    if args.use_rag:
        if args.model != MODEL_DIXON_COLES:
            print(
                f"  [note] --use-rag adjusts Dixon-Coles attack/defense rates; "
                f"model '{args.model}' ignores sentiment."
            )
        if config.rag_live:
            mode = f"LIVE news API + {args.rag_provider} LLM"
        else:
            mode = "deterministic mock"
        print(f"RAG sentiment fusion: ON ({mode}, lookback {args.rag_lookback_days}d).")

    pipeline = TradingPipeline(watchlist, config=config, client=client)

    print(f"Fitting L4 model '{args.model}' on international history...")
    pipeline.fit_model()
    print(f"Model version: {pipeline.model_version}")
    print("Top 10 sides per the fitted model:")
    print(pipeline.leaderboard(10).to_string(index=False))

    if args.loop:
        pipeline.run_forever()
    else:
        signals = pipeline.run_once()
        bets = [s for s in signals if s.is_bet]
        print(f"\nDone. Flagged {len(bets)} paper bet(s) -> {args.ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
