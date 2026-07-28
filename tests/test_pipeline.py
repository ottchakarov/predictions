import csv
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wc_bot.pipeline import (  # noqa: E402
    MODEL_DIXON_COLES,
    MODEL_ELO,
    PipelineConfig,
    TradingPipeline,
    WatchedMarket,
)
from wc_bot.polymarket import OrderBook, PolymarketClient  # noqa: E402
from wc_bot.ingest import WalletPosition, WalletStat  # noqa: E402


def _watchlist():
    # Brazil strongly favoured vs USA; both teams exist in the dataset.
    return [
        WatchedMarket(
            match_id="BRA-USA",
            home_team="Brazil",
            away_team="United States",
            neutral=True,
            tokens={
                "Yes": {"token_id": "T_YES", "model": "home"},
                "No": {"token_id": "T_NO", "model": "draw+away"},
            },
            market_slug="will-brazil-win",
            market_type="moneyline",
            match_time="2026-06-25T18:00:00+00:00",
        )
    ]


def _client():
    # YES (Brazil) badly underpriced at ~0.41 vs model ~0.68 -> clear +EV bet.
    books = {
        "T_YES": OrderBook("T_YES", best_bid=0.40, best_ask=0.42, bid_size=1e3, ask_size=1e3),
        "T_NO": OrderBook("T_NO", best_bid=0.58, best_ask=0.60, bid_size=1e3, ask_size=1e3),
    }
    return PolymarketClient(mock=True, mock_books=books)


@pytest.mark.parametrize(
    "model_kind, expected_version",
    [(MODEL_ELO, "elo-1.0"), (MODEL_DIXON_COLES, "dixon_coles-1.0")],
)
def test_end_to_end_loop_for_each_model(tmp_path, model_kind, expected_version):
    ledger_path = tmp_path / "trades.csv"
    config = PipelineConfig(
        ledger_path=str(ledger_path),
        model_kind=model_kind,
        as_of="2026-06-01",  # strict point-in-time cutoff (enforced for Dixon-Coles)
    )
    pipeline = TradingPipeline(_watchlist(), config=config, client=_client())

    signals = pipeline.run_once(verbose=False)

    assert pipeline.model_version == expected_version
    bets = [s for s in signals if s.is_bet]
    assert bets, f"{model_kind} produced no +EV bet through the full loop"

    # The flagged bet must be persisted with the correct model_version.
    rows = list(csv.DictReader(ledger_path.open()))
    assert rows, "ledger empty"
    assert all(r["model_version"] == expected_version for r in rows)
    yes = next(r for r in rows if r["outcome"] == "Yes")
    assert float(yes["edge"]) > 0  # de-vig + edge routing worked


def test_wide_spread_token_is_skipped(tmp_path):
    # YES has a thin 10c book (> 5c guard) -> no midpoint -> skipped, no bet logged.
    books = {
        "T_YES": OrderBook("T_YES", best_bid=0.36, best_ask=0.46, bid_size=1e3, ask_size=1e3),
        "T_NO": OrderBook("T_NO", best_bid=0.58, best_ask=0.60, bid_size=1e3, ask_size=1e3),
    }
    client = PolymarketClient(mock=True, mock_books=books, max_spread=0.05)
    config = PipelineConfig(
        ledger_path=str(tmp_path / "trades.csv"), model_kind=MODEL_ELO, as_of="2026-06-01"
    )
    pipeline = TradingPipeline(_watchlist(), config=config, client=client)

    signals = pipeline.run_once(verbose=False)

    # The Yes outcome (wide spread) must never reach the decision engine.
    assert all(s.outcome != "Yes" for s in signals)


def test_incomplete_three_way_market_is_skipped(tmp_path):
    # A 1X2 market whose Draw token has no live quote: de-vigging only Home+Away
    # would renormalize a partial book into a fake 100% edge. The whole market
    # must be skipped -> no signals, no ledger rows.
    wm = WatchedMarket(
        match_id="BRA-USA-3W",
        home_team="Brazil",
        away_team="United States",
        neutral=True,
        tokens={
            "Home": {"token_id": "T_H", "model": "home"},
            "Draw": {"token_id": "T_D", "model": "draw"},
            "Away": {"token_id": "T_A", "model": "away"},
        },
        market_slug="bra-usa-3way",
        market_type="1x2",
        match_time="2026-06-25T18:00:00+00:00",
    )
    # Home is badly underpriced (would be a strong +EV bet if the book completed),
    # but the Draw token is absent -> incomplete book.
    books = {
        "T_H": OrderBook("T_H", best_bid=0.40, best_ask=0.42, bid_size=1e3, ask_size=1e3),
        "T_A": OrderBook("T_A", best_bid=0.18, best_ask=0.20, bid_size=1e3, ask_size=1e3),
    }
    client = PolymarketClient(mock=True, mock_books=books)
    ledger_path = tmp_path / "trades.csv"
    config = PipelineConfig(
        ledger_path=str(ledger_path), model_kind=MODEL_ELO, as_of="2026-06-01"
    )
    pipeline = TradingPipeline([wm], config=config, client=client)

    signals = pipeline.run_once(verbose=False)

    assert signals == []  # nothing priced off a partial book
    rows = list(csv.DictReader(ledger_path.open())) if ledger_path.exists() else []
    assert rows == []  # no manufactured edge written


def test_concluded_market_empty_book_is_skipped_gracefully(tmp_path, capsys):
    # Croatia vs Ghana concluded -> CLOB 404/empty for every token. The loop must
    # skip with a "Market Closed or Empty" log, place no trades, and not crash.
    wm = WatchedMarket(
        match_id="HRV-GHA",
        home_team="Croatia",
        away_team="Ghana",
        neutral=True,
        tokens={
            "Yes": {"token_id": "T_YES", "model": "home"},
            "No": {"token_id": "T_NO", "model": "draw+away"},
        },
        market_slug="hrv-gha",
        market_type="moneyline",
        match_time="2026-06-27T12:00:00+00:00",
    )

    class _ResolvedClient(PolymarketClient):
        def get_order_book(self, token_id):  # simulate a concluded market's 404
            raise RuntimeError("404 Not Found for /book")

    client = _ResolvedClient()
    ledger_path = tmp_path / "trades.csv"
    config = PipelineConfig(
        ledger_path=str(ledger_path), model_kind=MODEL_ELO, as_of="2026-06-01"
    )
    pipeline = TradingPipeline([wm], config=config, client=client)

    signals = pipeline.run_once(verbose=True)  # must not raise

    assert signals == []
    out = capsys.readouterr().out
    assert "Market Closed or Empty" in out
    assert "[error]" not in out  # skipped gracefully, not caught as a crash
    rows = list(csv.DictReader(ledger_path.open())) if ledger_path.exists() else []
    assert rows == []


def test_unknown_model_kind_raises(tmp_path):
    config = PipelineConfig(ledger_path=str(tmp_path / "t.csv"), model_kind="bogus")
    pipeline = TradingPipeline(_watchlist(), config=config, client=_client())
    with pytest.raises(ValueError, match="Unknown model_kind"):
        pipeline.fit_model()


class _ContradictingTracker:
    """Sharp tracker whose capital sits overwhelmingly on the opposing side."""

    def get_top_wallets(self, market_filter="Sports", min_roi=0.10, *, as_of=None):
        return [WalletStat("0xsharp", 0.6, 5000.0, 30, as_of)]

    def get_recent_positions(self, wallet_addresses, token_id, *, as_of=None):
        # Net short the target token => >60% contradicting => veto.
        return {"0xsharp": WalletPosition("0xsharp", token_id, -900.0, 0.5, as_of)}


def test_sharp_veto_is_logged_as_terminal_row(tmp_path):
    ledger_path = tmp_path / "trades.csv"
    config = PipelineConfig(
        ledger_path=str(ledger_path), model_kind=MODEL_ELO, as_of="2026-06-01"
    )
    pipeline = TradingPipeline(_watchlist(), config=config, client=_client())
    pipeline.sharp = _ContradictingTracker()  # inject contradicting sharp money

    signals = pipeline.run_once(verbose=False)

    yes = next(s for s in signals if s.outcome == "Yes")
    assert not yes.is_bet and yes.stake == 0.0 and yes.sharp_multiplier == 0.0

    rows = list(csv.DictReader(ledger_path.open()))
    veto = next(r for r in rows if r["outcome"] == "Yes")
    assert veto["status"] == "VETOED"
    assert float(veto["stake"]) == 0.0
    assert float(veto["pnl"]) == 0.0
    # The foregone edge must remain auditable.
    assert float(veto["model_prob"]) > 0 and float(veto["market_price"]) > 0
    assert float(veto["edge"]) > 0


def test_rag_sentiment_flows_into_dixon_coles(tmp_path):
    # Baseline (RAG off) home probability for the Brazil "Yes" token.
    base_cfg = PipelineConfig(
        ledger_path=str(tmp_path / "b.csv"), model_kind=MODEL_DIXON_COLES, as_of="2026-06-01"
    )
    base = TradingPipeline(_watchlist(), config=base_cfg, client=_client())
    base_yes = next(s for s in base.run_once(verbose=False) if s.outcome == "Yes")

    # RAG on: crush the home team's sentiment and boost the away team. The fused
    # attack/defense adjustment must lower Brazil's modelled win probability.
    rag_cfg = PipelineConfig(
        ledger_path=str(tmp_path / "r.csv"),
        model_kind=MODEL_DIXON_COLES,
        as_of="2026-06-01",
        use_rag=True,
    )
    rag = TradingPipeline(_watchlist(), config=rag_cfg, client=_client())
    assert rag.news_fetcher is not None and rag.sentiment_agent is not None

    calls = []

    def fake_team_sentiment(team, as_of, fetcher, lookback_days=7):
        calls.append((team, as_of, lookback_days))
        return -1.0 if team == "Brazil" else 1.0

    rag.sentiment_agent.team_sentiment = fake_team_sentiment
    rag_yes = next(s for s in rag.run_once(verbose=False) if s.outcome == "Yes")

    # Sentiment was fetched point-in-time for both sides, with the configured window.
    teams_called = {c[0] for c in calls}
    assert {"Brazil", "United States"} <= teams_called
    assert all(c[2] == 7 for c in calls)
    # The negative-home / positive-away signal pushed Brazil's prob down.
    assert rag_yes.model_prob < base_yes.model_prob


def test_rag_is_ignored_for_elo(tmp_path):
    # Elo.match_probabilities takes no sentiment kwargs; enabling RAG must not
    # crash and must not attempt to score sentiment for a non-DC model.
    config = PipelineConfig(
        ledger_path=str(tmp_path / "t.csv"),
        model_kind=MODEL_ELO,
        as_of="2026-06-01",
        use_rag=True,
    )
    pipeline = TradingPipeline(_watchlist(), config=config, client=_client())
    pipeline.sentiment_agent.team_sentiment = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("Elo must not invoke sentiment scoring")
    )
    signals = pipeline.run_once(verbose=False)
    assert any(s.outcome == "Yes" for s in signals)


def test_run_once_obeys_as_of_override_on_first_fit(tmp_path):
    # The very first fit happens lazily inside run_once. A backtest replay passes
    # as_of_override; it must take precedence over config.as_of on that first fit.
    config = PipelineConfig(
        ledger_path=str(tmp_path / "t.csv"),
        model_kind=MODEL_DIXON_COLES,
        as_of="2026-06-01",  # would be used if the override were ignored
    )
    pipeline = TradingPipeline(_watchlist(), config=config, client=_client())
    assert not pipeline._fitted  # nothing fit yet

    override = pd.Timestamp("2018-01-01")
    pipeline.run_once(verbose=False, as_of_override=override)

    # The override (not config.as_of) drove the point-in-time training cutoff.
    assert pipeline.model.reference_date_ == override


def test_dixon_coles_defaults_as_of_to_now_for_live(tmp_path):
    # No explicit as_of: live polling must still train strictly on date < now.
    config = PipelineConfig(
        ledger_path=str(tmp_path / "t.csv"), model_kind=MODEL_DIXON_COLES, as_of=None
    )
    pipeline = TradingPipeline(_watchlist(), config=config, client=_client())
    pipeline.fit_model()
    assert pipeline.model_version == "dixon_coles-1.0"
    # reference date should be ~now (within the training window), not None.
    assert pipeline.model.reference_date_ is not None
    assert pipeline.model.knows_team("Brazil")
