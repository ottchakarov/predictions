"""The chassis: wire L1 -> L4 -> L5 -> L6 -> L7 into one autonomous loop.

    load history (L1) -> fit Elo (L4) -> for each watched match:
        pull Polymarket book (L5) -> compute edge & size (L6) -> log paper bet (L7)

Run it once (``run_once``) or let it poll forever (``run_forever``). The exact
same control flow becomes the live bot once L7 swaps the CSV write for a signed
CLOB order.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .decision import DecisionConfig, TradeSignal, evaluate_market
from .elo import Elo, EloConfig
from .features import FeatureStore
from .ingest import SharpTracker, load_matches
from .ledger import STATUS_INCOMPLETE, STATUS_OPEN, STATUS_VETOED, PaperLedger
from .models import DixonColesConfig, DixonColesModel
from .polymarket import PolymarketClient
from .rag import NewsFetcher, SentimentAgent

_BASE_OUTCOMES = {"home", "draw", "away"}

# Default watchlist files, selected by run mode (live odds vs offline mock).
_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
MOCK_WATCHLIST = str(_CONFIG_DIR / "watchlist.mock.json")
LIVE_WATCHLIST = str(_CONFIG_DIR / "watchlist.live.json")


def default_watchlist_path(*, live: bool) -> str:
    """Return the watchlist that matches the run mode.

    Live odds default to the real-markets file; offline runs to the mock file.
    """
    return LIVE_WATCHLIST if live else MOCK_WATCHLIST

# Model registry: the config switch toggles between these L4 engines. Every model
# obeys the same interface (fit / match_probabilities / knows_team / model_version).
MODEL_ELO = "elo"
MODEL_DIXON_COLES = "dixon_coles"
_DIXON_COLES_ALIASES = {MODEL_DIXON_COLES, "dixoncoles", "dc"}


@dataclass
class WatchedMarket:
    """One match we track, linking Elo team names to Polymarket outcome tokens.

    ``tokens`` maps a Polymarket outcome label -> {"token_id", "model"}, where
    ``model`` selects the Elo probability for that label. ``model`` may be a base
    outcome ('home'/'draw'/'away') or a '+'-joined combination, e.g. 'draw+away'
    for a "Will the home team fail to win?" NO token.
    """

    match_id: str
    home_team: str
    away_team: str
    neutral: bool
    tokens: Dict[str, Dict[str, str]]
    market_slug: str = ""
    market_type: str = "moneyline"
    match_time: str = ""        # ISO-8601 kickoff; used by settlement & skipping
    is_active: bool = True      # set False once concluded/settled to skip polling
    condition_id: str = ""      # on-chain Polymarket conditionId (resolution key)

    @property
    def label(self) -> str:
        return f"{self.home_team} vs {self.away_team}"


@dataclass
class PipelineConfig:
    data_path: Optional[str] = None
    ledger_path: str = "data/paper_trades.csv"
    bankroll: float = 1000.0
    poll_seconds: float = 60.0
    # L4 model selection: MODEL_ELO or MODEL_DIXON_COLES.
    model_kind: str = MODEL_ELO
    # Point-in-time training cutoff (ISO date). Strictly enforced for Dixon-Coles
    # (FeatureStore keeps only fixtures with date < as_of) and used as the
    # time-decay reference. None => use all available history.
    as_of: Optional[str] = None
    # Enable the L6 on-chain SharpTracker as a sizing modifier. Off by default;
    # when on, every trade logs its sharp multiplier. ``sharp_live`` toggles real
    # Goldsky subgraph queries vs the empty mock tracker.
    use_sharp_tracker: bool = False
    sharp_live: bool = False
    # L2 RAG sentiment fusion (Dixon-Coles only). ``rag_live`` toggles real news
    # API + LLM calls vs the deterministic mock; ``rag_lookback_days`` bounds the
    # point-in-time news window per match.
    use_rag: bool = False
    rag_live: bool = False
    rag_lookback_days: int = 7
    # LLM provider for live sentiment ("gemini" | "openai" | "anthropic"); the API
    # key is read from the provider's env var (GEMINI_API_KEY, etc.). rag_model=None
    # uses that provider's default model.
    rag_provider: str = "gemini"
    rag_model: Optional[str] = None
    elo: EloConfig = field(default_factory=EloConfig)
    dixon_coles: DixonColesConfig = field(default_factory=DixonColesConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)


class TradingPipeline:
    def __init__(
        self,
        watchlist: List[WatchedMarket],
        *,
        config: Optional[PipelineConfig] = None,
        client: Optional[PolymarketClient] = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.watchlist = watchlist
        self.client = client or PolymarketClient()
        # L6 sharp-money sizing modifier. ``sharp_live`` hits the Goldsky subgraph;
        # otherwise an empty mock tracker (-> neutral 1.0 multiplier).
        self.sharp = (
            SharpTracker(mock=not self.config.sharp_live)
            if self.config.use_sharp_tracker
            else None
        )
        # L2 RAG sentiment collaborators (Dixon-Coles only). ``rag_live`` hits the
        # live news API + LLM; otherwise the deterministic mock.
        if self.config.use_rag:
            self.news_fetcher = NewsFetcher(mock=not self.config.rag_live)
            self.sentiment_agent = SentimentAgent(
                mock=not self.config.rag_live,
                provider=self.config.rag_provider,
                model=self.config.rag_model,
            )
        else:
            self.news_fetcher = None
            self.sentiment_agent = None
        # Per-cycle sentiment cache (news is fetched up to now(); cleared each cycle).
        self._sentiment_cache: Dict[str, float] = {}
        # Decision-time cutoff for point-in-time sharp lookups; set each cycle.
        self._cycle_as_of: Optional[pd.Timestamp] = None
        # The concrete L4 model is built lazily in fit_model() from config.
        self.model = None
        self.model_version: str = ""
        self.ledger = PaperLedger(self.config.ledger_path)
        self._fitted = False

    # ------------------------------------------------------------------ L1/L4
    def fit_model(
        self, *, as_of_override: Optional[pd.Timestamp] = None
    ) -> "TradingPipeline":
        """Build and fit the configured L4 model (Elo or Dixon-Coles).

        ``as_of_override`` (used by the live loop) forces the point-in-time cutoff
        for this fit, taking precedence over ``config.as_of`` so a long-running
        process advances its training window each cycle.
        """
        kind = self.config.model_kind.strip().lower()
        if kind == MODEL_ELO:
            self.model = self._fit_elo()
        elif kind in _DIXON_COLES_ALIASES:
            self.model = self._fit_dixon_coles(as_of_override=as_of_override)
        else:
            raise ValueError(
                f"Unknown model_kind {self.config.model_kind!r}; "
                f"expected {MODEL_ELO!r} or {MODEL_DIXON_COLES!r}."
            )

        self.model_version = self.model.model_version
        self._fitted = True

        for wm in self.watchlist:
            for team in (wm.home_team, wm.away_team):
                if not self.model.knows_team(team):
                    print(
                        f"  [warn] '{team}' unknown to {self.model_version} — check "
                        f"the spelling against the dataset's team names."
                    )
        return self

    def _fit_elo(self) -> Elo:
        # L1 ingestion feeds Elo directly (ratings are recursive, not as_of-bound).
        matches = load_matches(self.config.data_path)
        return Elo(config=self.config.elo).fit(matches)

    def _fit_dixon_coles(
        self, *, as_of_override: Optional[pd.Timestamp] = None
    ) -> DixonColesModel:
        # L3 FeatureStore enforces strict point-in-time retrieval (date < as_of)
        # before the L4 model ever sees a row — no look-ahead leakage.
        store = FeatureStore(self.config.data_path)
        as_of, origin = self._resolve_as_of(as_of_override)
        train = store.training_frame(as_of)
        print(
            f"  FeatureStore: {len(train):,} fixtures (date < {as_of.date()}, "
            f"{origin}) -> Dixon-Coles."
        )
        return DixonColesModel(config=self.config.dixon_coles).fit(
            train, reference_date=as_of
        )

    def model_uses_as_of(self) -> bool:
        """True for models whose training is point-in-time bound (Dixon-Coles)."""
        return self.config.model_kind.strip().lower() in _DIXON_COLES_ALIASES

    @staticmethod
    def _utc_now() -> pd.Timestamp:
        """Current UTC instant, tz-naive to match the dataset's date column."""
        return pd.Timestamp.now(tz="UTC").tz_localize(None)

    def _resolve_as_of(
        self, as_of_override: Optional[pd.Timestamp] = None
    ) -> tuple[pd.Timestamp, str]:
        """Resolve the point-in-time cutoff.

        Precedence: an explicit per-cycle ``as_of_override`` (live loop) > a static
        ``config.as_of`` (backtest/replay) > the current timestamp. Either way the
        result keeps training strictly on ``date < as_of`` — never the future.
        """
        if as_of_override is not None:
            return pd.Timestamp(as_of_override), "live now()"
        if self.config.as_of:
            return pd.Timestamp(self.config.as_of), "config"
        return self._utc_now(), "live now()"

    def leaderboard(self, n: int = 10) -> pd.DataFrame:
        """Top teams per the fitted model (ratings for Elo, attack for DC)."""
        if not self._fitted:
            self.fit_model()
        if isinstance(self.model, Elo):
            return self.model.top(n)
        if isinstance(self.model, DixonColesModel):
            return self.model.team_strengths().head(n)
        return pd.DataFrame()

    # ------------------------------------------------------------- one cycle
    def run_once(
        self, *, verbose: bool = True, as_of_override: Optional[pd.Timestamp] = None
    ) -> List[TradeSignal]:
        if not self._fitted:
            # Reproducibility: the FIRST fit must honour the caller's per-cycle
            # cutoff too, not silently fall back to config.as_of (point-in-time).
            self.fit_model(as_of_override=as_of_override)

        # Decision-time cutoff for point-in-time sharp lookups: the live instant
        # (or override), else a static backtest as_of, else now.
        self._cycle_as_of, _ = self._resolve_as_of(as_of_override)
        # News is fetched up to the cycle cutoff; reset the cache each cycle so a
        # long-running loop re-reads fresh sentiment as time advances.
        self._sentiment_cache = {}

        all_signals: List[TradeSignal] = []
        for wm in self.watchlist:
            if not wm.is_active:
                if verbose:
                    print(f"  [skip] {wm.label}: inactive/concluded — not polling.")
                continue
            try:
                signals = self._process_match(wm, verbose=verbose)
            except Exception as exc:  # noqa: BLE001 - keep the loop alive
                print(f"  [error] {wm.label}: {exc}")
                continue
            all_signals.extend(signals)
        return all_signals

    def run_forever(self, *, verbose: bool = True) -> None:
        print(
            f"Starting autonomous loop (poll every {self.config.poll_seconds}s). "
            f"Ctrl-C to stop."
        )
        refit_each_cycle = self.model_uses_as_of()
        try:
            while True:
                if refit_each_cycle:
                    # Point-in-time correctness for long-running processes: compute
                    # the cutoff fresh every iteration and refit on date < now, so
                    # the training window advances and never relies on a stale
                    # config default.
                    now = self._utc_now()
                    if verbose:
                        print(f"\n[cycle] refitting {self.config.model_kind} as_of={now.isoformat()}")
                    self.fit_model(as_of_override=now)
                    self.run_once(verbose=verbose, as_of_override=now)
                else:
                    self.run_once(verbose=verbose)
                time.sleep(self.config.poll_seconds)
        except KeyboardInterrupt:
            print("\nStopped.")

    # --------------------------------------------------------------- internals
    def _team_sentiment(self, team: str) -> float:
        """Point-in-time squad-state score for ``team`` (cached per cycle).

        News is fetched up to the cycle cutoff (now() for live polling) and scored.
        Any failure degrades to 0.0 (neutral) so the loop never breaks on a flaky
        news API or LLM.
        """
        if team in self._sentiment_cache:
            return self._sentiment_cache[team]
        as_of = (self._cycle_as_of or self._utc_now()).to_pydatetime()
        try:
            score = self.sentiment_agent.team_sentiment(
                team,
                as_of,
                self.news_fetcher,
                lookback_days=self.config.rag_lookback_days,
            )
        except Exception as exc:  # noqa: BLE001 - safe degrade to neutral
            print(f"  [warn] RAG sentiment for '{team}' failed: {exc}")
            score = 0.0
        self._sentiment_cache[team] = score
        return score

    def _process_match(
        self, wm: WatchedMarket, *, verbose: bool
    ) -> List[TradeSignal]:
        # L2 RAG: only Dixon-Coles consumes sentiment (it adjusts the log-linear
        # attack/defense rates). Computed before pricing so the edge reflects it.
        sentiment_kwargs: Dict[str, float] = {}
        if self.config.use_rag and isinstance(self.model, DixonColesModel):
            home_s = self._team_sentiment(wm.home_team)
            away_s = self._team_sentiment(wm.away_team)
            sentiment_kwargs = {"home_sentiment": home_s, "away_sentiment": away_s}
            if verbose and (home_s or away_s):
                print(
                    f"  [rag] {wm.label}: sentiment home={home_s:+.2f} away={away_s:+.2f}"
                )

        # Model-agnostic: Elo and Dixon-Coles both return {home, draw, away}.
        # For Dixon-Coles this dict is the collapsed matrix_to_1x2() output, which
        # flows unchanged into the de-vig + Kelly logic below.
        base_probs = self.model.match_probabilities(
            wm.home_team, wm.away_team, neutral=wm.neutral, **sentiment_kwargs
        )

        market_prices: Dict[str, float] = {}
        model_probs: Dict[str, float] = {}
        token_by_outcome: Dict[str, str] = {}

        for outcome_label, spec in wm.tokens.items():
            token_id = spec["token_id"]
            # L5: spread-guarded top-of-book midpoint. A wide/one-sided book yields
            # midpoint=None so we skip it rather than price an edge off a bad quote.
            price = self.client.get_live_price(token_id)
            if price.midpoint is None:
                if verbose:
                    # An empty/one-sided/unavailable book == a concluded or not-yet-
                    # open market; a present-but-untrustworthy quote (crossed/wide
                    # spread) is a liquidity guard. Label them distinctly.
                    book_empty = price.best_bid is None or price.best_ask is None
                    label = "Market Closed or Empty" if book_empty else "thin/unreliable quote"
                    print(
                        f"  [skip] {wm.label} / {outcome_label} ({token_id[:12]}…): "
                        f"{label} — {price.reason or 'no midpoint'}"
                    )
                continue
            market_prices[outcome_label] = price.midpoint
            model_probs[outcome_label] = _resolve_model_prob(spec["model"], base_probs)
            token_by_outcome[outcome_label] = token_id

        # ANTI-LEAKAGE (execution.mdc): a market is priced ALL-or-NOTHING. If any
        # required outcome lacks a valid quote, de-vigging the survivors would
        # renormalize a partial book to 100% and manufacture a fake edge. So we
        # flag the whole market STATUS_INCOMPLETE and skip it — no trades.
        required = list(wm.tokens.keys())
        missing = [o for o in required if o not in market_prices]
        if missing:
            if verbose:
                print(
                    f"  [skip] {wm.label}: {STATUS_INCOMPLETE} — no valid quote for "
                    f"{missing} of {required}; market not priced "
                    f"(no partial-book renormalization)."
                )
            return []

        # L6: pass the sharp tracker, the per-cycle as_of, and outcome->token map
        # so stakes are scaled (and vetoed) by point-in-time sharp-money signals.
        # ``required_outcomes`` makes the all-or-nothing guard explicit in L6 too.
        as_of = self._cycle_as_of.to_pydatetime() if self._cycle_as_of is not None else None
        signals = evaluate_market(
            model_probs,
            market_prices,
            bankroll=self.config.bankroll,
            config=self.config.decision,
            sharp=self.sharp,
            as_of=as_of,
            token_ids=token_by_outcome,
            required_outcomes=required,
        )

        if verbose:
            probs_str = {k: round(v, 3) for k, v in base_probs.items()}
            print(f"\n  {wm.label}  ({self.model_version}: {probs_str})")
        for s in signals:
            # A sharp veto zeroes the stake (is_bet=False, multiplier==0). We still
            # log it as a terminal VETOED row so the foregone edge is auditable.
            vetoed = (not s.is_bet) and s.sharp_multiplier == 0.0
            if verbose:
                flag = "  *** BET ***" if s.is_bet else ("  *** VETOED ***" if vetoed else "")
                if s.sharp_multiplier != 1.0:
                    flag += f" [sharp x{s.sharp_multiplier:.2f}]"
                print(
                    f"    {s.outcome:<14} model={s.model_prob:.3f} "
                    f"mkt={s.market_price:.3f} fair={s.fair_market_prob:.3f} "
                    f"edge={s.edge:+.3f} stake=${s.stake:.2f}{flag}"
                )
            if s.is_bet or vetoed:
                token_id = token_by_outcome.get(s.outcome, "")
                written = self.ledger.record(
                    s,
                    match_id=wm.match_id,
                    market_slug=wm.market_slug,
                    match=wm.label,
                    bankroll=self.config.bankroll,
                    token_id=token_id,
                    market_type=wm.market_type,
                    match_time=wm.match_time,
                    model_version=self.model_version,
                    status=STATUS_VETOED if vetoed else STATUS_OPEN,
                    pnl=0.0 if vetoed else None,
                )
                kind = "VETO" if vetoed else "paper trade"
                if verbose:
                    if written:
                        print(f"      -> logged {kind} ({wm.match_id}/{s.outcome}).")
                    else:
                        print(
                            f"      -> de-dup: existing row for "
                            f"{wm.match_id}/{s.outcome}, skipped."
                        )
        return signals


def _resolve_model_prob(spec: str, elo_probs: Dict[str, float]) -> float:
    """Resolve a 'model' spec like 'home' or 'draw+away' into a probability."""
    parts = [p.strip() for p in spec.split("+") if p.strip()]
    unknown = [p for p in parts if p not in _BASE_OUTCOMES]
    if unknown:
        raise ValueError(
            f"Unknown model outcome(s) {unknown}; valid base outcomes are "
            f"{sorted(_BASE_OUTCOMES)} (combine with '+')."
        )
    return sum(elo_probs[p] for p in parts)


def load_watchlist(path: str | Path) -> List[WatchedMarket]:
    data = json.loads(Path(path).read_text())
    return [
        WatchedMarket(
            match_id=m["match_id"],
            home_team=m["home_team"],
            away_team=m["away_team"],
            neutral=bool(m.get("neutral", True)),
            tokens=m["tokens"],
            market_slug=m.get("market_slug", ""),
            market_type=m.get("market_type", "moneyline"),
            match_time=m.get("match_time", ""),
            is_active=bool(m.get("is_active", True)),
            condition_id=m.get("condition_id", ""),
        )
        for m in data["matches"]
    ]
