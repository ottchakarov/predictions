"""L2 - News retrieval + LLM sentiment.

Two collaborators, each with a deterministic ``mock`` path (for offline/CI use)
and a live external path:

* ``NewsFetcher`` pulls recent headlines for a team. It is the L2 analogue of the
  L3 FeatureStore's point-in-time contract: ``get_team_news`` returns ONLY items
  published strictly before ``as_of`` (and within ``lookback_days``), so a
  backtest training/predicting "as of" a kickoff can never read news that didn't
  exist yet. No look-ahead, ever (data_hygiene.mdc). The live path targets a news
  API (NewsAPI shape by default) and passes ``as_of`` through as the API ``to``
  bound *and* re-filters client-side with a strict ``<``.
* ``SentimentAgent`` turns those headlines into a single squad-state score in
  [-1, 1]. -1.0 = crisis / key injuries / manager fired, +1.0 = perfect health &
  high morale, 0.0 = neutral / no signal. The live path calls an LLM and parses a
  structured JSON ``sentiment_score``.

EXECUTION DISCIPLINE (execution.mdc): both live paths degrade SAFELY. Any network
timeout, rate-limit (HTTP 429), or LLM/JSON parse failure returns the neutral
default (``[]`` / ``0.0``) so the trading loop never crashes and simply falls back
to the unmodified Dixon-Coles rates.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)

NEWSAPI_URL = "https://newsapi.org/v2/everything"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models"

# Default model + env var holding the API key, per provider.
_PROVIDER_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-latest",
    "gemini": "gemini-2.0-flash",
    "google": "gemini-2.0-flash",
}
_PROVIDER_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "google": "GEMINI_API_KEY",
}
DEFAULT_LLM_MODEL = _PROVIDER_MODELS["openai"]


def _default_model(provider: str) -> str:
    return _PROVIDER_MODELS.get(provider, DEFAULT_LLM_MODEL)


def _default_api_key(provider: str) -> Optional[str]:
    env = _PROVIDER_KEY_ENV.get(provider)
    return os.getenv(env) if env else None

# Keyword lexicons for the mock "LLM". The live SentimentAgent replaces these with
# a model call; the contract (a float in [-1, 1]) is unchanged.
_NEGATIVE_TERMS = (
    "injury", "injured", "doubt", "doubtful", "ruled out", "sidelined", "crisis",
    "suspended", "suspension", "fitness concern", "knock", "strain", "surgery",
    "manager fired", "sacked",
)
_POSITIVE_TERMS = (
    "fit", "return", "returns", "boost", "fully fit", "in form", "morale",
    "available", "recovered", "back in training", "confident",
)

_SYSTEM_PROMPT = (
    "You are a sports quantitative analyst specialising in football (soccer) "
    "squad availability and morale. You will be given recent news headlines and "
    "snippets about a single national team ahead of a match. Read them and judge "
    "the team's CURRENT squad state. Respond with ONLY a JSON object of the form "
    '{"sentiment_score": <float>} where sentiment_score is a single number in '
    "[-1.0, 1.0]: -1.0 means a severe negative state (star players injured or "
    "suspended, manager fired, locker-room crisis), 0.0 means neutral / no "
    "material news, and +1.0 means an optimal state (full health, key players "
    "returning, high morale). Do not include any text outside the JSON object."
)


@dataclass(frozen=True)
class NewsItem:
    """One headline, with the timestamp used for point-in-time filtering."""

    team: str
    headline: str
    published: datetime
    source: str = "mock-wire"


class NewsFetcher:
    """Retrieves team headlines as-of a cutoff (mock or live news API)."""

    def __init__(
        self,
        *,
        mock: bool = True,
        api_key: Optional[str] = None,
        base_url: str = NEWSAPI_URL,
        timeout: float = 10.0,
        page_size: int = 20,
    ) -> None:
        self.mock = mock
        # Fall back to the NEWSAPI_KEY env var (loaded from .env by the runner).
        self.api_key = api_key or os.getenv("NEWSAPI_KEY")
        self.base_url = base_url
        self.timeout = timeout
        self.page_size = page_size
        self._session = None  # lazy requests.Session

    def get_team_news(
        self,
        team_name: str,
        as_of: datetime,
        lookback_days: int = 7,
    ) -> List[NewsItem]:
        """Return headlines for ``team_name`` published in [as_of - lookback, as_of).

        STRICT point-in-time: every returned item has ``published < as_of`` and
        ``published >= as_of - lookback_days``. ``as_of`` is required precisely so
        a caller cannot accidentally fetch "now" during a historical backtest.

        The live path degrades to ``[]`` on any failure (timeout, rate limit, bad
        payload) so the pipeline falls back to neutral 0.0 sentiment.
        """
        if as_of is None:
            raise ValueError("get_team_news requires an explicit as_of (no look-ahead).")
        cutoff = _as_utc(as_of)
        window_start = cutoff - timedelta(days=lookback_days)

        if self.mock:
            raw_items = _synthetic_headlines(team_name, cutoff)
        else:
            try:
                raw_items = self._fetch_live(team_name, window_start, cutoff)
            except Exception as exc:  # noqa: BLE001 - safe degrade, never crash loop
                logger.warning("NewsFetcher live fetch failed for %s: %s", team_name, exc)
                return []

        # Enforce the strict window regardless of source (the API 'to' bound is
        # inclusive and may return an item stamped exactly at the cutoff).
        return [it for it in raw_items if window_start <= it.published < cutoff]

    # ----------------------------------------------------------------- live
    def _get_session(self):
        if self._session is None:
            import requests

            self._session = requests.Session()
            self._session.headers.update({"User-Agent": "wc-bot/0.1 (read-only)"})
        return self._session

    def _fetch_live(
        self, team_name: str, window_start: datetime, cutoff: datetime
    ) -> List[NewsItem]:
        """Query the news API for [window_start, cutoff) and map to NewsItem.

        Point-in-time is enforced at the HTTP layer via the ``to`` bound (= cutoff)
        and the ``from`` bound (= window_start); the caller re-applies a strict
        ``<`` filter. Raises on any transport/HTTP error (handled by caller).
        """
        params = {
            "q": f'"{team_name}" AND (injury OR squad OR lineup OR fitness OR manager OR morale)',
            "from": _iso_minute(window_start),
            "to": _iso_minute(cutoff),       # API 'to' is the look-ahead guard
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": self.page_size,
            "apiKey": self.api_key,
        }
        resp = self._get_session().get(self.base_url, params=params, timeout=self.timeout)
        resp.raise_for_status()  # 429 rate-limit / 5xx -> HTTPError -> caller -> []
        articles = (resp.json() or {}).get("articles", []) or []
        items: List[NewsItem] = []
        for art in articles:
            published = _parse_iso(art.get("publishedAt"))
            if published is None:
                continue
            title = (art.get("title") or "").strip()
            desc = (art.get("description") or "").strip()
            headline = f"{title}. {desc}".strip(". ") if desc else title
            if not headline:
                continue
            source = ((art.get("source") or {}).get("name")) or "news-api"
            items.append(NewsItem(team=team_name, headline=headline, published=published, source=source))
        return items


class SentimentAgent:
    """Scores squad state from headlines (keyword mock, or a live LLM)."""

    def __init__(
        self,
        *,
        mock: bool = True,
        provider: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        min_call_interval: float = 3.0,
        max_retries: int = 2,
    ) -> None:
        self.mock = mock
        self.provider = provider
        # Model + key default per provider; key falls back to the provider's env
        # var (e.g. GEMINI_API_KEY), loaded from .env by the runner.
        self.model = model or _default_model(provider)
        self.api_key = api_key or _default_api_key(provider)
        self.timeout = timeout
        # Free-tier LLM APIs (e.g. Gemini ~15 req/min) 429 on back-to-back calls.
        # Enforce a minimum gap between calls and retry-with-backoff on 429.
        self.min_call_interval = min_call_interval
        self.max_retries = max_retries
        self._last_call_ts = 0.0

    def analyze_squad_state(self, news_items: Sequence[NewsItem]) -> float:
        """Return a squad-state score in [-1.0, 1.0].

        Empty input -> 0.0 (neutral safe default). The live path prompts an LLM and
        parses a structured ``sentiment_score``; ANY error (timeout, bad/non-JSON
        response, missing/0ut-of-range field) defaults to 0.0 so the Dixon-Coles
        fusion never receives a bad value and the loop never crashes.
        """
        if not news_items:
            return 0.0
        if self.mock:
            scores = [_score_text(it.headline) for it in news_items]
            return _clamp(sum(scores) / len(scores))

        try:
            raw = self._call_llm(self._build_user_prompt(news_items))
            return _clamp(self._parse_score(raw))
        except Exception as exc:  # noqa: BLE001 - safe degrade to neutral
            logger.warning("SentimentAgent LLM scoring failed: %s", exc)
            return 0.0

    def team_sentiment(
        self,
        team_name: str,
        as_of: datetime,
        fetcher: NewsFetcher,
        *,
        lookback_days: int = 7,
    ) -> float:
        """Convenience: fetch point-in-time news then score it (used by pipeline)."""
        return self.analyze_squad_state(
            fetcher.get_team_news(team_name, as_of, lookback_days=lookback_days)
        )

    # ----------------------------------------------------------------- live
    @staticmethod
    def _build_user_prompt(news_items: Sequence[NewsItem]) -> str:
        lines = [
            f"{i + 1}. [{it.published.date()}] {it.headline}"
            for i, it in enumerate(news_items)
        ]
        team = news_items[0].team if news_items else "the team"
        return (
            f"Team: {team}\nRecent news (most recent first):\n" + "\n".join(lines)
            + '\n\nReturn ONLY: {"sentiment_score": <float in [-1.0, 1.0]>}'
        )

    def _throttle(self) -> None:
        """Sleep so consecutive LLM calls are >= ``min_call_interval`` apart."""
        if self.min_call_interval <= 0:
            return
        wait = self.min_call_interval - (time.monotonic() - self._last_call_ts)
        if wait > 0:
            time.sleep(wait)
        self._last_call_ts = time.monotonic()

    def _call_llm(self, user_prompt: str) -> str:
        """Call the configured LLM and return the raw text response.

        Isolated so the test suite can patch THIS method (no network in CI). The
        default provider uses the OpenAI Python SDK with JSON-mode output.

        Throttled: we sleep so consecutive calls are at least ``min_call_interval``
        apart (the home/away lookups for a match fire back-to-back, which trips
        free-tier rate limits otherwise).
        """
        self._throttle()
        if self.provider == "openai":
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key, timeout=self.timeout)
            resp = client.chat.completions.create(
                model=self.model,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return resp.choices[0].message.content or ""
        if self.provider == "anthropic":
            import anthropic

            client = anthropic.Anthropic(api_key=self.api_key, timeout=self.timeout)
            resp = client.messages.create(
                model=self.model,
                max_tokens=64,
                temperature=0.0,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return resp.content[0].text or ""
        if self.provider in ("gemini", "google"):
            # Gemini REST (no extra SDK needed; `requests` is already a dep). JSON
            # mode via responseMimeType so we always get a parseable object.
            import requests

            url = f"{GEMINI_URL}/{self.model}:generateContent"
            body = {
                "system_instruction": {"parts": [{"text": _SYSTEM_PROMPT}]},
                "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                "generationConfig": {
                    "temperature": 0.0,
                    "responseMimeType": "application/json",
                },
            }
            for attempt in range(self.max_retries + 1):
                resp = requests.post(
                    url, params={"key": self.api_key}, json=body, timeout=self.timeout
                )
                # 429 = rate/quota, 5xx = transient overload (e.g. 503 UNAVAILABLE):
                # both are retryable. Exponential backoff, then retry.
                retryable = resp.status_code == 429 or 500 <= resp.status_code < 600
                if retryable and attempt < self.max_retries:
                    backoff = self.min_call_interval * (attempt + 1) * 2
                    logger.warning("Gemini %s; backing off %.1fs (attempt %d).",
                                   resp.status_code, backoff, attempt + 1)
                    time.sleep(backoff)
                    continue
                resp.raise_for_status()
                data = resp.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
        raise ValueError(f"Unknown LLM provider {self.provider!r}")

    @staticmethod
    def _parse_score(raw: str) -> float:
        """Extract ``sentiment_score`` as a float from the LLM's JSON response."""
        data = json.loads(raw)
        return float(data["sentiment_score"])


# ----------------------------------------------------------------- internals
def _as_utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _clamp(x: float) -> float:
    return float(max(-1.0, min(1.0, x)))


def _iso_minute(dt: datetime) -> str:
    return _as_utc(dt).strftime("%Y-%m-%dT%H:%M:%S")


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp (incl. trailing 'Z') into a tz-aware datetime."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    return _as_utc(dt)


def _score_text(text: str) -> float:
    """Keyword sentiment in [-1, 1] (stand-in for an LLM)."""
    low = text.lower()
    neg = sum(term in low for term in _NEGATIVE_TERMS)
    pos = sum(term in low for term in _POSITIVE_TERMS)
    if neg == 0 and pos == 0:
        return 0.0
    return (pos - neg) / (pos + neg)


def _synthetic_headlines(team_name: str, as_of: datetime) -> List[NewsItem]:
    """Deterministic mock wire: a few dated headlines strictly before ``as_of``.

    The tone is seeded by a hash of (team, as_of-date) so a given fixture always
    yields the same signal across backtest runs (reproducibility), and items are
    stamped at fixed offsets before the cutoff so the strict window is exercised.
    """
    import hashlib

    seed = int(hashlib.sha256(f"{team_name}|{as_of.date()}".encode()).hexdigest(), 16)
    tone = seed % 3  # 0 negative, 1 neutral, 2 positive

    templates = {
        0: [
            f"{team_name} hit by injury crisis ahead of fixture",
            f"Key {team_name} forward ruled out with a knock",
            f"{team_name} captain a major doubt, fitness concern grows",
        ],
        1: [
            f"{team_name} name unchanged squad for the match",
            f"Coach previews {team_name}'s upcoming fixture",
            f"{team_name} complete final training session",
        ],
        2: [
            f"{team_name} receive boost as star striker returns fully fit",
            f"{team_name} squad in form and high on morale",
            f"Injured {team_name} defender recovered and available",
        ],
    }[tone]

    # Offsets in days before the cutoff: 1, 3, 10. The 10-day item falls outside
    # a 7-day lookback and must be filtered by get_team_news.
    offsets = (1, 3, 10)
    return [
        NewsItem(
            team=team_name,
            headline=headline,
            published=as_of - timedelta(days=off),
            source="mock-wire",
        )
        for headline, off in zip(templates, offsets)
    ]
