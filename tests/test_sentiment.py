import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wc_bot.rag import NewsFetcher, NewsItem, SentimentAgent  # noqa: E402

AS_OF = datetime(2026, 6, 27, tzinfo=timezone.utc)


# ----------------------------------------------------------- NewsFetcher (PIT)
def test_news_is_strictly_before_as_of():
    items = NewsFetcher().get_team_news("Croatia", AS_OF, lookback_days=14)
    assert items  # mock yields some
    for it in items:
        assert it.published < AS_OF  # no look-ahead, ever (data_hygiene.mdc)


def test_news_respects_lookback_window():
    wide = NewsFetcher().get_team_news("Croatia", AS_OF, lookback_days=14)
    narrow = NewsFetcher().get_team_news("Croatia", AS_OF, lookback_days=2)
    # The 10-day-old synthetic item is dropped by the 2-day window.
    assert len(narrow) < len(wide)
    for it in narrow:
        assert it.published >= AS_OF - timedelta(days=2)


def test_news_requires_as_of():
    with pytest.raises(ValueError):
        NewsFetcher().get_team_news("Croatia", None)


def test_news_is_deterministic():
    a = NewsFetcher().get_team_news("Brazil", AS_OF)
    b = NewsFetcher().get_team_news("Brazil", AS_OF)
    assert [i.headline for i in a] == [i.headline for i in b]


# --------------------------------------------------- NewsFetcher (live, mocked)
def _fake_response(articles):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"articles": articles}
    return resp


def test_live_news_parses_and_filters_strictly():
    # cutoff = 2026-06-27, 7d window => [2026-06-20, 2026-06-27).
    arts = [
        {  # in-window, strictly before cutoff -> kept
            "title": "Croatia injury crisis",
            "description": "key man out",
            "publishedAt": "2026-06-26T10:00:00Z",
            "source": {"name": "Wire"},
        },
        {  # exactly at cutoff -> dropped by strict '<' (no look-ahead)
            "title": "Future leak",
            "description": "",
            "publishedAt": "2026-06-27T00:00:00Z",
            "source": {"name": "Wire"},
        },
        {  # older than lookback -> dropped
            "title": "Too old",
            "description": "",
            "publishedAt": "2026-06-01T00:00:00Z",
            "source": {"name": "Wire"},
        },
    ]
    fetcher = NewsFetcher(mock=False, api_key="k")
    session = MagicMock()
    session.get.return_value = _fake_response(arts)
    with patch.object(fetcher, "_get_session", return_value=session):
        items = fetcher.get_team_news("Croatia", AS_OF, lookback_days=7)

    assert len(items) == 1
    assert items[0].published < AS_OF
    assert items[0].source == "Wire"
    # Point-in-time enforced at the HTTP layer too: 'to' bound == cutoff.
    _, kwargs = session.get.call_args
    assert kwargs["params"]["to"] == "2026-06-27T00:00:00"
    assert kwargs["params"]["from"] == "2026-06-20T00:00:00"
    assert kwargs["timeout"] == fetcher.timeout


def test_live_news_returns_empty_on_timeout():
    fetcher = NewsFetcher(mock=False, api_key="k")
    session = MagicMock()
    session.get.side_effect = TimeoutError("connection timed out")
    with patch.object(fetcher, "_get_session", return_value=session):
        assert fetcher.get_team_news("Croatia", AS_OF) == []


def test_live_news_returns_empty_on_rate_limit():
    fetcher = NewsFetcher(mock=False, api_key="k")
    resp = MagicMock()
    resp.raise_for_status.side_effect = RuntimeError("429 Too Many Requests")
    session = MagicMock()
    session.get.return_value = resp
    with patch.object(fetcher, "_get_session", return_value=session):
        assert fetcher.get_team_news("Croatia", AS_OF) == []


# -------------------------------------------------------- SentimentAgent
def _item(headline: str) -> NewsItem:
    return NewsItem(team="X", headline=headline, published=AS_OF - timedelta(days=1))


def test_empty_news_is_neutral_safe_default():
    assert SentimentAgent().analyze_squad_state([]) == 0.0


def test_injury_news_is_negative():
    items = [_item("Star striker ruled out with injury"), _item("Captain a major doubt")]
    assert SentimentAgent().analyze_squad_state(items) < 0


def test_recovery_news_is_positive():
    items = [_item("Key forward returns fully fit"), _item("Squad high on morale, boost")]
    assert SentimentAgent().analyze_squad_state(items) > 0


def test_score_is_bounded():
    items = [_item("injury injury injured ruled out crisis suspended") for _ in range(5)]
    s = SentimentAgent().analyze_squad_state(items)
    assert -1.0 <= s <= 1.0


def test_team_sentiment_composes_fetch_and_analyze():
    s = SentimentAgent().team_sentiment("Croatia", AS_OF, NewsFetcher(), lookback_days=14)
    assert -1.0 <= s <= 1.0


# ------------------------------------------------- SentimentAgent (live, mocked)
def test_live_agent_parses_json_score():
    agent = SentimentAgent(mock=False, api_key="k")
    with patch.object(agent, "_call_llm", return_value='{"sentiment_score": -0.7}'):
        assert agent.analyze_squad_state([_item("injury")]) == -0.7


def test_live_agent_clamps_out_of_range():
    agent = SentimentAgent(mock=False)
    with patch.object(agent, "_call_llm", return_value='{"sentiment_score": 5.0}'):
        assert agent.analyze_squad_state([_item("boost")]) == 1.0


def test_live_agent_neutral_on_bad_json():
    agent = SentimentAgent(mock=False)
    with patch.object(agent, "_call_llm", return_value="not json at all"):
        assert agent.analyze_squad_state([_item("injury")]) == 0.0


def test_live_agent_neutral_on_llm_error():
    agent = SentimentAgent(mock=False)
    with patch.object(agent, "_call_llm", side_effect=TimeoutError("slow")):
        assert agent.analyze_squad_state([_item("injury")]) == 0.0


def test_live_agent_empty_news_skips_llm():
    agent = SentimentAgent(mock=False)
    with patch.object(agent, "_call_llm") as call:
        assert agent.analyze_squad_state([]) == 0.0
    call.assert_not_called()


def test_call_llm_uses_openai_sdk_json_mode(monkeypatch):
    fake_openai = MagicMock()
    fake_client = fake_openai.OpenAI.return_value
    fake_client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content='{"sentiment_score": 0.3}'))
    ]
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    agent = SentimentAgent(mock=False, api_key="k", model="m", min_call_interval=0)
    out = agent._call_llm("prompt")

    assert '"sentiment_score": 0.3' in out
    fake_openai.OpenAI.assert_called_once_with(api_key="k", timeout=agent.timeout)
    _, kwargs = fake_client.chat.completions.create.call_args
    assert kwargs["model"] == "m"
    assert kwargs["response_format"] == {"type": "json_object"}


# ------------------------------------------------- SentimentAgent (Gemini, mocked)
def _gemini_response(code, text='{"sentiment_score": -0.4}'):
    resp = MagicMock()
    resp.status_code = code
    resp.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": text}]}}]
    }
    if code >= 400:
        resp.raise_for_status.side_effect = RuntimeError(f"{code} error")
    else:
        resp.raise_for_status.return_value = None
    return resp


def test_gemini_provider_defaults_and_parses(monkeypatch):
    import requests

    captured = {}

    def fake_post(url, params=None, json=None, timeout=None):
        captured["url"] = url
        captured["key"] = (params or {}).get("key")
        return _gemini_response(200)

    monkeypatch.setattr(requests, "post", fake_post)
    agent = SentimentAgent(mock=False, provider="gemini", api_key="k", min_call_interval=0)
    assert agent.model == "gemini-2.0-flash"  # provider default
    assert agent.analyze_squad_state([_item("injury")]) == -0.4
    assert "gemini-2.0-flash:generateContent" in captured["url"]
    assert captured["key"] == "k"


def test_gemini_retries_on_429_then_succeeds(monkeypatch):
    import requests
    import wc_bot.rag.sentiment as S

    responses = [_gemini_response(429), _gemini_response(200, '{"sentiment_score": 0.5}')]
    calls = {"n": 0}

    def fake_post(url, params=None, json=None, timeout=None):
        r = responses[calls["n"]]
        calls["n"] += 1
        return r

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(S.time, "sleep", lambda s: None)  # no real backoff wait
    agent = SentimentAgent(mock=False, provider="gemini", api_key="k",
                           min_call_interval=0, max_retries=2)
    assert agent.analyze_squad_state([_item("boost")]) == 0.5
    assert calls["n"] == 2  # 429 once, then retried successfully


def test_gemini_api_key_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "env-key-123")
    agent = SentimentAgent(mock=False, provider="gemini")
    assert agent.api_key == "env-key-123"


def test_throttle_spaces_consecutive_calls(monkeypatch):
    import wc_bot.rag.sentiment as S

    slept = []
    monkeypatch.setattr(S.time, "sleep", lambda s: slept.append(s))
    agent = SentimentAgent(mock=False, provider="gemini", api_key="k", min_call_interval=0.5)
    agent._last_call_ts = S.time.monotonic()  # pretend a call just happened
    agent._throttle()
    assert slept and 0 < slept[0] <= 0.5
