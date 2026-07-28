import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wc_bot.pipeline import (  # noqa: E402
    LIVE_WATCHLIST,
    MOCK_WATCHLIST,
    default_watchlist_path,
    load_watchlist,
)


def test_default_path_follows_run_mode():
    assert default_watchlist_path(live=True).endswith("watchlist.live.json")
    assert default_watchlist_path(live=False).endswith("watchlist.mock.json")


def test_mock_watchlist_is_purely_mock():
    wms = load_watchlist(MOCK_WATCHLIST)
    assert wms
    for wm in wms:
        for spec in wm.tokens.values():
            assert "mock_price" in spec  # mock entries are priced offline
            assert not spec["token_id"].isdigit()  # fake ids, not real CLOB hashes


def test_live_watchlist_is_purely_real():
    wms = load_watchlist(LIVE_WATCHLIST)
    assert wms
    for wm in wms:
        assert wm.condition_id.startswith("0x")  # on-chain resolution key present
        for spec in wm.tokens.values():
            assert "mock_price" not in spec
            assert spec["token_id"].isdigit()  # real ERC-1155 CTF token id


def test_live_watchlist_has_a_world_cup_moneyline_market():
    wms = load_watchlist(LIVE_WATCHLIST)
    wm = wms[0]
    # Binary moneyline: Yes = home win, No = draw-or-away.
    assert "Yes" in wm.tokens and "No" in wm.tokens
    assert wm.tokens["Yes"]["model"] == "home"
    assert wm.tokens["No"]["model"] == "draw+away"
    assert wm.home_team and wm.away_team
