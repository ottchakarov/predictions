import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wc_bot.ledger import FIELDNAMES, STATUS_OPEN, STATUS_SETTLED  # noqa: E402


def _load_settle_module():
    spec = importlib.util.spec_from_file_location(
        "settle_ledger", ROOT / "scripts" / "settle_ledger.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


settle_mod = _load_settle_module()


def _row(**overrides) -> dict:
    base = {c: "" for c in FIELDNAMES}
    base.update(
        {
            "match_id": "M1",
            "match": "A vs B",
            "outcome": "Yes",
            "market_price": "0.50",
            "stake": "50.0",
            "token_id": "T1",
            "match_time": "2026-06-25T18:00:00+00:00",
            "status": STATUS_OPEN,
        }
    )
    base.update(overrides)
    return base


def test_resolution_price_is_deterministic():
    a = settle_mod.get_resolution_price("TOKEN_ABC")
    b = settle_mod.get_resolution_price("TOKEN_ABC")
    assert a == b
    assert a in (0.0, 1.0)


def test_pnl_math_win_and_loss():
    now = datetime(2026, 6, 26, tzinfo=timezone.utc)
    # Pick token ids whose mocked resolution we read directly, so the test is
    # independent of the hashing detail.
    win_token = next(
        t for t in (f"W{i}" for i in range(100))
        if settle_mod.get_resolution_price(t) == 1.0
    )
    loss_token = next(
        t for t in (f"L{i}" for i in range(100))
        if settle_mod.get_resolution_price(t) == 0.0
    )

    df = pd.DataFrame(
        [
            _row(token_id=win_token, market_price="0.50", stake="50.0"),
            _row(token_id=loss_token, market_price="0.40", stake="40.0", match_id="M2"),
        ]
    )[FIELDNAMES]

    updated, settled = settle_mod.settle(df, now=now)

    assert len(settled) == 2
    # Win: shares = 50/0.50 = 100, payout 100, pnl = 100 - 50 = 50.
    win_pnl = settled.loc[settled["entry"] == 0.50, "pnl"].iloc[0]
    assert abs(win_pnl - 50.0) < 1e-6
    # Loss: payout 0, pnl = -stake = -40.
    loss_pnl = settled.loc[settled["entry"] == 0.40, "pnl"].iloc[0]
    assert abs(loss_pnl - (-40.0)) < 1e-6
    assert (updated["status"] == STATUS_SETTLED).all()


def test_open_row_without_match_time_raises():
    now = datetime(2026, 6, 26, tzinfo=timezone.utc)
    df = pd.DataFrame([_row(match_time="")])[FIELDNAMES]  # blank match_time, OPEN
    with pytest.raises(ValueError, match="match_time"):
        settle_mod.settle(df, now=now)


def test_open_row_with_unparseable_match_time_raises():
    now = datetime(2026, 6, 26, tzinfo=timezone.utc)
    df = pd.DataFrame([_row(match_time="not-a-date")])[FIELDNAMES]
    with pytest.raises(ValueError):
        settle_mod.settle(df, now=now)


def test_future_match_not_settled_and_settled_untouched():
    now = datetime(2026, 6, 26, tzinfo=timezone.utc)
    df = pd.DataFrame(
        [
            _row(match_time="2026-12-01T00:00:00+00:00"),  # future -> stays OPEN
            _row(match_id="M2", status=STATUS_SETTLED, pnl="99.0"),  # immutable
        ]
    )[FIELDNAMES]

    updated, settled = settle_mod.settle(df, now=now)

    assert settled.empty
    assert updated.iloc[0]["status"] == STATUS_OPEN
    # Already-settled row is left exactly as-is.
    assert updated.iloc[1]["status"] == STATUS_SETTLED
    assert updated.iloc[1]["pnl"] == "99.0"
