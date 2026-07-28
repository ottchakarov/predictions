import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wc_bot.decision import TradeSignal  # noqa: E402
from wc_bot.ledger import STATUS_OPEN, PaperLedger  # noqa: E402


def _signal(outcome: str = "Yes", *, sharp_multiplier=1.0, net_sharp_alignment=0.0) -> TradeSignal:
    return TradeSignal(
        outcome=outcome,
        model_prob=0.60,
        market_price=0.50,
        fair_market_prob=0.50,
        edge=0.10,
        ev_per_dollar=0.20,
        kelly_fraction=0.05,
        stake=50.0,
        is_bet=True,
        sharp_multiplier=sharp_multiplier,
        net_sharp_alignment=net_sharp_alignment,
    )


def _record(ledger: PaperLedger, *, match_id, token_id, market_type="moneyline"):
    return ledger.record(
        _signal(),
        match_id=match_id,
        market_slug="slug",
        match="A vs B",
        bankroll=1000.0,
        token_id=token_id,
        market_type=market_type,
        match_time="2026-06-25T18:00:00+00:00",
    )


def test_first_write_succeeds_then_dedup_blocks(tmp_path):
    led = PaperLedger(tmp_path / "ledger.csv")
    assert _record(led, match_id="M1", token_id="T1") is True
    # Same composite key -> blocked.
    assert _record(led, match_id="M1", token_id="T1") is False
    rows = (tmp_path / "ledger.csv").read_text().strip().splitlines()
    assert len(rows) == 2  # header + one trade


def test_different_token_or_market_type_is_distinct(tmp_path):
    led = PaperLedger(tmp_path / "ledger.csv")
    assert _record(led, match_id="M1", token_id="T1") is True
    assert _record(led, match_id="M1", token_id="T2") is True  # different token
    assert _record(led, match_id="M1", token_id="T1", market_type="total") is True
    rows = (tmp_path / "ledger.csv").read_text().strip().splitlines()
    assert len(rows) == 4  # header + three distinct trades


def test_dedup_survives_restart(tmp_path):
    path = tmp_path / "ledger.csv"
    assert _record(PaperLedger(path), match_id="M1", token_id="T1") is True
    # A fresh ledger object must load existing keys from disk and still block.
    assert _record(PaperLedger(path), match_id="M1", token_id="T1") is False


def test_stale_header_is_migrated_and_dedup_still_works(tmp_path):
    # Simulate a ledger written under an OLD schema (no model_version / sharp cols).
    path = tmp_path / "ledger.csv"
    old_header = [
        "timestamp", "match_id", "market_slug", "match", "market_type", "outcome",
        "model_prob", "market_price", "fair_market_prob", "edge", "ev_per_dollar",
        "kelly_fraction", "stake", "bankroll", "token_id", "match_time", "status",
        "resolution_price", "pnl", "settled_at",
    ]
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(old_header)
        w.writerow([
            "2026-06-01T00:00:00+00:00", "M1", "slug", "A vs B", "moneyline", "Yes",
            "0.6", "0.5", "0.5", "0.1", "0.2", "0.05", "50.0", "1000.0", "T1",
            "2026-06-25T18:00:00+00:00", "OPEN", "", "", "",
        ])

    # Opening the ledger must migrate the header in place...
    led = PaperLedger(path)
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames is not None
        row = next(reader)
    from wc_bot.ledger import FIELDNAMES
    assert list(row.keys()) == FIELDNAMES        # columns realigned
    assert row["token_id"] == "T1"               # value preserved by name, not position
    assert row["model_version"] == ""            # new column present, blank

    # ...and the de-dup key for the migrated row must still block a re-write.
    assert _record(led, match_id="M1", token_id="T1") is False


def test_new_rows_default_to_open(tmp_path):
    path = tmp_path / "ledger.csv"
    _record(PaperLedger(path), match_id="M1", token_id="T1")
    content = path.read_text()
    assert STATUS_OPEN in content


def test_sharp_columns_are_logged(tmp_path):
    path = tmp_path / "ledger.csv"
    led = PaperLedger(path)
    led.record(
        _signal(sharp_multiplier=1.35, net_sharp_alignment=0.7),
        match_id="M1",
        market_slug="slug",
        match="A vs B",
        bankroll=1000.0,
        token_id="T1",
        market_type="moneyline",
        match_time="2026-06-25T18:00:00+00:00",
    )
    with open(path, newline="") as fh:
        row = next(csv.DictReader(fh))
    assert "sharp_multiplier" in row and "net_sharp_alignment" in row
    assert float(row["sharp_multiplier"]) == 1.35
    assert float(row["net_sharp_alignment"]) == 0.7
