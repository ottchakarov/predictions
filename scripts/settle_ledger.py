#!/usr/bin/env python3
"""Settlement & PnL engine for the paper ledger.

Operational role: take OPEN paper trades whose match has concluded, resolve them,
compute realised PnL, and flip those rows to SETTLED. It is the *only* writer
allowed to mutate existing ledger rows, and it mutates **only** the OPEN rows it
settles — every other row (already SETTLED, or OPEN-but-not-yet-concluded) is
written back byte-for-byte, preserving an immutable audit trail.

PnL model (binary outcome token bought at ``entry_price``):

    shares  = stake / entry_price          # contracts bought with the staked $
    payout  = shares * resolution_price     # resolution_price is $1.00 win / $0.00 loss
    pnl     = payout - stake

Run::

    python scripts/settle_ledger.py                      # settle concluded OPEN rows
    python scripts/settle_ledger.py --dry-run            # preview, write nothing
    python scripts/settle_ledger.py --ledger data/paper_trades.csv
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wc_bot.ledger import FIELDNAMES, STATUS_OPEN, STATUS_SETTLED  # noqa: E402


def get_resolution_price(token_id: str, *, client: Optional[object] = None) -> float:
    """Return the settlement price of an outcome token: 1.0 (win) or 0.0 (loss).

    STUB. Structured to accept a real data source later: when ``client`` is wired
    up this should query Polymarket for the resolved outcome — e.g. the Gamma
    market's ``umaResolutionStatus`` / post-resolution ``outcomePrices``, or the
    on-chain UMA oracle result for the market's ``conditionId``. Until then we
    return a deterministic pseudo-resolution derived from the token id so PnL is
    stable and reproducible across re-runs.
    """
    if client is not None:
        # TODO(live): replace with real resolution lookup, e.g.
        #   market = client.get_market_by_condition(condition_id)
        #   return float(market.resolved_outcome_price(token_id))
        raise NotImplementedError("Live resolution source not wired up yet.")

    digest = int(hashlib.sha256(token_id.encode("utf-8")).hexdigest(), 16)
    return 1.0 if digest % 2 == 0 else 0.0


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(value: str) -> Optional[datetime]:
    value = (value or "").strip()
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    # Treat naive timestamps as UTC so comparisons are well-defined.
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _match_conclusion_time(row: pd.Series) -> datetime:
    """Return an OPEN row's match kickoff time, or raise on missing/bad data.

    STRICT point-in-time hygiene: there is **no fallback** to the bet timestamp.
    Guessing when an event concluded introduces non-deterministic settlement, so
    an OPEN row without a valid explicit ``match_time`` halts the run.
    """
    raw = (row.get("match_time", "") or "").strip()
    if not raw:
        raise ValueError(
            f"OPEN row (match_id={row.get('match_id')!r}, token_id="
            f"{row.get('token_id')!r}) has no 'match_time'. Refusing to guess the "
            f"event conclusion time — settlement must be point-in-time correct."
        )
    parsed = _parse_ts(raw)
    if parsed is None:
        raise ValueError(
            f"OPEN row (match_id={row.get('match_id')!r}) has an unparseable "
            f"'match_time': {raw!r}. Expected ISO-8601."
        )
    return parsed


def load_ledger(path: Path) -> pd.DataFrame:
    # keep_default_na=False so empty cells stay "" (not NaN) and round-trip cleanly.
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in FIELDNAMES:
        if col not in df.columns:
            df[col] = ""
    # Default a missing/blank status to OPEN (legacy rows).
    df["status"] = df["status"].replace("", STATUS_OPEN)
    return df[FIELDNAMES]


def settle(df: pd.DataFrame, *, now: Optional[datetime] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Settle eligible OPEN rows in-place on a copy. Returns (full_df, settled_df)."""
    now = now or _now()
    df = df.copy()
    settled_rows = []

    for idx, row in df.iterrows():
        if row["status"] != STATUS_OPEN:
            continue
        # Strict: raises if an OPEN row lacks a valid match_time (halts the run).
        if _match_conclusion_time(row) >= now:
            continue  # match not concluded yet

        token_id = row["token_id"]
        entry_price = float(row["market_price"]) if row["market_price"] else 0.0
        stake = float(row["stake"]) if row["stake"] else 0.0
        if entry_price <= 0.0:
            continue  # cannot value a contract with no entry price

        resolution = get_resolution_price(token_id)
        shares = stake / entry_price
        payout = shares * resolution
        pnl = round(payout - stake, 2)

        df.at[idx, "status"] = STATUS_SETTLED
        df.at[idx, "resolution_price"] = f"{resolution:.2f}"
        df.at[idx, "pnl"] = f"{pnl:.2f}"
        df.at[idx, "settled_at"] = now.isoformat()

        settled_rows.append(
            {
                "match": row["match"],
                "outcome": row["outcome"],
                "entry": round(entry_price, 4),
                "resolution": resolution,
                "stake": round(stake, 2),
                "pnl": pnl,
            }
        )

    return df, pd.DataFrame(settled_rows)


def print_summary(settled: pd.DataFrame) -> None:
    if settled.empty:
        print("No concluded OPEN trades to settle.")
        return

    print("\nSettled trades")
    print("-" * 60)
    print(settled.to_string(index=False))

    total_staked = settled["stake"].sum()
    total_pnl = settled["pnl"].sum()
    wins = int((settled["resolution"] == 1.0).sum())
    n = len(settled)
    roi = (total_pnl / total_staked * 100.0) if total_staked else 0.0

    print("-" * 60)
    print(f"  bets settled : {n}")
    print(f"  win rate     : {wins}/{n} ({wins / n * 100:.1f}%)")
    print(f"  total staked : ${total_staked:.2f}")
    print(f"  realised PnL : ${total_pnl:+.2f}")
    print(f"  ROI          : {roi:+.2f}%")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", default=str(ROOT / "data" / "paper_trades.csv"))
    parser.add_argument(
        "--dry-run", action="store_true", help="compute & print PnL but do not write"
    )
    args = parser.parse_args()

    path = Path(args.ledger)
    if not path.exists():
        print(f"Ledger not found: {path}")
        return 1

    df = load_ledger(path)
    open_count = int((df["status"] == STATUS_OPEN).sum())
    print(f"Loaded {len(df)} ledger row(s); {open_count} OPEN.")

    updated, settled = settle(df)
    print_summary(settled)

    if settled.empty:
        return 0

    if args.dry_run:
        print("\n[dry-run] No changes written.")
        return 0

    updated.to_csv(path, index=False)
    print(f"\nUpdated {len(settled)} row(s) to SETTLED -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
