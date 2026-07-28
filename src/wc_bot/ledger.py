"""L7 - Paper ledger (append-only, idempotent).

Instead of signing a Polygon transaction, flagged bets are appended to a local
CSV. This is the safety valve that makes the whole skeleton runnable live: the
exact same loop that *would* trade just records what it would have done, so you
can measure realised edge before risking a cent.

Operational guarantees (execution-discipline phase):
* **Idempotency / de-dup.** A polling loop re-sees the same edge every cycle. We
  enforce *one trade maximum per (match_id, token_id, market_type)*: if a row for
  that composite key already exists, ``record`` blocks the write and returns
  False. This keeps the paper trail honest under ``run_forever``.
* **Immutability of past entries.** ``record`` only ever *appends*; it never
  rewrites or mutates an existing row. Settlement (see ``scripts/settle_ledger``)
  is the only writer permitted to flip a row's ``status`` OPEN -> SETTLED, and it
  touches nothing else.

When you graduate to real execution, this file is the single seam to replace:
swap ``record`` for a call to the Polymarket CLOB order endpoint (with wallet
signing) and keep everything else identical.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Set, Tuple

from .decision import TradeSignal

# The composite key that defines "the same bet" for de-duplication purposes.
DEDUP_KEY = ("match_id", "token_id", "market_type")

FIELDNAMES = [
    "timestamp",
    "match_id",
    "market_slug",
    "match",
    "market_type",
    "model_version",
    "outcome",
    "model_prob",
    "market_price",
    "fair_market_prob",
    "edge",
    "ev_per_dollar",
    "kelly_fraction",
    "sharp_multiplier",
    "net_sharp_alignment",
    "stake",
    "bankroll",
    "token_id",
    "match_time",
    "status",
    "resolution_price",
    "pnl",
    "settled_at",
]

STATUS_OPEN = "OPEN"
STATUS_SETTLED = "SETTLED"
STATUS_VETOED = "VETOED"  # signal cleared the edge filter but sharp money vetoed it
STATUS_INCOMPLETE = "INCOMPLETE"  # market missing a required quote; never priced

LedgerKey = Tuple[str, str, str]


@dataclass
class PaperLedger:
    path: Path
    _keys: Set[LedgerKey] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="") as fh:
                csv.DictWriter(fh, fieldnames=FIELDNAMES).writeheader()
        else:
            # A file written under an older schema has a stale header; appending
            # new-schema rows to it silently MISALIGNS columns (e.g. token_id),
            # which corrupts the de-dup key and lets duplicate trades through.
            # Migrate it to the current schema (by column name) before use.
            self._migrate_if_stale()
        # Load existing state into memory so de-dup survives process restarts.
        self._keys = self._load_keys()

    def _migrate_if_stale(self) -> None:
        """Rewrite the ledger to the current FIELDNAMES if its header drifted.

        Maps existing rows by column NAME into the current schema (new columns are
        left blank), preserving recorded values while restoring column alignment.
        """
        with self.path.open("r", newline="") as fh:
            reader = csv.DictReader(fh)
            header = reader.fieldnames or []
            if header == FIELDNAMES:
                return
            rows = [
                {col: (row.get(col) or "") for col in FIELDNAMES} for row in reader
            ]
        with self.path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        print(
            f"  [ledger] migrated {self.path.name} to current schema "
            f"({len(header)}->{len(FIELDNAMES)} columns)."
        )

    # ----------------------------------------------------------- idempotency
    def _load_keys(self) -> Set[LedgerKey]:
        """Load the current CSV state and index it by the de-dup composite key."""
        keys: Set[LedgerKey] = set()
        if not self.path.exists():
            return keys
        with self.path.open("r", newline="") as fh:
            for row in csv.DictReader(fh):
                keys.add(self._key_from_row(row))
        return keys

    def reload(self) -> None:
        """Re-sync the in-memory de-dup index from disk (e.g. after settlement)."""
        self._keys = self._load_keys()

    @staticmethod
    def _key_from_row(row: dict) -> LedgerKey:
        return tuple((row.get(col) or "") for col in DEDUP_KEY)  # type: ignore[return-value]

    def has_record(self, match_id: str, token_id: str, market_type: str) -> bool:
        """True if a trade already exists for this (match_id, token_id, market_type)."""
        return (match_id, token_id, market_type) in self._keys

    # ---------------------------------------------------------------- writer
    def record(
        self,
        signal: TradeSignal,
        *,
        match_id: str,
        market_slug: str,
        match: str,
        bankroll: float,
        token_id: str = "",
        market_type: str = "moneyline",
        match_time: str = "",
        model_version: str = "",
        status: str = STATUS_OPEN,
        pnl: Optional[float] = None,
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """Append a paper trade. Returns True if written, False if blocked by de-dup.

        Strict "Block" policy: at most one trade per (match_id, token_id,
        market_type). Existing rows are never mutated here.

        ``status`` defaults to OPEN (a live paper trade awaiting settlement). Pass
        STATUS_VETOED with ``pnl=0.0`` to record a sharp-vetoed signal: a 0-stake,
        terminal row that the settlement engine ignores but that preserves the
        model probability / market price so the foregone edge stays auditable.
        """
        key: LedgerKey = (match_id, token_id, market_type)
        if key in self._keys:
            return False

        ts = (timestamp or datetime.now(timezone.utc)).isoformat()
        row = {
            "timestamp": ts,
            "match_id": match_id,
            "market_slug": market_slug,
            "match": match,
            "market_type": market_type,
            "model_version": model_version,
            "outcome": signal.outcome,
            "model_prob": round(signal.model_prob, 4),
            "market_price": round(signal.market_price, 4),
            "fair_market_prob": round(signal.fair_market_prob, 4),
            "edge": round(signal.edge, 4),
            "ev_per_dollar": round(signal.ev_per_dollar, 4),
            "kelly_fraction": round(signal.kelly_fraction, 4),
            "sharp_multiplier": round(signal.sharp_multiplier, 4),
            "net_sharp_alignment": round(signal.net_sharp_alignment, 4),
            "stake": signal.stake,
            "bankroll": round(bankroll, 2),
            "token_id": token_id,
            "match_time": match_time,
            "status": status,
            "resolution_price": "",
            "pnl": "" if pnl is None else round(pnl, 2),
            "settled_at": "",
        }
        with self.path.open("a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=FIELDNAMES).writerow(row)
        self._keys.add(key)
        return True
