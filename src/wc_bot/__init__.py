"""World Cup paper-trading skeleton.

A thin, end-to-end "walking skeleton" that runs the full loop:

    L1 ingest -> L4 Elo model -> L5 Polymarket -> L6 edge -> L7 paper ledger

Each layer is deliberately the simplest thing that works so the chassis can be
validated before swapping in heavier models (Soccermatics features, ML, RAG).
"""

__version__ = "0.1.0"
