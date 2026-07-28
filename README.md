# World Cup Paper-Trading Skeleton

A thin, **end-to-end "walking skeleton"** for a prediction-market trading bot. It
runs the entire pipeline — from raw match data to a logged paper bet — using the
simplest possible version of every layer. The point is to prove a real,
*executable* edge exists **before** investing in heavy modelling, so each layer
is a clean seam you can later upgrade in isolation.

```
L1 Ingest   ──►  L4 Elo model  ──►  L5 Polymarket  ──►  L6 Edge + Kelly  ──►  L7 Paper ledger
historical       baseline win       read-only order      vig removal,          append-only CSV
match data       probabilities      book midpoints       EV filter, sizing     (no on-chain tx)
```

Everything is **read-only / paper**: it never signs a Polygon transaction. The
only seam you change to go live is L7 (`ledger.py`).

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Head-to-head L4 backtest — Elo vs Dixon-Coles, point-in-time, with PnL
python scripts/backtest_h2h.py

# 2. Offline end-to-end demo — synthetic books, defaults to config/watchlist.mock.json
python scripts/run_skeleton.py

# 3. Live, read-only single pass — live CLOB midpoints, defaults to config/watchlist.live.json
python scripts/run_skeleton.py --live-odds

# 4. Live autonomous loop with the on-chain sharp-money sizing modifier (Ctrl-C to stop)
python scripts/run_skeleton.py --live-odds --sharp --loop --interval 90
```

`backtest_h2h.py` steps through history period-by-period (refitting both models
only on data strictly prior to each period — no leakage) and prints log-loss,
Brier, calibration error (ECE) and a flat-stake theoretical ROI for Elo vs
Dixon-Coles side by side, so model upgrades are judged on a rigorous baseline
before any feature engineering. Add `--use-rag` to fuse the L2 news-sentiment
signal into a third `DC+RAG` column (point-in-time per match) and have the report
state whether the signal actually improves log-loss/Brier over the pure baseline.

## The layers (and the book that informs each)

| Layer | File | Source |
|------|------|--------|
| **L1** Data ingestion (point-in-time) | `src/wc_bot/ingest/` (`matches.py`, `chain_tracker.py`) | — |
| **L2** Agentic RAG sentiment (mock + live news API/LLM) | `src/wc_bot/rag/sentiment.py` | FinGPT |
| **L4** Elo baseline model | `src/wc_bot/elo.py` | *Soccermatics* (later) |
| **L5** Polymarket order book | `src/wc_bot/polymarket.py` | Vitalik on prediction markets |
| **L6** Edge / EV / Kelly | `src/wc_bot/decision.py` | *Logic of Sports Betting*, *Algorithmic Trading* |
| **L7** Paper ledger | `src/wc_bot/ledger.py` | *Algorithmic Trading* (execution) |
| Loop | `src/wc_bot/pipeline.py` | — |

### Data: `data/international_results.csv`
Every men's international from 1872–present (`martj42/international_results`). We
use the **full history to seed Elo** and the FIFA World Cup subset to evaluate.

### The one invariant that matters: point-in-time correctness
Per López de Prado, no computation ever sees the future. `ingest.load_matches`
guarantees chronological order, and `Elo.fit(record=True)` records each
prediction using only ratings that existed *before* that match — a
leakage-free backtest log.

## Going live: replace the placeholders

Watchlists are split by run mode: `config/watchlist.mock.json` (`MOCK_*` ids +
`mock_price`, used by default) and `config/watchlist.live.json` (real CLOB token
ids + on-chain `condition_id`, used by default with `--live-odds`). For real
trading, edit `watchlist.live.json`:

1. Find the World Cup market via the Gamma API and read `clobTokenIds` /
   `conditionId` (token ids are ephemeral — refresh before each run).
2. Put each real CLOB **token id** into the watchlist (no `mock_price`).
3. Make sure each `home_team` / `away_team` matches the dataset spelling
   (e.g. `United States`, `South Korea`); the runner warns on misses.
4. `model` selects the model probability per outcome token: a base outcome
   (`home`/`draw`/`away`) or a `+`-combo (e.g. `draw+away` for a "home team
   does NOT win" token).

`--sharp` adds the L6 on-chain sharp-money sizing modifier; with `--live-odds` it
queries Polymarket's Goldsky orderbook subgraph (point-in-time, with graceful
fallback to raw Kelly on any subgraph error).

## Execution discipline & settlement

The ledger is **append-only and idempotent**, and a separate engine handles PnL.

**De-duplication (1 trade per token per match).** Under `--loop` the same edge
reappears every cycle. `PaperLedger` enforces a strict "Block" policy on the
composite key `[match_id, token_id, market_type]`: if a row exists, the write is
skipped (and logged). State is loaded from the CSV on startup, so de-dup survives
restarts.

**Skipping concluded matches.** Set `"is_active": false` on a watchlist match and
`run_skeleton.py` won't query Polymarket for it.

**Settlement & PnL.** Once a match's `match_time` is in the past, settle the
open paper trades:

```bash
python scripts/settle_ledger.py --dry-run   # preview PnL, write nothing
python scripts/settle_ledger.py             # flip concluded OPEN rows -> SETTLED
```

It computes `pnl = stake * (resolution_price - entry_price) / entry_price`,
prints a summary (win rate, total staked, realised PnL, ROI), and rewrites the
CSV. It is the **only** writer allowed to mutate rows, and it touches *only* the
OPEN rows it settles — already-SETTLED and not-yet-concluded rows round-trip
untouched, preserving the audit trail. `get_resolution_price(token_id)` is a
deterministic stub today, structured to swap in the real Polymarket/UMA
resolution later.

Ledger schema: `timestamp, match_id, market_slug, match, market_type, outcome,
model_prob, market_price, fair_market_prob, edge, ev_per_dollar, kelly_fraction,
stake, bankroll, token_id, match_time, status, resolution_price, pnl, settled_at`.

## L3 Feature Store + L4 Dixon-Coles (the model upgrade)

The Elo baseline now has a generative successor, with a clean L3/L4 split.

**`src/wc_bot/features/store.py` — `FeatureStore` (L3).** Parses the results CSV
into a structured frame (`home_team, away_team, home_goals, away_goals, date,
neutral`) and serves it with **point-in-time** retrieval. `training_frame(as_of)`
returns only fixtures *strictly* before `as_of` (`date < as_of`), the anti-
leakage guarantee for backtests. It is fully decoupled — it knows nothing about
any model.

**`src/wc_bot/models/dixon_coles.py` — `DixonColesModel` (L4).** A weighted
Dixon-Coles bivariate Poisson model:

- attack/defence vectors per team via **weighted maximum likelihood** (analytic
  gradient + L-BFGS-B; fits all 336 teams over ~49k matches in <1s);
- **time decay** (`half_life_days`) so recent internationals dominate;
- the Dixon-Coles **`rho`** low-score correction, fit by profiled 1-D MLE;
- `score_matrix()` for exact scorelines and `matrix_to_1x2()` to collapse to
  Home/Draw/Away.

It mirrors the Elo interface (`fit` / `predict` / `match_probabilities`), so it is
a **drop-in** for the pipeline:

```python
from wc_bot.features import FeatureStore
from wc_bot.models import DixonColesModel

fs = FeatureStore()
model = DixonColesModel().fit(fs.training_frame(as_of=kickoff), reference_date=kickoff)
model.match_probabilities("Brazil", "United States", neutral=True)
# -> {'home': 0.73, 'draw': 0.17, 'away': 0.10}
```

> Note on rigour: `rho` is the DC low-score *correction*, distinct from the time-
> decay half-life. The attack/defence block is fit with an exact analytic
> gradient; `rho` is then profiled on the full likelihood (block-coordinate),
> keeping the math clean and fast.

### Model-agnostic pipeline

The pipeline toggles between L4 models via config; both obey the same interface
(`fit` / `match_probabilities` / `knows_team` / `model_version`), and the Dixon-
Coles path wires the `FeatureStore` in with strict point-in-time enforcement.

```bash
python scripts/run_skeleton.py --mock --model elo
python scripts/run_skeleton.py --mock --model dixon_coles --as-of 2026-06-01
```

For Dixon-Coles, `--as-of` is enforced as `date < as_of` in the `FeatureStore`
(no look-ahead) and used as the time-decay reference. Every paper trade records a
`model_version` column (`elo-1.0` / `dixon_coles-1.0`) so you can attribute each
bet to the algorithm that generated it.

## Tuning knobs

- `--target-edge` (default 0.04): minimum model-vs-fair-market edge to act on —
  your buffer against fees and model error.
- `--kelly` (default 0.25): fractional-Kelly multiplier. Full Kelly is
  growth-optimal but far too volatile; quarter-Kelly is the usual practical pick.
- `DecisionConfig.max_stake_fraction` (0.05): hard per-bet bankroll cap.

## Upgrade path (the chassis is plug-and-play)

- **Better model (L4):** swap `Elo` for a Dixon-Coles / xG / ML model — only the
  object returning `match_probabilities(...)` changes.
- **Features (L3):** add a feature store feeding the model (*Soccermatics*).
- **RAG (L2):** fuse injuries/news/lineups into the probability (FinGPT). Run
  `run_skeleton.py --model dixon_coles --use-rag` (add `--live-odds` to use the
  live news API + LLM; otherwise a deterministic mock). Sentiment is fetched
  strictly point-in-time and degrades safely to neutral `0.0` on any API/LLM error.
- **Real execution (L7):** replace `PaperLedger.record` with a signed CLOB order.

## Tests

```bash
pytest -q
```
