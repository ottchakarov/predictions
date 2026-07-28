# Quantitative Architecture Decision Record — World Cup Betting Bot

**Status:** Living document
**Scope:** How the seminal literature in this workspace constrains the design of
our prediction-market bot, with concrete, enforceable mappings to
`DixonColesModel`, `FeatureStore`, and `decision.py`.

This document is derived **only** from the texts provided in the workspace:

| Tag | Source |
|-----|--------|
| **[SM]**  | David Sumpter — *Soccermatics* |
| **[AFML]**| Marcos López de Prado — *Advances in Financial Machine Learning* |
| **[LSB]** | Ed Miller & Matthew Davidow — *The Logic of Sports Betting* |
| **[CHAN]**| Ernest Chan — *Algorithmic Trading: Winning Strategies and Their Rationale* |
| **[FG]**  | *FinGPT: Open-Source Financial Large Language Models* (L2/RAG layer only) |

The hard, non-negotiable rules distilled from this analysis are enforced in the
repository's root `.cursorrules`. This document is the *rationale*; `.cursorrules`
is the *law*.

---

## Part I — The Scoring Model (Poisson / xG) → `DixonColesModel`

### Principles extracted from *Soccermatics* [SM]

1. **Goals are Poisson-distributed because goal timing is memoryless.** Sumpter
   shows that the per-match goal histogram for a full league season is captured
   "remarkably well" by a Poisson distribution, and explains *why*: a Poisson
   distribution "arises whenever the timing of previous events has no effect on
   future events… neither the number of goals scored so far, nor the amount of
   time played, influences the probability of another goal being scored" [SM,
   Ch.1]. Empirically he uses a constant per-minute hazard (≈ 2.79 goals / 90 min
   = 0.031 per minute).

2. **Each team has a scoring rate and a conceding rate, split home/away.** The
   modelling recipe is explicit: "calculate a scoring rate and a conceding rate
   for each team and then simulate matches between them" — e.g. Arsenal 2012/13
   scored 2.47 at home / 1.32 away and conceded 1.21 at home / 0.74 away [SM,
   Ch.1]. These four numbers per team **are** attack strength, defence strength,
   and home advantage.

3. **Outcome probabilities come from simulating/expanding the score
   distribution.** Sumpter runs the season "10,000 times" and summarises how
   often each outcome occurs. The exact-score distribution is the primitive;
   1X2 and tournament-win probabilities are *derived* from it.

4. **Expected goals (xG)** is the quality-of-chances signal that explains results
   better than the scoreline alone [SM, later chapters]. It is the natural L3
   feature that should *feed* the rate parameters, not replace them.

### Mapping → `src/wc_bot/models/dixon_coles.py`

| Theory [SM] | Required implementation | Status |
|-------------|------------------------|--------|
| Memoryless goal process ⇒ Poisson | `score_matrix()` builds `Poisson(λ) ⊗ Poisson(μ)` | ✅ |
| Per-team scoring/conceding rates | `attack_` / `defense_` vectors, `expected_goals()` returns (λ, μ) | ✅ |
| Home/away split | `home_adv_` term added to λ only, dropped when `neutral=True` | ✅ |
| Outcome probs derived from score distribution | `matrix_to_1x2()` collapses the exact-score matrix | ✅ |
| Pure Poisson under-counts low scores | Dixon-Coles `rho` correction on the 0-0/1-0/0-1/1-1 cells | ✅ |
| xG as input signal | **TODO**: allow `FeatureStore` xG aggregates to inform priors on λ, μ | ⬜ |

**Architectural requirements (MUST):**

- **R-M1.** The model MUST remain *generative over exact scores*. 1X2, double
  chance, over/under, and "to win the tournament" probabilities MUST all be
  collapsed from a single `score_matrix()` — never modelled as independent
  heads. This is the [SM] "simulate then summarise" principle and guarantees
  internal coherence across the many World Cup market types.
- **R-M2.** The model MUST keep parameters minimal and the link function linear
  in log-space (`log λ = c + home + attack_i + defence_j`). This is mandated by
  [CHAN] (see Part III, data-snooping): "make the model as simple as possible,
  with as few parameters as possible… nonlinear models are more susceptible to
  data-snooping bias." The L2 ridge penalty on `attack_`/`defense_` is the
  concrete enforcement and also shrinks thin-sample national teams.
- **R-M3.** World Cup structure MUST be respected: group-stage matches permit
  draws and "dead rubbers"; knockouts cannot end level. The 1X2 collapse is for
  group stage; knockout markets MUST use the draw mass redistributed via
  extra-time/penalty logic, not the raw `draw` cell.

---

## Part II — Data Hygiene & Backtesting → `FeatureStore` (+ CV harness)

### Principles extracted from *Advances in Financial ML* [AFML] and *Algorithmic Trading* [CHAN]

1. **Look-ahead bias is the cardinal sin.** [CHAN, Ch.1]: look-ahead bias "means
   that your backtest program is using tomorrow's prices to determine today's
   trading signals… using future information to make a 'prediction' at the
   current time." His prescribed structural defence is decisive: **if the
   backtest and live programs are the same code, differing only in the data fed
   in, "there can be no look-ahead bias."**

2. **Purging.** [AFML, Ch.7, Snippet 7.1 `getTrainTimes`]: when labels span a
   time interval `[t0, t1]`, any training observation whose label interval
   *overlaps* a test label interval leaks information and MUST be dropped from
   the training set (train-starts-within-test, train-ends-within-test, and
   train-envelops-test cases).

3. **Embargoing.** [AFML, Ch.7, Snippet 7.2 `getEmbargoTimes`]: because of serial
   correlation, observations immediately *after* the test set must also be
   removed — an embargo of `pctEmbargo` of the sample — so post-test training
   bars cannot bleed test information backward.

4. **Standard K-Fold is prohibited.** [AFML, Ch.7, Snippet 7.3 `PurgedKFold`]:
   the CV class is built with `shuffle=False`, contiguous test folds, and a
   purged+embargoed train set precisely because plain K-Fold leaks under
   overlapping/serially-correlated labels. Model selection MUST use
   `PurgedKFold` (or Combinatorial Purged CV, [AFML, Ch.12]).

5. **Observations are not IID; weight by uniqueness.** [AFML, Ch.4]: overlapping
   labels share information, so samples MUST be weighted by their average
   uniqueness rather than treated as independent.

6. **Data-snooping bias.** [CHAN, Ch.1]: too many free parameters fit "random
   ethereal market patterns"; tweaking a model until it passes the out-of-sample
   set "turn[s] the out-of-sample data into in-sample data." Prefer few
   parameters and linearity.

### Mapping → `src/wc_bot/features/store.py`

| Theory | Required implementation | Status |
|--------|------------------------|--------|
| No look-ahead [CHAN] | `training_frame(as_of)` returns strictly `date < as_of` | ✅ |
| Same code paper/live [CHAN] | pipeline is identical for `--mock` and live; only the client/ledger differ | ✅ |
| Purging [AFML 7.1] | **TODO**: emit a `t1` (label start/end) per observation and a purged split | ⬜ |
| Embargo [AFML 7.2] | **TODO**: `pctEmbargo` parameter in the CV splitter | ⬜ |
| No standard K-Fold [AFML 7.3] | **TODO**: `purged_kfold(as_of, n_splits, pct_embargo)` generator | ⬜ |
| Uniqueness weights [AFML 4] | **TODO**: combine with the model's time-decay weights | ⬜ |

**Architectural requirements (MUST):**

- **R-D1 (point-in-time).** Every retrieval that targets a moment MUST use a
  strict inequality (`date < as_of`). The match being predicted, and everything
  after it, MUST be invisible. (Implemented; do not regress.)
- **R-D2 (label intervals).** A sports observation's "label" resolves at
  `match_time`, but features built from rolling form/xG windows have an
  information span. `FeatureStore` MUST expose, per observation, an interval
  `(info_start, resolve_time)` so the CV layer can purge overlaps per [AFML 7.1].
- **R-D3 (purged, embargoed CV only).** Any cross-validation used for model
  selection, hyper-parameter tuning, or `rho`/`half_life` calibration MUST use a
  purged + embargoed splitter. `sklearn.KFold`/`cross_val_score` with the default
  shuffle are **banned** for time-series data here [AFML 7.3].
- **R-D4 (decoupling).** `FeatureStore` MUST NOT import any model and MUST NOT
  contain Dixon-Coles/Elo/Poisson math. It produces structured, time-stamped
  frames; models own all statistics. (Implemented; preserve this boundary.)
- **R-D5 (weighting).** Training weights MUST reflect both recency (time decay,
  justified by regime-shift risk in [CHAN]) and uniqueness ([AFML 4]) once
  overlapping-window features exist.

---

## Part III — Pricing, Vig Removal, True Edge & Execution → `decision.py`

### Principles extracted from *The Logic of Sports Betting* [LSB] and *Algorithmic Trading* [CHAN]

1. **Break-even percentage = implied probability.** [LSB]: "Break-even percentage
   = Risk / (Risk + Win)"; e.g. -110 ⇒ 110/210 = 52.4%. This is the market's
   implied probability for an outcome.

2. **The hold (vig) is baked into prices.** [LSB]: a "standard" -110/-110 market
   has break-evens summing to 104.8%, i.e. a 4.5% hold (10/220). "All sportsbook
   markets have a positive hold." The raw implied probabilities are inflated and
   sum to >100%.

3. **You MUST remove the hold before judging a bet.** The true probability is
   obtained by normalising the break-even percentages to sum to 1 ("convert all
   the bets to their break-even percentages" then de-vig) [LSB]. Comparing your
   model to the *raw* price systematically overstates your edge by the hold.

4. **Edge = your probability vs the no-vig price; confirm with CLV.** [LSB]
   introduces Closing Line Value as the key diagnostic: compare the break-even
   you got to the break-even at market close. A long-run "average closing line
   value of at least half the hold" indicates a genuine, winning process. Market
   *resistance* (your price worse than close) is "a big red flag. Stop betting."

5. **Transaction costs are not optional in a backtest.** [CHAN, Ch.1]
   explicitly warns that he omitted transaction costs from example code but they
   "are crucial for a meaningful backtest." EV MUST be computed net of fees and
   slippage.

6. **Size with (fractional) Kelly.** [CHAN] frames position sizing via the Kelly
   formula for optimal leverage, tempered: "it is always better to be
   underleveraged than overleveraged." ⇒ fractional Kelly, never full.

### Mapping → `src/wc_bot/decision.py`

| Theory | Required implementation | Status |
|--------|------------------------|--------|
| Break-even = implied prob [LSB] | order-book mid / quote treated as implied prob | ✅ |
| Remove the hold first [LSB] | `remove_vig()` normalises outcomes to sum to 1 | ✅ |
| Edge vs **no-vig** price [LSB] | `edge = model_prob − fair_market_prob` (fair = de-vigged) | ✅ |
| EV net of costs [CHAN] | **TODO**: explicit fee/slippage term in `ev_per_dollar` | ⬜ |
| CLV diagnostic [LSB] | **TODO**: record entry break-even & closing break-even in ledger | ⬜ |
| Fractional Kelly [CHAN] | `kelly_fraction()` × `config.kelly_fraction`, capped | ✅ |

**Architectural requirements (MUST):**

- **R-E1 (de-vig before EV).** `remove_vig()` MUST be applied to the full set of
  an event's outcome prices *before* any edge or EV calculation. Computing EV
  against a raw quote is prohibited [LSB]. (Implemented; do not bypass.)
- **R-E2 (edge buffer ≥ friction).** `target_edge` MUST be set relative to the
  measured market hold and transaction costs, encoding the [LSB] standard that a
  bet should clear roughly half the hold in CLV terms. A configurable buffer that
  ignores the actual hold is non-compliant.
- **R-E3 (costs in EV).** `ev_per_dollar` MUST be net of taker fees, gas, and
  expected slippage from order-book depth [CHAN]. The current `target_edge`
  proxy MUST be upgraded to an explicit cost model.
- **R-E4 (fractional Kelly + hard cap).** Position size MUST use fractional Kelly
  with a per-bet bankroll cap. Full Kelly is banned [CHAN]. (Implemented.)
- **R-E5 (CLV logging).** The ledger MUST persist entry break-even and (at
  settlement) closing break-even so the process can be validated by CLV over
  hundreds of bets [LSB], independent of realised PnL variance.

---

## Part IV — Cross-cutting architecture

- **Unified backtest/live code path [CHAN].** The pipeline that paper-trades is
  the same one that will trade live; the *only* permitted difference is the data
  source and the L7 sink (CSV ledger vs signed CLOB order). This is our primary
  structural defence against look-ahead bias and implementation drift.
- **Strict settlement hygiene [AFML/CHAN].** Settlement MUST use an explicit,
  point-in-time `match_time`; guessing conclusion times is banned (enforced in
  `scripts/settle_ledger.py`, which raises on a missing `match_time`).
- **L2 RAG layer [FG].** Unstructured signals (injuries, lineups, news,
  sentiment) per *FinGPT* feed the L3 feature store as *timestamped* inputs only;
  any news item MUST carry the time it became known, or it violates R-D1.

## Compliance checklist (see `.cursorrules` for the enforced rules)

- [ ] Point-in-time retrieval everywhere (`date < as_of`).
- [ ] Purged + embargoed CV; no default K-Fold.
- [ ] De-vig before every EV/edge calculation.
- [ ] EV net of fees + slippage.
- [ ] Fractional Kelly with hard cap.
- [ ] Generative exact-score model; markets derived, not independently modelled.
- [ ] `FeatureStore` free of model logic.
- [ ] Backtest and live share one code path.
