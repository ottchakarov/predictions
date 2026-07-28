"""L8 - Visualization Layer: real-time paper-trading dashboard (Streamlit).

Reads the L7 paper ledger (``data/paper_trades.csv``) and renders KPIs, a
cumulative-bankroll curve, a model-vs-market edge scatter, and a recent-trades
table. The odds display (implied %, decimal, American) is toggleable in the
sidebar and applied consistently to the table and chart tooltips.

Run::

    streamlit run dashboard/app.py
"""
from __future__ import annotations

import math
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = ROOT / "data" / "paper_trades.csv"

FMT_IMPLIED = "Implied Probability (%)"
FMT_DECIMAL = "Decimal Odds"
FMT_AMERICAN = "American Odds"
ODDS_FORMATS = (FMT_IMPLIED, FMT_DECIMAL, FMT_AMERICAN)

# Numeric columns in the ledger (everything else is text / ids).
_NUMERIC_COLS = (
    "model_prob", "market_price", "fair_market_prob", "edge", "ev_per_dollar",
    "kelly_fraction", "sharp_multiplier", "net_sharp_alignment", "stake",
    "bankroll", "resolution_price", "pnl",
)


# ============================================================ odds conversions
# All inputs are implied probabilities in (0, 1) — exactly how model_prob and
# market_price are stored in the ledger. These are pure and side-effect free.
def to_decimal(prob: float) -> float:
    """Implied probability -> decimal odds (payout multiple). 0.25 -> 4.0."""
    if prob is None or not (0.0 < prob < 1.0):
        return math.nan
    return 1.0 / prob


def to_american(prob: float) -> float:
    """Implied probability -> American (moneyline) odds. 0.25 -> +300, 0.80 -> -400."""
    if prob is None or not (0.0 < prob < 1.0):
        return math.nan
    dec = 1.0 / prob
    return (dec - 1.0) * 100.0 if dec >= 2.0 else -100.0 / (dec - 1.0)


def format_odds(prob: float, fmt: str) -> str:
    """Render an implied probability in the chosen display format."""
    if prob is None or (isinstance(prob, float) and math.isnan(prob)):
        return "—"
    if fmt == FMT_DECIMAL:
        return f"{to_decimal(prob):.2f}"
    if fmt == FMT_AMERICAN:
        return f"{to_american(prob):+.0f}"
    return f"{prob * 100:.1f}%"


# ==================================================================== data load
def load_ledger(path: Path) -> pd.DataFrame:
    """Load the paper ledger fresh (no cache, for real-time view).

    Returns an EMPTY DataFrame (with the right columns where possible) if the file
    is missing or has no data rows, so the UI can render a clean empty state.
    """
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError):
        return pd.DataFrame()
    if df.empty:
        return df

    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.sort_values("timestamp")
    # Executed vs vetoed (stake == 0 is the sharp-veto / no-trade terminal row).
    df["is_executed"] = df.get("stake", 0).fillna(0) > 0
    df["trade_state"] = df["is_executed"].map({True: "Executed", False: "Vetoed"})
    return df.reset_index(drop=True)


# ====================================================================== layout
st.set_page_config(
    page_title="WC2026 Quant Execution Dashboard",
    page_icon="⚽",
    layout="wide",
)

st.title("FIFA World Cup 2026 — Round of 32: Quant Execution Dashboard")
st.caption(
    "L8 monitoring of the paper-trading bot — model probabilities vs live "
    "Polymarket prices, edge, and PnL. Read-only; no orders are signed here."
)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Display")
    odds_fmt = st.radio("Odds format", ODDS_FORMATS, index=0)

    st.divider()
    st.header("Refresh")
    if st.button("🔄 Reload data", use_container_width=True):
        st.rerun()
    # Optional live auto-refresh (only if streamlit-autorefresh is installed).
    try:
        from streamlit_autorefresh import st_autorefresh

        auto = st.checkbox("Auto-refresh", value=False)
        if auto:
            secs = st.slider("Interval (s)", 2, 60, 5)
            st_autorefresh(interval=secs * 1000, key="ledger_autorefresh")
    except ImportError:
        st.caption(
            "Tip: `pip install streamlit-autorefresh` to enable live "
            "auto-refresh; otherwise use the Reload button."
        )

    st.divider()
    st.caption(f"Ledger: `{LEDGER_PATH.relative_to(ROOT)}`")

df = load_ledger(LEDGER_PATH)

# ---------------------------------------------------------------- empty state
if df.empty:
    st.info(
        "No paper trades logged yet. Run the bot to populate the ledger:\n\n"
        "```\npython scripts/run_skeleton.py --live-odds --model dixon_coles\n```",
        icon="📭",
    )
    st.stop()

# ---------------------------------------------------------------- KPIs
realized_pnl = df["pnl"].fillna(0).sum() if "pnl" in df else 0.0
n_executed = int(df["is_executed"].sum())
n_vetoed = int((~df["is_executed"]).sum())
avg_edge = df["edge"].mean() if "edge" in df else float("nan")
settled = int((df.get("status") == "SETTLED").sum()) if "status" in df else 0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Realized PnL", f"${realized_pnl:,.2f}", help=f"{settled} settled trade(s)")
k2.metric("Trades Executed", f"{n_executed:,}")
k3.metric("Trades Vetoed", f"{n_vetoed:,}", help="stake == 0.0 (e.g. sharp veto)")
k4.metric(
    "Average Edge",
    "—" if math.isnan(avg_edge) else f"{avg_edge * 100:.2f}%",
    help="mean(model_prob - fair_market_prob) across logged signals",
)

st.divider()

# ---------------------------------------------------------------- charts
left, right = st.columns(2)

with left:
    st.subheader("Cumulative bankroll")
    start_bankroll = float(df["bankroll"].dropna().iloc[0]) if "bankroll" in df and df["bankroll"].notna().any() else 0.0
    curve = df.copy()
    curve["cum_bankroll"] = start_bankroll + curve["pnl"].fillna(0).cumsum()
    if curve["timestamp"].notna().any():
        line = (
            alt.Chart(curve)
            .mark_line(point=True)
            .encode(
                x=alt.X("timestamp:T", title="Time"),
                y=alt.Y("cum_bankroll:Q", title="Bankroll ($)", scale=alt.Scale(zero=False)),
                tooltip=[
                    alt.Tooltip("timestamp:T", title="Time"),
                    alt.Tooltip("match:N", title="Match"),
                    alt.Tooltip("cum_bankroll:Q", title="Bankroll", format="$,.2f"),
                    alt.Tooltip("pnl:Q", title="Trade PnL", format="$,.2f"),
                ],
            )
            .properties(height=340)
        )
        st.altair_chart(line, use_container_width=True)
    if realized_pnl == 0:
        st.caption(
            "Bankroll is flat until trades settle (open trades have no realized "
            "PnL yet). Run `scripts/settle_ledger.py` after matches conclude."
        )

with right:
    st.subheader("Model vs market (edge)")
    scat = df.copy()
    # Pre-format odds strings so tooltips honour the sidebar toggle.
    scat["model_odds"] = scat["model_prob"].map(lambda p: format_odds(p, odds_fmt))
    scat["market_odds"] = scat["market_price"].map(lambda p: format_odds(p, odds_fmt))
    scat["edge_pct"] = scat["edge"] * 100

    points = (
        alt.Chart(scat)
        .mark_circle(size=140, opacity=0.8)
        .encode(
            x=alt.X("market_price:Q", title="Market implied prob", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("model_prob:Q", title="Model prob", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "trade_state:N",
                title="State",
                scale=alt.Scale(domain=["Executed", "Vetoed"], range=["#2ca02c", "#d62728"]),
            ),
            tooltip=[
                alt.Tooltip("match:N", title="Match"),
                alt.Tooltip("outcome:N", title="Outcome"),
                alt.Tooltip("model_odds:N", title=f"Model ({odds_fmt})"),
                alt.Tooltip("market_odds:N", title=f"Market ({odds_fmt})"),
                alt.Tooltip("edge_pct:Q", title="Edge", format="+.2f"),
                alt.Tooltip("trade_state:N", title="State"),
            ],
        )
    )
    # y = x reference: above the line = model sees value vs the market.
    diag = (
        alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
        .mark_line(strokeDash=[4, 4], color="gray")
        .encode(x="x:Q", y="y:Q")
    )
    st.altair_chart((diag + points).properties(height=340), use_container_width=True)

st.divider()

# ---------------------------------------------------------------- recent trades
st.subheader("Recent trades")
recent = df.sort_values("timestamp", ascending=False).head(20).copy()
recent["Model"] = recent["model_prob"].map(lambda p: format_odds(p, odds_fmt))
recent["Market"] = recent["market_price"].map(lambda p: format_odds(p, odds_fmt))
recent["Edge"] = (recent["edge"] * 100).map(lambda x: f"{x:+.2f}%")
recent["Stake"] = recent["stake"].map(lambda x: f"${x:,.2f}")
recent["PnL"] = recent["pnl"].map(lambda x: "—" if pd.isna(x) else f"${x:,.2f}")

display_cols = {
    "timestamp": "Time",
    "match": "Match",
    "outcome": "Outcome",
    "model_version": "Model",
    "Model": f"Model ({odds_fmt})",
    "Market": f"Market ({odds_fmt})",
    "Edge": "Edge",
    "Stake": "Stake",
    "trade_state": "State",
    "status": "Status",
    "PnL": "PnL",
}
present = {k: v for k, v in display_cols.items() if k in recent.columns}
table = recent[list(present)].rename(columns=present)
st.dataframe(table, use_container_width=True, hide_index=True)

st.caption(
    f"Showing {len(recent)} of {len(df):,} logged signals · odds shown as "
    f"**{odds_fmt}** · edge always in percentage points."
)
