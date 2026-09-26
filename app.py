"""
XAU Monitor — gold (XAUUSD) intraday dashboard.  Phase 0: data, session clock, chart.

Chart and levels are in COMEX GC=F futures prices. Broker XAUUSD spot =
futures minus the basis entered in the sidebar (fed live from MT5 at Phase 8).
Every panel is isolated in try/except so one failure never white-screens the app.
"""
from datetime import datetime, timedelta, timezone

import streamlit as st

st.set_page_config(page_title="XAU Monitor", page_icon="🟡", layout="wide")

import pandas as pd
import plotly.graph_objects as go

import xau_config as cfg
import xau_data as xd
import xau_sessions as xs


def panel(name, fn, *args, **kwargs):
    """Run a panel; show a readable error instead of crashing the page."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:  # noqa: BLE001
        st.warning(f"{name} unavailable — {type(e).__name__}: {e}")
        return None


def fmt(x, nd=2, prefix=""):
    return "—" if x is None else f"{prefix}{x:,.{nd}f}"


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    basis = st.number_input(
        "Basis: GC=F minus broker XAUUSD ($)", value=float(cfg.DEFAULT_BASIS),
        step=0.5, format="%.2f",
        help="Check your MT5 XAUUSD price against GC=F and enter the difference. "
             "Goes live from MT5 at Phase 8.")
    days_shown = st.slider("Trading days on chart", 1, 10, 3)
    show_bands = st.checkbox("Session bands", value=True)
    show_rounds = st.checkbox("Round-number levels", value=True)
    if st.button("Refresh data"):
        st.cache_data.clear()
    st.caption(f"Signal timeframe {cfg.SIGNAL_INTERVAL} · cache {cfg.CACHE_TTL_SEC}s · "
               f"times in EAT ({cfg.DISPLAY_TZ})")

now = datetime.now(timezone.utc)
bundle = panel("Data", xd.fetch_bundle) or {
    "gold": xd._empty(), "primary": cfg.PRIMARY, "cross": {}, "errors": ["data panel failed"],
    "fetched_at": now}
gold = bundle["gold"]


# ── Header ───────────────────────────────────────────────────────────────────
def render_header():
    status = xs.market_status(now)
    label = xs.session_label(now)
    nxt = xs.next_event(now)
    now_eat = now.astimezone(xs.DISP)
    st.title("XAU Monitor")
    cols = st.columns(4)
    cols[0].metric("EAT time", now_eat.strftime("%a %H:%M"))
    cols[1].metric("Market", status["state"])
    cols[2].metric("Session", label)
    if nxt:
        mins = int((nxt["start"] - now_eat).total_seconds() // 60)
        cols[3].metric("Next event", nxt["label"],
                       f"in {mins // 60}h {mins % 60:02d}m · {nxt['start']:%H:%M}",
                       delta_color="off")
    if bundle["errors"]:
        with st.expander(f"Data notes ({len(bundle['errors'])})"):
            for e in bundle["errors"]:
                st.write("•", e)


panel("Header", render_header)


# ── Price metrics ────────────────────────────────────────────────────────────
def render_metrics():
    if gold.empty:
        st.error("No gold data. Try Refresh; yfinance may be rate-limiting.")
        return
    ch = xd.change_vs_prev_close(gold)
    td = xs.trading_day(gold.index[-1].to_pydatetime())
    asia = xs.asian_range(gold, td)
    pdl = xs.prev_day_levels(gold, td)
    age = xd.bar_age_minutes(gold, now)

    c = st.columns(6)
    c[0].metric(f"{bundle['primary']} last", fmt(ch["last"]),
                None if ch["change"] is None else f"{ch['change']:+,.2f} ({ch['pct']:+.2f}%)")
    c[1].metric("XAUUSD spot est.", fmt(xd.to_spot(ch["last"], basis)),
                f"basis {basis:+.2f}", delta_color="off")
    c[2].metric("Asian high", fmt(asia["high"]) if asia else "—")
    c[3].metric("Asian low", fmt(asia["low"]) if asia else "—")
    c[4].metric("Prev day high", fmt(pdl["high"]) if pdl else "—")
    c[5].metric("Prev day low", fmt(pdl["low"]) if pdl else "—")

    if age is not None and xs.market_status(now)["open"] and age > cfg.STALE_AFTER_MIN:
        st.warning(f"Last bar is {age:.0f} min old — yfinance feed may be delayed.")


panel("Price metrics", render_metrics)


# ── Cross-asset strip ────────────────────────────────────────────────────────
def render_cross():
    st.subheader("Cross-asset")
    cols = st.columns(len(cfg.CROSS_ASSETS))
    for col, (t, name) in zip(cols, cfg.CROSS_ASSETS.items()):
        ch = xd.change_vs_prev_close(bundle["cross"].get(t))
        if ch is None:
            col.metric(name, "—")
            continue
        if ch["change"] is None:
            delta = None
        elif t in cfg.YIELD_TICKERS:
            delta = f"{ch['change'] * 100:+.1f} bp"
        else:
            delta = f"{ch['pct']:+.2f}%"
        col.metric(name, fmt(ch["last"], 3 if t in ("JPY=X",) or t in cfg.YIELD_TICKERS else 2),
                   delta)
    st.caption("Change vs previous CME trading-day close. Phase 3 turns these into the "
               "cross-asset agreement score.")


panel("Cross-asset", render_cross)


# ── Session clock ────────────────────────────────────────────────────────────
def render_clock():
    st.subheader("Today's session clock (EAT)")
    now_eat = now.astimezone(xs.DISP)
    rows = []
    for ev in xs.schedule_for_display_date(now_eat.date()):
        if ev["end"] and ev["start"] <= now_eat < ev["end"]:
            state = "now"
        elif ev["start"] <= now_eat:
            state = "done"
        else:
            state = "upcoming"
        when = ev["start"].strftime("%H:%M")
        if ev["end"]:
            when += "–" + ev["end"].strftime("%H:%M")
        rows.append({"Time": when, "Event": ev["label"], "Type": ev["kind"],
                     "Status": state, "Note": ev["note"]})
    if not rows:
        st.info("No sessions today (weekend).")
        return
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Built from London / New York / Shanghai local times, so it shifts "
               "automatically when UK (Oct 25) and US (Nov 1) clocks change.")


# ── Chart ────────────────────────────────────────────────────────────────────
def _gap_breaks(idx_local: pd.DatetimeIndex, step=pd.Timedelta(minutes=15)):
    """Missing 15m slots inside gaps > 30 min (weekends, daily break)."""
    vals = []
    for a, b in zip(idx_local[:-1], idx_local[1:]):
        if b - a > pd.Timedelta(minutes=30):
            t = a + step
            while t < b:
                vals.append(t)
                t += step
    return vals


def render_chart():
    if gold.empty:
        return
    days = xs.trading_day_index(gold.index)
    keep = sorted(set(days))[-days_shown:]
    df = gold[pd.Index(days).isin(keep)]
    if df.empty:
        return
    td = keep[-1]
    x = df.index.tz_convert(cfg.DISPLAY_TZ).tz_localize(None)

    fig = go.Figure(go.Candlestick(
        x=x, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name=bundle["primary"], increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350"))

    if show_bands:
        for w in xs.session_windows(df.index[0].to_pydatetime(),
                                    df.index[-1].to_pydatetime() + timedelta(minutes=15)):
            fig.add_vrect(
                x0=pd.Timestamp(w["start"]).tz_convert(cfg.DISPLAY_TZ).tz_localize(None),
                x1=pd.Timestamp(w["end"]).tz_convert(cfg.DISPLAY_TZ).tz_localize(None),
                fillcolor=cfg.BAND_COLORS[w["name"]], line_width=0, layer="below")

    def hline(y, text, color, dash):
        fig.add_hline(y=y, line_color=color, line_dash=dash, line_width=1,
                      annotation_text=text, annotation_position="right",
                      annotation_font_size=10, annotation_font_color=color)

    asia = xs.asian_range(gold, td)
    if asia:
        hline(asia["high"], "Asia H", "#9b59b6", "dot")
        hline(asia["low"], "Asia L", "#9b59b6", "dot")
    pdl = xs.prev_day_levels(gold, td)
    if pdl:
        hline(pdl["high"], "PDH", "#e67e22", "dash")
        hline(pdl["low"], "PDL", "#e67e22", "dash")
    if show_rounds:
        lo, hi = float(df["Low"].min()), float(df["High"].max())
        for lvl in xs.round_levels(float(df["Close"].iloc[-1])):
            if lo <= lvl <= hi:
                fig.add_hline(y=lvl, line_color="rgba(150,150,150,0.35)", line_width=1)

    fig.update_layout(
        height=560, margin=dict(l=10, r=60, t=30, b=10), xaxis_rangeslider_visible=False,
        showlegend=False, title=f"{bundle['primary']} {cfg.SIGNAL_INTERVAL} · EAT",
        yaxis_title="GC=F ($)")
    breaks = _gap_breaks(x)
    if breaks:
        fig.update_xaxes(rangebreaks=[dict(values=breaks, dvalue=15 * 60 * 1000)])
    st.plotly_chart(fig, width="stretch")
    st.caption("Bands: blue = London, green = London–NY overlap, amber = New York. "
               "Levels are futures prices; subtract the basis for broker spot.")


left, right = st.columns([3, 2])
with left:
    panel("Chart", render_chart)
with right:
    panel("Session clock", render_clock)
