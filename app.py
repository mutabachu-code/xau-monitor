"""
XAU Monitor — gold (XAUUSD) intraday dashboard.
Phase 0: data, session clock, chart.  Phase 1: L5 technicals, L6 liquidity.
Phase 2: L1 rates, L2 dollar (correlation-weighted).
Phase 3: L3 cross-asset agreement (implied move, structural bid, silver, metals).
Phase 4: L8 regime classifier (which driver is in control, trade style).

Chart and levels are in COMEX GC=F futures prices. Broker XAUUSD spot =
futures minus the basis entered in the sidebar (fed live from MT5 at Phase 8).
Every panel is isolated in try/except so one failure never white-screens the app.
"""
from datetime import datetime, timedelta, timezone

import streamlit as st

st.set_page_config(page_title="XAU Monitor", page_icon="🟡", layout="wide")

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import xau_config as cfg
import xau_crossasset as xc
import xau_data as xd
import xau_dollar as xdl
import xau_liquidity as xl
import xau_rates as xr
import xau_regime as xg
import xau_sessions as xs
import xau_technicals as xt
import xau_macro as xm
import xau_runtime as xrt

# Reload project modules if a git push changed them (Streamlit Cloud can keep
# stale copies in memory), dependency order: config first, app-level last.
_reloaded = xrt.ensure_fresh([cfg, xs, xd, xt, xl, xm, xr, xdl, xc, xg])
REQUIRED_CONFIG_VERSION = 4
if getattr(cfg, "CONFIG_VERSION", 0) < REQUIRED_CONFIG_VERSION:
    st.error(f"xau_config.py on the server is older than app.py expects "
             f"(version {getattr(cfg, 'CONFIG_VERSION', 'none')} < "
             f"{REQUIRED_CONFIG_VERSION}). Push the latest xau_config.py, then "
             "Manage app → Reboot.")
    st.stop()


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
    st.subheader("Overlays")
    show_vwap = st.checkbox("VWAP + σ bands", value=True)
    show_emas = st.checkbox("EMA 9 / 21 / 50", value=True)
    show_cpr = st.checkbox("CPR + R1/S1", value=True)
    show_sweeps = st.checkbox("Liquidity sweeps", value=True)
    show_rsi = st.checkbox("RSI pane", value=True)
    if st.button("Refresh data"):
        st.cache_data.clear()
    st.caption(f"Signal timeframe {cfg.SIGNAL_INTERVAL} · cache {cfg.CACHE_TTL_SEC}s · "
               f"times in EAT ({cfg.DISPLAY_TZ})")

now = datetime.now(timezone.utc)
bundle = panel("Data", xd.fetch_bundle) or {
    "gold": xd._empty(), "primary": cfg.PRIMARY, "cross": {}, "errors": ["data panel failed"],
    "fetched_at": now}
gold = bundle["gold"]
tech = xt.get_tech_report(gold)
liq = xl.get_liq_report(gold)
fred = xr.load_fred()
rates = xr.get_rates_report(bundle, fred)
dollar = xdl.get_dollar_report(bundle)
xasset = xc.get_xasset_report(bundle)
regime = xg.get_regime_report(bundle, tech, rates, dollar, xasset)


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
regime_slot = st.empty()          # filled once the regime helpers are defined


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
    st.caption("Change vs previous CME trading-day close. DXY, yields and USD/JPY feed "
               "L1/L2 below; silver, copper and oil feed L3.")


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


def _local(ts):
    return pd.Timestamp(ts).tz_convert(cfg.DISPLAY_TZ).tz_localize(None)


def render_chart():
    if gold.empty:
        return
    days = xs.trading_day_index(gold.index)
    keep = sorted(set(days))[-days_shown:]
    mask = pd.Index(days).isin(keep)
    df = gold[mask]
    if df.empty:
        return
    td = keep[-1]
    x = df.index.tz_convert(cfg.DISPLAY_TZ).tz_localize(None)
    fr = tech.frame[mask] if (tech.ok and tech.frame is not None) else None

    rows = 2 if (show_rsi and fr is not None) else 1
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.78, 0.22] if rows == 2 else [1.0])
    fig.add_trace(go.Candlestick(
        x=x, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name=bundle["primary"], increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350"), row=1, col=1)

    if show_bands:
        for w in xs.session_windows(df.index[0].to_pydatetime(),
                                    df.index[-1].to_pydatetime() + timedelta(minutes=15)):
            fig.add_vrect(x0=_local(w["start"]), x1=_local(w["end"]),
                          fillcolor=cfg.BAND_COLORS[w["name"]], line_width=0,
                          layer="below", row="all", col=1)

    if fr is not None and show_vwap:
        fig.add_trace(go.Scatter(x=x, y=fr["vwap"], name="VWAP", mode="lines",
                                 line=dict(color="#f1c40f", width=1.6)), row=1, col=1)
        for k, dash in ((1, "dot"), (2, "dash")):
            for sgn in (1, -1):
                fig.add_trace(go.Scatter(
                    x=x, y=fr["vwap"] + sgn * k * fr["sd"], mode="lines",
                    name=f"VWAP {'+' if sgn > 0 else '−'}{k}σ", showlegend=False,
                    line=dict(color="rgba(241,196,15,0.45)", width=1, dash=dash),
                    hoverinfo="skip"), row=1, col=1)
    if fr is not None and show_emas:
        for col_, color in (("ema9", "#3498db"), ("ema21", "#9b59b6"), ("ema50", "#e74c3c")):
            fig.add_trace(go.Scatter(x=x, y=fr[col_], name=col_.upper(), mode="lines",
                                     line=dict(color=color, width=1)), row=1, col=1)

    x_today0 = _local(xs.asian_window(td)["start"])
    x_end = x[-1] + pd.Timedelta(minutes=15)

    def seg(y, text, color, dash="solid", width=1):
        fig.add_shape(type="line", x0=x_today0, x1=x_end, y0=y, y1=y,
                      line=dict(color=color, dash=dash, width=width), row=1, col=1)
        fig.add_annotation(x=x_end, y=y, text=text, showarrow=False, xanchor="left",
                           font=dict(size=10, color=color), row=1, col=1)

    if show_cpr and tech.ok and tech.cpr:
        c = tech.cpr
        fig.add_shape(type="rect", x0=x_today0, x1=x_end, y0=c["BC"], y1=c["TC"],
                      fillcolor="rgba(52,152,219,0.18)", line_width=0, row=1, col=1)
        seg(c["P"], "P", "#3498db", "dot")
        seg(c["R1"], "R1", "#95a5a6", "dash")
        seg(c["S1"], "S1", "#95a5a6", "dash")
        vp = c.get("virgin_prior")
        if vp:
            fig.add_shape(type="rect", x0=x_today0, x1=x_end, y0=vp["BC"], y1=vp["TC"],
                          fillcolor="rgba(0,0,0,0)", line=dict(color="#3498db", dash="dot"),
                          row=1, col=1)

    asia = xs.asian_range(gold, td)
    if asia:
        seg(asia["high"], "Asia H", "#9b59b6", "dot")
        seg(asia["low"], "Asia L", "#9b59b6", "dot")
    pdl = xs.prev_day_levels(gold, td)
    if pdl:
        seg(pdl["high"], "PDH", "#e67e22", "dash")
        seg(pdl["low"], "PDL", "#e67e22", "dash")
    if liq.ok:
        for lv in liq.levels:
            if lv.name in ("EQH", "EQL", "PWH", "PWL"):
                lo_, hi_ = float(df["Low"].min()), float(df["High"].max())
                if lo_ - 30 <= lv.price <= hi_ + 30:
                    seg(lv.price, lv.name, "#16a085", "dashdot")
    if show_rounds:
        lo, hi = float(df["Low"].min()), float(df["High"].max())
        for lvl in xs.round_levels(float(df["Close"].iloc[-1])):
            if lo <= lvl <= hi:
                fig.add_hline(y=lvl, line_color="rgba(150,150,150,0.35)", line_width=1,
                              row=1, col=1)

    if show_sweeps and liq.ok and liq.sweeps:
        for sw in liq.sweeps:
            bull = sw["dir"] == "bullish"
            live = sw["points"] != 0
            fig.add_trace(go.Scatter(
                x=[_local(sw["time"])], y=[sw["extreme"]], mode="markers",
                marker=dict(symbol="triangle-up" if bull else "triangle-down", size=13,
                            color=("#26a69a" if bull else "#ef5350") if live else "#7f8c8d",
                            line=dict(color="white", width=1)),
                name=f"{sw['level']} sweep",
                hovertext=f"{sw['level']} {sw['dir']} sweep · {sw['why']}",
                hoverinfo="text", showlegend=False), row=1, col=1)

    if rows == 2:
        fig.add_trace(go.Scatter(x=x, y=fr["rsi"], name="RSI", mode="lines",
                                 line=dict(color="#8e44ad", width=1.2)), row=2, col=1)
        for lvl, clr in ((70, "#ef5350"), (50, "#7f8c8d"), (30, "#26a69a")):
            fig.add_hline(y=lvl, line_color=clr, line_width=1, line_dash="dot", row=2, col=1)
        fig.update_yaxes(range=[0, 100], title_text="RSI", row=2, col=1)

    fig.update_layout(
        height=680 if rows == 2 else 560, margin=dict(l=10, r=70, t=30, b=10),
        xaxis_rangeslider_visible=False, showlegend=True,
        legend=dict(orientation="h", y=1.02, x=0, font=dict(size=10)),
        title=f"{bundle['primary']} {cfg.SIGNAL_INTERVAL} · EAT")
    fig.update_yaxes(title_text="GC=F ($)", row=1, col=1)
    breaks = _gap_breaks(x)
    if breaks:
        fig.update_xaxes(rangebreaks=[dict(values=breaks, dvalue=15 * 60 * 1000)])
    st.plotly_chart(fig, width="stretch")
    st.caption("Bands: blue = London, green = London–NY overlap, amber = New York. "
               "Shaded blue box = today's CPR. Triangles = liquidity sweeps (grey = "
               "expired or invalidated). Levels are futures prices; subtract the basis "
               "for broker spot.")


# ── Macro layers (L1 rates, L2 dollar) ───────────────────────────────────────
LINK_COLORS = {"Rates": "#2a78d6", "Dollar": "#eb6834"}   # validated categorical pair


def _fmt_corr(c):
    return "—" if c is None else f"{c:+.2f}"


def render_macro(rep, title):
    st.subheader(title)
    if not rep.ok:
        st.info(f"{rep.layer} layer unavailable: {rep.error}")
        return
    c = st.columns(4)
    c[0].metric(f"Score (±{rep.max_pts})", f"{rep.score:+.1f}", _bias_badge(rep.bias),
                delta_color="off")
    wt = "—" if rep.weight is None else f"{rep.weight * 100:.0f}%"
    c[1].metric("Layer weight", wt, f"max now ±{rep.eff_max:g}", delta_color="off")
    c[2].metric("Corr 20d", _fmt_corr(rep.corr_long), f"5d {_fmt_corr(rep.corr_short)}",
                delta_color="off",
                help=f"{rep.corr_source}; textbook sign is {rep.corr_expected}")
    c[3].metric("Raw signal", f"{rep.raw:+.2f}", "unweighted", delta_color="off")
    rows = [{"Input": k, "Signal (−1…+1)": None if v is None else round(v, 2),
             "Reading": rep.details.get(k, "")} for k, v in rep.components.items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"Correlation: {rep.corr_source}. Positive signal = bullish gold.")
    for n in rep.notes:
        st.write("•", n)


def render_link_chart():
    fig = go.Figure()
    shown = False
    for rep in (rates, dollar):
        s = rep.corr_series
        if not rep.ok or s is None or s.empty:
            continue
        orient = -1 if rep.corr_expected == "negative" else 1
        h = (orient * s).resample("1h").last().dropna()
        h.index = h.index.tz_convert(cfg.DISPLAY_TZ).tz_localize(None)
        color = LINK_COLORS[rep.layer]
        fig.add_trace(go.Scatter(x=h.index, y=h.values, mode="lines", name=rep.layer,
                                 line=dict(color=color, width=2),
                                 hovertemplate="%{x|%d %b %H:%M}<br>" + rep.layer +
                                 " link %{y:.2f}<extra></extra>"))
        shown = True
    if not shown:
        st.caption("Link-strength history needs more overlapping data.")
        return
    fig.add_hline(y=cfg.CORR_FULL, line_dash="dash", line_width=1,
                  line_color="rgba(128,128,128,0.7)",
                  annotation_text="full weight ≥ %.2f" % cfg.CORR_FULL,
                  annotation_position="bottom left",
                  annotation_font_size=10)
    fig.add_hline(y=0, line_width=1, line_color="rgba(128,128,128,0.5)")
    fig.update_layout(height=280, margin=dict(l=10, r=20, t=30, b=30),
                      title="How strongly gold is trading off yields and the dollar "
                            "(rolling 5-day, textbook direction = up)",
                      title_font_size=13, hovermode="x unified",
                      legend=dict(orientation="h", y=-0.2, x=0),
                      yaxis=dict(range=[-1, 1], title="Link strength",
                                 gridcolor="rgba(128,128,128,0.15)"),
                      xaxis=dict(showgrid=False))
    st.plotly_chart(fig, width="stretch")


def render_macro_readings():
    rd = rates.readings if rates.ok else {}
    bits = []
    if rd.get("tnx_level") is not None:
        bits.append(f"US 10y {rd['tnx_level']:.2f}%")
    if rd.get("real_10y") is not None:
        bits.append(f"10y real (TIPS) {rd['real_10y']:.2f}%")
    if rd.get("breakeven_10y") is not None:
        bits.append(f"10y breakeven {rd['breakeven_10y']:.2f}%")
    if bits:
        st.caption(" · ".join(bits) + " — real yield and breakeven are daily FRED data.")
    elif fred and all(v is None for v in fred.values()):
        st.caption("FRED real-yield data unavailable right now; the rates layer runs "
                   "on intraday Treasury futures only.")


# ── L3 cross-asset agreement ─────────────────────────────────────────────────
FLAG_TEXT = {
    "structural_bid": "Structural bid (G7)",
    "structural_offer": "Structural offer",
    "silver_nonconfirm_high": "Silver not confirming high (G3)",
    "silver_nonconfirm_low": "Silver not confirming low",
    "oil_spike_up": "Oil spike up (G8)",
    "oil_spike_down": "Oil spike down",
}


def render_xasset():
    st.subheader("L3 · Cross-asset agreement")
    if not xasset.ok:
        st.info(f"Cross-asset layer unavailable: {xasset.error}")
        return
    imp = xasset.implied
    c = st.columns(4)
    c[0].metric(f"Score (±{cfg.XASSET_MAX})", f"{xasset.score:+.1f}",
                _bias_badge(xasset.bias), delta_color="off")
    c[1].metric("Residual z", "—" if not imp else f"{imp['resid_z']:+.1f}",
                None if not imp else f"gap {imp['resid_pct']:+.2f}%",
                delta_color="off")
    c[2].metric("Move type", xasset.move_type)
    c[3].metric("Gold/silver ratio", "—" if xasset.gs_ratio is None else f"{xasset.gs_ratio:.1f}",
                None if xasset.gs_ratio_chg is None else f"{xasset.gs_ratio_chg:+.2f}% today",
                delta_color="off")
    pts = {"Residual": 8, "Structural": 5, "Silver": 4, "Metals context": 3}
    rows = [{"Component": k, "Points": v, "Max": pts[k], "Reading": xasset.details.get(k, "")}
            for k, v in xasset.components.items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    on = [FLAG_TEXT[k] for k, v in xasset.flags.items() if v and k in FLAG_TEXT]
    if on:
        st.write("**Flags for the master signal:** " + " · ".join(on))
    for n in xasset.notes:
        st.write("•", n)


def render_implied_chart():
    imp = xasset.implied if xasset.ok else None
    if not imp:
        st.caption("Implied-move chart appears once there are 20 days of overlapping "
                   "driver data.")
        return
    path = imp["path"]
    x = path.index.tz_convert(cfg.DISPLAY_TZ).tz_localize(None)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=path["actual"], mode="lines", name="Gold actual",
                             line=dict(color="#2a78d6", width=2),
                             hovertemplate="%{x|%a %H:%M}<br>actual %{y:+.2f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=path["implied"], mode="lines",
                             name="Implied by yields · USD · oil",
                             line=dict(color="#eb6834", width=2, dash="dash"),
                             hovertemplate="%{x|%a %H:%M}<br>implied %{y:+.2f}%<extra></extra>"))
    fig.add_hline(y=0, line_width=1, line_color="rgba(128,128,128,0.5)")
    fig.update_layout(height=300, margin=dict(l=10, r=20, t=40, b=30),
                      title=f"Gold vs what its drivers imply (last {imp['bars']} bars, "
                            "cumulative %)", title_font_size=13, hovermode="x unified",
                      legend=dict(orientation="h", y=-0.2, x=0),
                      yaxis=dict(title="% move", gridcolor="rgba(128,128,128,0.15)",
                                 ticksuffix="%"),
                      xaxis=dict(showgrid=False))
    st.plotly_chart(fig, width="stretch")
    m = xasset.model
    contrib = " · ".join(f"{k.replace('_bp', '').replace('dxy', 'USD')} "
                         f"{v:+.2f}%" for k, v in imp["contrib"].items())
    st.caption(f"Gap between the lines = residual. Driver contributions: {contrib}. "
               f"Model fit on {m['n']} bars from the prior {cfg.IMPLIED_TRAIN_DAYS} "
               f"trading days, R² {m['r2']:.2f}.")


# ── L8 regime ────────────────────────────────────────────────────────────────
REGIME_ICON = {"rates-led": "🏦", "flow-led": "🌊", "risk-off": "🚨", "trend": "📈",
               "chop": "↔️", "mixed": "❔"}


def render_regime_banner():
    if not regime.ok:
        return
    txt = (f"**{REGIME_ICON.get(regime.regime, '')} Regime: {regime.label}** · "
           f"strength {regime.strength:.0%} · style: *{regime.style}* · "
           f"L8 {regime.score:+.1f}/{cfg.REGIME_MAX}  \n{regime.playbook}")
    box = {"risk-off": st.error, "chop": st.warning, "mixed": st.warning}.get(
        regime.regime, st.info)
    box(txt)


def render_regime():
    st.subheader("L8 · Regime")
    if not regime.ok:
        st.info(f"Regime unavailable: {regime.error}")
        return
    c = st.columns(4)
    c[0].metric(f"Score (±{cfg.REGIME_MAX})", f"{regime.score:+.1f}",
                _bias_badge(regime.bias), delta_color="off")
    ranked = sorted(regime.candidates.items(), key=lambda kv: -kv[1])
    ru = ranked[1] if len(ranked) > 1 else None
    c[1].metric("Runner-up", "—" if not ru else f"{ru[1]:.0%}",
                None if not ru else ru[0], delta_color="off")
    c[2].metric("Strength", f"{regime.strength:.0%}")
    c[3].metric("Direction", f"{regime.direction:+.2f}")
    rows = [{"Candidate": cfg.REGIME_LABELS[k], "Strength": v, "Why": regime.why.get(k, "")}
            for k, v in sorted(regime.candidates.items(), key=lambda kv: -kv[1])]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    for n in regime.notes:
        st.write("•", n)


def render_regime_chart():
    if not regime.ok or not regime.candidates:
        return
    items = sorted(regime.candidates.items(), key=lambda kv: kv[1])
    names = [cfg.REGIME_LABELS[k] for k, _ in items]
    vals = [v for _, v in items]
    colors = ["#2a78d6" if k == regime.regime else "rgba(128,128,128,0.45)"
              for k, _ in items]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=colors,
                           text=[f"{v:.0%}" for v in vals], textposition="outside",
                           hovertemplate="%{y}: %{x:.0%}<extra></extra>"))
    fig.add_vline(x=cfg.REGIME_MIN_STRENGTH, line_dash="dash", line_width=1,
                  line_color="rgba(128,128,128,0.7)",
                  annotation_text=f"min {cfg.REGIME_MIN_STRENGTH:.0%}",
                  annotation_position="top", annotation_font_size=10)
    fig.update_layout(height=260, margin=dict(l=10, r=40, t=40, b=20),
                      title="Which driver is in control (candidate strength)",
                      title_font_size=13, showlegend=False,
                      xaxis=dict(range=[0, 1.15], tickformat=".0%",
                                 gridcolor="rgba(128,128,128,0.15)"),
                      yaxis=dict(showgrid=False), bargap=0.35)
    st.plotly_chart(fig, width="stretch")


# ── Layer panels ─────────────────────────────────────────────────────────────
def _bias_badge(bias):
    return {"LONG": "🟢 LONG", "SHORT": "🔴 SHORT"}.get(bias, "⚪ NEUTRAL")


def render_tech():
    st.subheader("L5 · Technicals")
    if not tech.ok:
        st.info(f"Technicals unavailable: {tech.error}")
        return
    c = st.columns(4)
    c[0].metric(f"Score (±{cfg.TECH_MAX})", f"{tech.score:+.1f}", _bias_badge(tech.bias),
                delta_color="off")
    c[1].metric("RSI 14", f"{tech.rsi:.1f}")
    c[2].metric("ADX", f"{tech.adx:.0f}", "chop" if tech.chop else "trending",
                delta_color="off")
    c[3].metric("ATR 15m", f"{tech.atr:.2f}", f"{tech.atr_regime} ({tech.atr_ratio:.1f}×)",
                delta_color="off")
    rows = [{"Component": k, "Points": v, "Max": m, "Reading": tech.details.get(k, "")}
            for k, v, m in ((k, tech.components[k], mx) for k, mx in
                            (("VWAP", 6), ("EMA stack", 5), ("1h trend", 4),
                             ("Structure", 5), ("RSI", 3), ("CPR", 2)))]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if tech.daily_atr:
        st.caption(f"Daily ATR (avg range, last 14 days): ${tech.daily_atr:,.2f}")
    for n in tech.notes:
        st.write("•", n)


def render_liq():
    st.subheader("L6 · Liquidity")
    if not liq.ok:
        st.info(f"Liquidity unavailable: {liq.error}")
        return
    c = st.columns(4)
    c[0].metric(f"Score (±{cfg.LIQ_MAX})", f"{liq.score:+.1f}", _bias_badge(liq.bias),
                delta_color="off")
    c[1].metric("RVOL now", f"{liq.rvol_now:.2f}×")
    c[2].metric("Asian range", liq.asian_state)
    act = liq.active
    c[3].metric("Active sweep", "—" if not act else f"{act['level']} ({act['dir']})",
                None if not act else f"{act['points']:+.1f} pts", delta_color="off")
    if act:
        st.caption(f"Active sweep scoring: {act['why']}")
    t = liq.targets
    tc = st.columns(2)
    for col, key, label in ((tc[0], "above", "Buy-side liquidity above"),
                            (tc[1], "below", "Sell-side liquidity below")):
        v = t.get(key)
        col.metric(label, "—" if not v else f"{v['name']} {v['price']:,.2f}",
                   None if not v else f"${v['dist']:,.2f} · {v['atr_mult']:.1f} ATR",
                   delta_color="off")
    if liq.sweeps:
        rows = [{"Time (EAT)": _local(s["time"]).strftime("%a %H:%M"), "Level": s["level"],
                 "Price": round(s["price"], 2), "Direction": s["dir"],
                 "RVOL": round(s["rvol"], 2), "Session": s["session"],
                 "Points": s["points"], "Status": s["why"]} for s in reversed(liq.sweeps)]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.caption(f"No sweeps in the last {cfg.SWEEP_SCAN_BARS} bars.")
    for n in liq.notes:
        st.write("•", n)


left, right = st.columns([3, 2])
with left:
    panel("Chart", render_chart)
with right:
    panel("Session clock", render_clock)

st.divider()
mcol1, mcol2 = st.columns(2)
with mcol1:
    panel("Rates", render_macro, rates, "L1 · Rates")
with mcol2:
    panel("Dollar", render_macro, dollar, "L2 · Dollar")
panel("Macro readings", render_macro_readings)
panel("Link strength", render_link_chart)

st.divider()
xcol1, xcol2 = st.columns([1, 1])
with xcol1:
    panel("Cross-asset", render_xasset)
with xcol2:
    panel("Implied move", render_implied_chart)

st.divider()
gcol1, gcol2 = st.columns([1, 1])
with gcol1:
    panel("Regime", render_regime)
with gcol2:
    panel("Regime chart", render_regime_chart)
with regime_slot.container():
    panel("Regime banner", render_regime_banner)

st.divider()
lcol, rcol = st.columns(2)
with lcol:
    panel("Technicals", render_tech)
with rcol:
    panel("Liquidity", render_liq)
