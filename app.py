"""
XAU Monitor — gold (XAUUSD) intraday dashboard.
Phase 0: data, session clock, chart.  Phase 1: L5 technicals, L6 liquidity.
Phase 2: L1 rates, L2 dollar (correlation-weighted).
Phase 3: L3 cross-asset agreement (implied move, structural bid, silver, metals).
Phase 4: L8 regime classifier (which driver is in control, trade style).
Phase 5: L4 flows/positioning (ETF, COT), L7 options/vol (GLD walls, skew, GVZ).
Phase 6: calendar gates (FOMC, CPI, NFP, PPI, PCE, auctions, rollover, weekly open).
Phase 7: master signal (±100, G1–G10, gates, plan) and forward-test journal.

Chart and levels are in COMEX GC=F futures prices. Broker XAUUSD spot =
futures minus the basis — measured live from the Swissquote spot feed (or the
manual sidebar value as a fallback). A 10-second ticker shows live spot, and the
whole app recomputes on the sidebar auto-refresh interval.
Every panel is isolated in try/except so one failure never white-screens the app.
"""
from datetime import datetime, timedelta, timezone

import streamlit as st

st.set_page_config(page_title="XAU Monitor", page_icon="🟡", layout="wide")

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Guard against partial pushes: name any project file missing from the repo
# instead of crashing with a redacted ModuleNotFoundError.
import os as _os
_APP_DIR = _os.path.dirname(_os.path.abspath(__file__))
_REQUIRED_FILES = ["xau_config.py", "xau_sessions.py", "xau_data.py", "xau_technicals.py",
                   "xau_liquidity.py", "xau_macro.py", "xau_rates.py", "xau_dollar.py",
                   "xau_crossasset.py", "xau_regime.py", "xau_runtime.py",
                   "xau_flows.py", "xau_options.py", "xau_calendar.py",
                   "xau_master_signal.py", "xau_journal.py", "xau_spot.py", "xau_bg.py"]
_missing = [f for f in _REQUIRED_FILES if not _os.path.exists(_os.path.join(_APP_DIR, f))]
if _missing:
    st.error("**Missing from the repo:** " + ", ".join(f"`{f}`" for f in _missing) +
             "  \nUpload them to the repo root next to app.py; the app redeploys "
             "on its own.")
    st.stop()

import xau_bg as xb
import xau_calendar as xk
import xau_config as cfg
import xau_crossasset as xc
import xau_data as xd
import xau_dollar as xdl
import xau_flows as xf
import xau_options as xo
import xau_journal as xj
import xau_liquidity as xl
import xau_master_signal as xms
import xau_rates as xr
import xau_regime as xg
import xau_sessions as xs
import xau_spot as xsp
import xau_technicals as xt
import xau_macro as xm
import xau_runtime as xrt

# Reload project modules if a git push changed them (Streamlit Cloud can keep
# stale copies in memory), dependency order: config first, app-level last.
_reloaded = xrt.ensure_fresh([cfg, xs, xd, xt, xl, xm, xr, xdl, xc, xg, xf, xo, xk,
                              xms, xj, xsp])
REQUIRED_CONFIG_VERSION = 9
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
    basis_mode = st.radio(
        "Basis (GC=F − spot)", ["Auto", "Manual"], horizontal=True,
        help="Auto measures GC=F last − live XAUUSD spot (Swissquote) and uses the median "
             "of recent readings. Manual uses the number below.")
    basis_manual = st.number_input(
        "Manual basis ($)", value=float(cfg.DEFAULT_BASIS), step=0.5, format="%.2f",
        help="Used in Manual mode, and in Auto mode until the first live reading.")
    refresh_label = st.selectbox("Auto-refresh (full recompute)", list(cfg.REFRESH_OPTIONS),
                                 index=list(cfg.REFRESH_OPTIONS).index(cfg.REFRESH_DEFAULT))
    live_ticker = st.checkbox(f"Live spot ticker (every {cfg.SPOT_TICK_SEC}s)", value=True)
    days_shown = st.slider("Trading days on chart", 1, 10, 3)
    show_bands = st.checkbox("Session bands", value=True)
    show_rounds = st.checkbox("Round-number levels", value=True)
    st.subheader("Overlays")
    show_vwap = st.checkbox("VWAP + σ bands", value=True)
    show_emas = st.checkbox("EMA 9 / 21 / 50", value=True)
    show_cpr = st.checkbox("CPR + R1/S1", value=True)
    show_sweeps = st.checkbox("Liquidity sweeps", value=True)
    show_rsi = st.checkbox("RSI pane", value=True)
    show_walls = st.checkbox("Option walls (GLD → GC=F)", value=True)
    show_events = st.checkbox("Economic events", value=True)
    show_plan = st.checkbox("Trade plan (entry / SL / TP)", value=True)
    if st.button("Refresh data"):
        st.cache_data.clear()
        xb.reset()
    st.caption(f"Signal timeframe {cfg.SIGNAL_INTERVAL} · Yahoo bars cached "
               f"{cfg.CACHE_TTL_SEC}s · spot {cfg.SPOT_TTL_SEC}s · options 15 min · "
               f"COT/FRED 6 h · times in EAT ({cfg.DISPLAY_TZ})")

now = datetime.now(timezone.utc)
# Start every slow download in parallel (background threads, no waiting) …
def _prefetch():
    xr.load_fred(wait=0)
    xf.prefetch()
    xo.prefetch()


try:
    _prefetch()
except Exception:  # noqa: BLE001 — prefetch is an optimisation, never fatal
    pass
# … then take the price bundle: instant after the first load, refreshed behind the scenes.
bundle = panel("Data", xd.get_bundle) or {
    "gold": xd._empty(), "primary": cfg.PRIMARY, "cross": {}, "errors": ["data panel failed"],
    "fetched_at": now}
gold = bundle["gold"]
import time as _time
st.session_state["_last_full"] = _time.time()
spot = xsp.fetch_spot(now)
basis_reading = xsp.live_basis(gold, spot, now)
auto_basis = xsp.record_basis(xsp.basis_store(), basis_reading, now)
_eb = xsp.effective_basis(basis_mode, basis_manual, auto_basis)
basis, basis_src = _eb["basis"], _eb["source"]
tech = xt.get_tech_report(gold)
liq = xl.get_liq_report(gold)
fred = xr.load_fred()
rates = xr.get_rates_report(bundle, fred)
dollar = xdl.get_dollar_report(bundle)
xasset = xc.get_xasset_report(bundle)
regime = xg.get_regime_report(bundle, tech, rates, dollar, xasset)
flows = xf.get_flow_report()
opts = xo.get_options_report(gold, bundle["cross"].get("^GVZ"))
gate = xk.get_gate_report(now, em_exhausted=bool(opts.flags.get("em_exhausted")))
LAYERS = {"L1": rates, "L2": dollar, "L3": xasset, "L4": flows,
          "L5": tech, "L6": liq, "L7": opts, "L8": regime}
signal = xms.get_master_signal(LAYERS, gold, gate, basis, now)
if "journal_df" not in st.session_state:
    st.session_state["journal_df"] = None
_jdf, _jsaved, _jerr = xj.step(signal, gold, regime.regime if regime.ok else "",
                               df=st.session_state["journal_df"])
st.session_state["journal_df"] = _jdf


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
    stat = xb.status()
    bad = {k: v for k, v in stat.items() if v["error"]}
    n = len(bundle["errors"]) + len(bad)
    with st.expander(f"Data sources & notes ({n})" if n else "Data sources"):
        rows = [{"Source": k, "Age": "—" if v["age_s"] is None else f"{v['age_s']}s",
                 "Updating": "yes" if v["busy"] else "", "Last error": v["error"],
                 "Retry in": f"{v['retry_in_s']}s" if v["error"] else ""}
                for k, v in sorted(stat.items())]
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        mode = bundle.get("mode", "full")
        st.caption(f"Price bars: last {mode} download "
                   f"{(now - bundle['fetched_at']).total_seconds():.0f}s ago · "
                   f"refreshed in the background every {cfg.CACHE_TTL_SEC}s "
                   f"({cfg.BUNDLE_INCR_PERIOD} incremental, full reload every "
                   f"{cfg.BUNDLE_FULL_REFRESH_SEC // 3600} h).")
        for e in bundle["errors"]:
            st.write("•", e)


panel("Header", render_header)

_TICK = cfg.SPOT_TICK_SEC if live_ticker else 0
_FULL = cfg.REFRESH_OPTIONS[refresh_label]
_RUN_EVERY = min([x for x in (_TICK, _FULL) if x], default=None)


@st.fragment(run_every=_RUN_EVERY)
def live_strip():
    now_f = datetime.now(timezone.utc)
    sp = xsp.fetch_spot(now_f)
    c = st.columns(5)
    if sp:
        c[0].metric("XAUUSD spot (live)", f"{sp['mid']:,.2f}",
                    f"{sp['source']} · {sp['age_sec']:.0f}s old" + (" · STALE" if sp["stale"] else ""),
                    delta_color="off")
        c[1].metric("Spread", "—" if sp["spread"] is None else f"{sp['spread']:.2f}",
                    None if sp["bid"] is None else f"bid {sp['bid']:,.2f} · ask {sp['ask']:,.2f}",
                    delta_color="off")
    else:
        c[0].metric("XAUUSD spot (live)", "—", "spot feed unreachable", delta_color="off")
        c[1].metric("Spread", "—")
    if not gold.empty:
        age = (now_f - gold.index[-1].to_pydatetime()).total_seconds() / 60
        c[2].metric(f"{bundle['primary']} futures", f"{float(gold['Close'].iat[-1]):,.2f}",
                    f"bar {age:.0f} min old (Yahoo)", delta_color="off")
    rd = xsp.live_basis(gold, sp, now_f)
    c[3].metric("Basis in use", f"{basis:+.2f}",
                f"{basis_src.split(' (')[0]}" + ("" if rd["basis"] is None
                                                 else f" · now {rd['basis']:+.2f}"),
                delta_color="off")
    since = _time.time() - st.session_state.get("_last_full", _time.time())
    nxt = "off" if not _FULL else f"next in {max(0, _FULL - since):.0f}s"
    c[4].metric("Last full refresh", f"{since:.0f}s ago", nxt, delta_color="off")
    if basis_mode == "Manual" and auto_basis is not None and \
            abs(basis_manual - auto_basis) > cfg.BASIS_DRIFT_WARN:
        st.warning(f"Manual basis {basis_manual:+.2f} is {basis_manual - auto_basis:+.2f} away "
                   f"from the live basis {auto_basis:+.2f} — spot levels will be off. "
                   "Switch to Auto or update the value.")
    if _FULL and since >= _FULL:
        st.rerun()


panel("Live prices", live_strip)
gate_slot = st.empty()            # filled once the gate helpers are defined
signal_slot = st.empty()          # master signal card
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
    if spot:
        c[1].metric("XAUUSD spot", fmt(spot["mid"]), f"live · basis {basis:+.2f}",
                    delta_color="off")
    else:
        c[1].metric("XAUUSD spot est.", fmt(xd.to_spot(ch["last"], basis)),
                    f"GC=F − basis {basis:+.2f}", delta_color="off")
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


class _ShapeBatch:
    """Collects shapes/annotations and applies them in one update_layout call.
    Plotly's add_shape/add_hline copy the whole shape list on every call, which
    made ~60 level lines cost ~0.5 s per refresh."""

    def __init__(self, fig):
        self.fig, self.sh, self.an = fig, [], []

    @staticmethod
    def _refs(row):
        return ("x", "y") if row in (1, None) else (f"x{row}", f"y{row}")

    @staticmethod
    def _line(kw):
        line = dict(kw.pop("line", {}) or {})
        for k in ("color", "width", "dash"):
            if f"line_{k}" in kw:
                line[k] = kw.pop(f"line_{k}")
        return line

    def add_shape(self, row=1, col=None, **kw):
        xr, yr = self._refs(row)
        kw.setdefault("xref", xr)
        kw.setdefault("yref", yr)
        kw["line"] = self._line(kw)
        self.sh.append(kw)

    def add_annotation(self, row=1, col=None, **kw):
        xr, yr = self._refs(row)
        kw.setdefault("xref", xr)
        kw.setdefault("yref", yr)
        self.an.append(kw)

    def add_hline(self, y, row=1, col=None, **kw):
        xr, yr = self._refs(row)
        self.sh.append(dict(type="line", xref=f"{xr} domain", x0=0, x1=1, yref=yr,
                            y0=y, y1=y, line=self._line(kw)))

    def add_hrect(self, y0, y1, row=1, col=None, fillcolor=None, line_width=0, **kw):
        xr, yr = self._refs(row)
        self.sh.append(dict(type="rect", xref=f"{xr} domain", x0=0, x1=1, yref=yr,
                            y0=y0, y1=y1, fillcolor=fillcolor, line=dict(width=line_width),
                            layer="below"))

    def add_vline(self, x, row="all", col=None, **kw):
        self.sh.append(dict(type="line", xref="x", x0=x, x1=x, yref="paper", y0=0, y1=1,
                            line=self._line(kw)))

    def add_vrect(self, x0, x1, row="all", col=None, fillcolor=None, line_width=0,
                  layer="below", **kw):
        self.sh.append(dict(type="rect", xref="x", x0=x0, x1=x1, yref="paper", y0=0, y1=1,
                            fillcolor=fillcolor, line=dict(width=line_width), layer=layer))

    def flush(self):
        self.fig.update_layout(shapes=self.sh, annotations=self.an)


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
    B = _ShapeBatch(fig)

    if show_bands:
        for w in xs.session_windows(df.index[0].to_pydatetime(),
                                    df.index[-1].to_pydatetime() + timedelta(minutes=15)):
            B.add_vrect(x0=_local(w["start"]), x1=_local(w["end"]),
                          fillcolor=cfg.BAND_COLORS[w["name"]], line_width=0,
                          layer="below", row="all", col=1)

    if show_events and gate.ok:
        x0, x1 = df.index[0], df.index[-1] + pd.Timedelta(hours=12)
        for ev in xk.all_events((x0 - pd.Timedelta(days=1)).date(), x1.date()):
            if not (x0 <= ev["when"] <= x1) or ev["impact"] not in ("fomc", "high", "medium"):
                continue
            big = ev["impact"] in ("fomc", "high")
            B.add_vline(x=_local(ev["when"]), line_width=1.2 if big else 0.8,
                          line_dash="dash" if big else "dot",
                          line_color="rgba(192,57,43,0.8)" if big else "rgba(128,128,128,0.6)",
                          row="all", col=1)
            if big:
                B.add_annotation(x=_local(ev["when"]), y=1.0, yref="paper", text=ev["name"],
                                   showarrow=False, textangle=-90, xanchor="right",
                                   yanchor="top", font=dict(size=9, color="#c0392b"))

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
        B.add_shape(type="line", x0=x_today0, x1=x_end, y0=y, y1=y,
                      line=dict(color=color, dash=dash, width=width), row=1, col=1)
        B.add_annotation(x=x_end, y=y, text=text, showarrow=False, xanchor="left",
                           font=dict(size=10, color=color), row=1, col=1)

    if show_cpr and tech.ok and tech.cpr:
        c = tech.cpr
        B.add_shape(type="rect", x0=x_today0, x1=x_end, y0=c["BC"], y1=c["TC"],
                      fillcolor="rgba(52,152,219,0.18)", line_width=0, row=1, col=1)
        seg(c["P"], "P", "#3498db", "dot")
        seg(c["R1"], "R1", "#95a5a6", "dash")
        seg(c["S1"], "S1", "#95a5a6", "dash")
        lo_v, hi_v = float(df["Low"].min()), float(df["High"].max())
        pad = (hi_v - lo_v) * 0.15
        for k in ("R2", "R3", "S2", "S3"):
            if lo_v - pad <= c[k] <= hi_v + pad:
                seg(c[k], k, "#b2babb", "dot")
        vp = c.get("virgin_prior")
        if vp:
            B.add_shape(type="rect", x0=x_today0, x1=x_end, y0=vp["BC"], y1=vp["TC"],
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
    if show_plan and signal.ok and signal.plan:
        pl = signal.plan
        live = signal.action in ("LONG", "SHORT")
        dash = "solid" if live else "dot"
        x_plan0 = x[max(0, len(x) - 24)]
        for y, txt, col in ((pl["entry"], "Entry", "#7f8c8d"), (pl["sl"], "SL", "#c0392b"),
                            (pl["tp1"], "TP1", "#27ae60"), (pl["tp2"], "TP2", "#1e8449")):
            B.add_shape(type="line", x0=x_plan0, x1=x_end, y0=y, y1=y,
                          line=dict(color=col, width=1.6 if live else 1, dash=dash),
                          row=1, col=1)
            B.add_annotation(x=x_plan0, y=y, text=f"{txt} {y:,.1f}", showarrow=False,
                               xanchor="right", font=dict(size=10, color=col),
                               bgcolor="rgba(255,255,255,0.8)", borderpad=1, row=1, col=1)

    if show_walls and opts.ok:
        if opts.call_wall:
            seg(opts.call_wall, "Call wall", "#c0392b", "longdash", 1.5)
        if opts.put_wall:
            seg(opts.put_wall, "Put wall", "#27ae60", "longdash", 1.5)
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
                B.add_hline(y=lvl, line_color="rgba(150,150,150,0.35)", line_width=1,
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
        B.add_hrect(y0=45, y1=55, fillcolor="rgba(128,128,128,0.12)", line_width=0,
                      row=2, col=1)
        for lvl, clr in ((70, "#ef5350"), (60, "#e67e22"), (50, "#7f8c8d"),
                         (40, "#e67e22"), (30, "#26a69a")):
            B.add_hline(y=lvl, line_color=clr, line_width=1, line_dash="dot", row=2, col=1)
        pb = (tech.cpr or {}).get("rsi_pullback") if tech.ok else None
        if pb and pb.get("bar_time") is not None and pb.get("rsi_at") is not None:
            col = {"CONFIRMED": "#27ae60", "CAUTION": "#e67e22",
                   "REJECTED": "#c0392b"}.get(pb["state"], "#7f8c8d")
            fig.add_trace(go.Scatter(
                x=[_local(pb["bar_time"])], y=[pb["rsi_at"]], mode="markers",
                marker=dict(size=11, color=col, symbol="circle",
                            line=dict(color="white", width=1)),
                name="RSI pullback", showlegend=False,
                hovertext=f"TC/BC pullback RSI {pb['rsi_at']:.0f} — {pb['state']}",
                hoverinfo="text"), row=2, col=1)
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
    B.flush()
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


# ── L4 flows / L7 options ───────────────────────────────────────────────────
def _layer_table(rep, pts):
    rows = [{"Component": k, "Points": v, "Max": pts[k], "Reading": rep.details.get(k, "")}
            for k, v in rep.components.items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def render_flows():
    st.subheader("L4 · Flows & positioning")
    if not flows.ok:
        st.info(f"Flows unavailable: {flows.error}. CFTC and ETF data come from "
                "publicreporting.cftc.gov and Yahoo daily bars.")
        return
    c = st.columns(4)
    c[0].metric(f"Score (±{cfg.FLOWS_MAX})", f"{flows.score:+.1f}", _bias_badge(flows.bias),
                delta_color="off")
    cs = flows.cot
    c[1].metric("MM net long", "—" if not cs else f"{cs['net'] / 1000:,.0f}k",
                None if not cs else f"{cs['week_chg'] / 1000:+,.1f}k w/w", delta_color="off")
    c[2].metric("3y percentile", "—" if not cs or cs["percentile"] is None
                else f"{cs['percentile']:.0f}th")
    c[3].metric("ETF source", "shares" if flows.etf_source.startswith("GLD shares")
                else "proxy" if flows.etf_source else "—")
    _layer_table(flows, cfg.FLOW_PTS)
    for n in flows.notes:
        st.write("•", n)


def render_cot_chart():
    h = flows.cot_history if flows.ok else None
    if h is None or len(h) < 10:
        return
    h = h.iloc[-cfg.COT_PCT_WEEKS:]
    fig = go.Figure(go.Scatter(x=h.index, y=h["net"] / 1000, mode="lines",
                               line=dict(color="#2a78d6", width=2), name="Managed money net",
                               hovertemplate="%{x|%d %b %Y}: %{y:,.0f}k<extra></extra>"))
    hi = h["net"].quantile(cfg.COT_CROWDED_PCT / 100) / 1000
    lo = h["net"].quantile(cfg.COT_WASHED_PCT / 100) / 1000
    for y, txt, pos in ((hi, f"crowded ({cfg.COT_CROWDED_PCT}th)", "top left"),
                        (lo, f"washed out ({cfg.COT_WASHED_PCT}th)", "bottom right")):
        fig.add_hline(y=y, line_dash="dash", line_width=1, line_color="rgba(128,128,128,0.7)",
                      annotation_text=txt, annotation_position=pos,
                      annotation_font_size=10)
    fig.update_layout(height=250, margin=dict(l=10, r=20, t=36, b=20), showlegend=False,
                      title="COMEX gold — managed money net position (thousand contracts)",
                      title_font_size=13, yaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
                      xaxis=dict(showgrid=False))
    st.plotly_chart(fig, width="stretch")


def render_options():
    st.subheader("L7 · Options & volatility")
    if not opts.ok:
        st.info(f"Options/vol unavailable: {opts.error}")
        return
    c = st.columns(4)
    c[0].metric(f"Score (±{cfg.OPTIONS_MAX})", f"{opts.score:+.1f}", _bias_badge(opts.bias),
                delta_color="off")
    c[1].metric("Daily exp. move", "—" if opts.em_daily is None else f"${opts.em_daily:,.0f}",
                None if opts.em_used is None else f"{opts.em_used:.0%} used", delta_color="off")
    c[2].metric("Dealer gamma", "—" if opts.gex is None else
                ("long" if opts.gex > 0 else "short"),
                None if opts.gex is None else ("dampens moves" if opts.gex > 0 else
                                               "amplifies moves"), delta_color="off")
    c[3].metric("GVZ", "—" if opts.gvz is None else f"{opts.gvz:.1f}",
                None if opts.gvz_day_pct is None else f"{opts.gvz_day_pct:+.1f}%",
                delta_color="off")
    _layer_table(opts, cfg.OPT_PTS)
    st.caption("GLD options scaled to GC=F by the live price ratio"
               + ("" if opts.ratio is None else f" ({opts.ratio:.2f}×)")
               + ". Most gold options trade on COMEX, so treat walls as approximate.")
    for n in opts.notes:
        st.write("•", n)


def render_oi_chart():
    t = opts.strikes if opts.ok else None
    if t is None or t.empty:
        return
    fig = go.Figure()
    fig.add_trace(go.Bar(x=t["strike_gold"], y=t["call_oi"], name="Call OI",
                         marker_color="#2a78d6",
                         hovertemplate="%{x:,.0f}: %{y:,.0f} calls<extra></extra>"))
    fig.add_trace(go.Bar(x=t["strike_gold"], y=-t["put_oi"], name="Put OI",
                         marker_color="#eb6834",
                         hovertemplate="%{x:,.0f}: %{customdata:,.0f} puts<extra></extra>",
                         customdata=t["put_oi"]))
    last = float(gold["Close"].iat[-1])
    fig.add_vline(x=last, line_width=1.5, line_color="rgba(128,128,128,0.9)",
                  annotation_text="price", annotation_position="top",
                  annotation_font_size=10)
    fig.update_layout(height=250, margin=dict(l=10, r=20, t=36, b=20), barmode="relative",
                      bargap=0.15, title="GLD open interest by strike (GC=F terms) — "
                      "calls up, puts down", title_font_size=13,
                      legend=dict(orientation="h", y=-0.25, x=0),
                      yaxis=dict(gridcolor="rgba(128,128,128,0.15)", title="contracts"),
                      xaxis=dict(showgrid=False))
    st.plotly_chart(fig, width="stretch")


# ── Calendar gates ───────────────────────────────────────────────────────────
GATE_ICON = {"OPEN": "🟢", "CAUTION": "🟡", "BLOCKED": "🔴"}


def _eat(ts):
    return ts.astimezone(xs.DISP)


def _countdown(mins):
    if mins is None:
        return "—"
    d, rem = divmod(int(mins), 1440)
    h, m = divmod(rem, 60)
    return (f"{d}d " if d else "") + f"{h}h {m:02d}m"


def render_gate_banner():
    if not gate.ok:
        st.warning("🟡 Entry gate: calendar check failed — trade manually around news. "
                   f"({gate.error})")
        return
    lines = [f"**{GATE_ICON[gate.state]} Entry gate: {gate.state}**"]
    if gate.reasons:
        lines.append("Blocked by: " + "; ".join(gate.reasons))
        if gate.resume_at:
            lines.append(f"Clears at {_eat(gate.resume_at):%a %H:%M} EAT")
    if gate.cautions:
        lines.append("Caution: " + "; ".join(gate.cautions))
    if gate.next_high:
        lines.append(f"Next high-impact: **{gate.next_high['name']}** "
                     f"{_eat(gate.next_high['when']):%a %d %b %H:%M} EAT "
                     f"(in {_countdown(gate.mins_to_next_high)})")
    box = {"OPEN": st.success, "CAUTION": st.warning, "BLOCKED": st.error}[gate.state]
    box("  \n".join(lines))


IMPACT_LABEL = {"fomc": "FOMC", "high": "High", "medium": "Medium", "low": "Low"}


def render_calendar():
    st.subheader("Economic calendar (EAT)")
    if not gate.ok:
        st.info(f"Calendar unavailable: {gate.error}")
        return
    rows = []
    for ev in gate.upcoming:
        pre, post = cfg.GATE_WINDOWS[ev["impact"]]
        loc = _eat(ev["when"])
        rows.append({"When (EAT)": f"{loc:%a %d %b %H:%M}", "Event": ev["name"],
                     "Impact": IMPACT_LABEL[ev["impact"]],
                     "Blocks": f"{_eat(ev['when'] - timedelta(minutes=pre)):%H:%M}–"
                               f"{_eat(ev['when'] + timedelta(minutes=post)):%H:%M}",
                     "In": _countdown((ev["when"] - now).total_seconds() / 60)})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.caption("No scheduled events in the lookahead window.")
    st.caption("Dates verified against BLS, BEA, the Fed and Treasury on 26 Sep 2026. "
               f"Add or override events in {cfg.EVENTS_CSV} (date,time_et,name,impact). "
               "November 10y/30y refunding auction dates are not yet published.")
    for n in gate.notes:
        st.write("•", n)


# ── Master signal ────────────────────────────────────────────────────────────
ACTION_ICON = {"LONG": "🟢", "SHORT": "🔴", "WAIT": "⏸️"}


def render_signal():
    if not signal.ok:
        st.error(f"Master signal unavailable: {signal.error}")
        return
    st.subheader("Master signal")
    c = st.columns(5)
    c[0].metric("Score (±100)", f"{signal.score:+.1f}",
                f"raw {signal.raw_total:+.1f} / ±{sum(cfg.LAYER_MAX.values())}",
                delta_color="off")
    c[1].metric("Action", f"{ACTION_ICON[signal.action]} {signal.action}",
                None if signal.action != "WAIT" or signal.direction == "NEUTRAL"
                else f"leaning {signal.direction.lower()}", delta_color="off")
    c[2].metric("Tier", signal.tier)
    c[3].metric("Size ×", f"{signal.size_mult:.2f}")
    c[4].metric("Layers agreeing", f"{signal.agree} / 8", f"{signal.active} active",
                delta_color="off")
    if signal.blocked_by:
        st.caption("Blocked by: " + ", ".join(signal.blocked_by))

    left, right = st.columns([1, 1])
    with left:
        p = signal.plan
        if p:
            hdr = "Trade plan" if signal.action != "WAIT" else \
                "Hypothetical plan (not actionable now)"
            st.markdown(f"**{hdr} — {p['direction']}**")
            rows = [
                {"Level": "Entry zone", "GC=F": f"{p['zone'][0]:,.2f} – {p['zone'][1]:,.2f}",
                 "XAUUSD spot": f"{p['zone'][0] - p['basis']:,.2f} – {p['zone'][1] - p['basis']:,.2f}",
                 "Basis of level": f"last {p['entry']:,.2f}"},
                {"Level": "Stop", "GC=F": f"{p['sl']:,.2f}", "XAUUSD spot": f"{p['spot']['sl']:,.2f}",
                 "Basis of level": f"{p['stop_src']} · risk ${p['risk']:,.2f} "
                                   f"({p['risk'] / p['atr']:.1f} ATR)"},
                {"Level": "TP1", "GC=F": f"{p['tp1']:,.2f}", "XAUUSD spot": f"{p['spot']['tp1']:,.2f}",
                 "Basis of level": p['tp1_src'] if p['tp1_src'].endswith("R")
                 else f"{p['tp1_src']} · {p['rr1']:.1f}R"},
                {"Level": "TP2", "GC=F": f"{p['tp2']:,.2f}", "XAUUSD spot": f"{p['spot']['tp2']:,.2f}",
                 "Basis of level": p['tp2_src'] if p['tp2_src'].endswith("R")
                 else f"{p['tp2_src']} · {p['rr2']:.1f}R"},
            ]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption(f"Take {cfg.JOURNAL_TP1_PART:.0%} at TP1 and move the stop to entry. "
                       f"Spot = GC=F − basis {p['basis']:+.2f} ({basis_src}).")
        else:
            st.caption("No directional plan — score below the C-tier threshold "
                       f"(|score| < {cfg.TIERS[-1][0]}).")
        if signal.rules:
            st.markdown("**Rules fired**")
            st.dataframe(pd.DataFrame(signal.rules).rename(columns={
                "code": "Rule", "rule": "What", "effect": "Effect", "detail": "Detail"}),
                hide_index=True, width="stretch")
        for n in signal.notes:
            st.write("•", n)
    with right:
        render_layer_bars()


def render_layer_bars():
    rows = list(reversed(signal.layers))
    names = [f"{r['key']} {r['name']}" for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=names, x=[r["max"] for r in rows], orientation="h",
                         marker_color="rgba(128,128,128,0.12)", hoverinfo="skip",
                         showlegend=False))
    fig.add_trace(go.Bar(y=names, x=[-r["max"] for r in rows], orientation="h",
                         marker_color="rgba(128,128,128,0.12)", hoverinfo="skip",
                         showlegend=False))
    fig.add_trace(go.Bar(
        y=names, x=[r["score"] for r in rows], orientation="h",
        marker_color=["#2a78d6" if r["score"] >= 0 else "#eb6834" for r in rows],
        text=[("n/a" if not r["ok"] else f"{r['score']:+.1f}") for r in rows],
        textposition="outside", showlegend=False,
        hovertemplate="%{y}: %{x:+.1f}<extra></extra>"))
    lim = max(cfg.LAYER_MAX.values()) * 1.25
    fig.update_layout(height=320, barmode="overlay", margin=dict(l=10, r=20, t=40, b=20),
                      title="Layer contributions (grey = each layer's max; blue bullish, "
                            "orange bearish)", title_font_size=13,
                      xaxis=dict(range=[-lim, lim], zeroline=True,
                                 zerolinecolor="rgba(128,128,128,0.6)",
                                 gridcolor="rgba(128,128,128,0.15)"),
                      yaxis=dict(showgrid=False), bargap=0.35)
    st.plotly_chart(fig, width="stretch")


def render_journal():
    st.subheader("Forward-test journal")
    df = st.session_state.get("journal_df")
    if df is None:
        df = xj.empty()
    s_ = xj.stats(df, now)
    c = st.columns(5)
    c[0].metric("Closed trades", s_["closed"], f"{s_['open']} open", delta_color="off")
    c[1].metric("Win rate (TP1 hit)", "—" if s_["win_rate"] is None else f"{s_['win_rate']:.0%}",
                f"target {s_['target']:.0%}", delta_color="off")
    c[2].metric("Net R", f"{s_['net_r']:+.2f}",
                None if s_["avg_r"] is None else f"avg {s_['avg_r']:+.2f}R", delta_color="off")
    c[3].metric("Test days", f"{s_['days']} / {s_['days_target']}")
    c[4].metric("MT5 gate", "ready" if s_["on_track"] and s_["days"] >= s_["days_target"]
                else "not yet", "needs ≥20 trades, ≥75%, 60 days", delta_color="off")
    if len(df):
        show = df.sort_values("bar_utc", ascending=False).head(25).copy()
        show["bar (EAT)"] = show["bar_utc"].dt.tz_convert(cfg.DISPLAY_TZ).dt.strftime("%a %d %b %H:%M")
        st.dataframe(show[["bar (EAT)", "direction", "tier", "score", "entry", "sl", "tp1",
                           "tp2", "status", "r_mult", "regime"]],
                     hide_index=True, width="stretch")
    else:
        st.caption("No signals logged yet. A trade is logged the first time the master "
                   "action turns LONG or SHORT on a 15m bar.")
    d1, d2 = st.columns(2)
    with d1:
        st.download_button("Download journal CSV", xj.to_csv_bytes(df),
                           file_name="xau_journal.csv", mime="text/csv")
    with d2:
        up = st.file_uploader("Restore journal CSV", type=["csv"],
                              label_visibility="collapsed")
        if up is not None and st.session_state.get("_restored") != up.name:
            restored = xj.from_csv_bytes(up.getvalue())
            st.session_state["journal_df"] = restored
            st.session_state["_restored"] = up.name
            xj.save(restored)
            st.success(f"Restored {len(restored)} journal rows.")
    note = "" if _jsaved else " Saving to disk failed — the journal lives in this browser session."
    st.caption("The journal only advances while the dashboard is open, and Streamlit Cloud "
               "wipes files on reboot or redeploy — download it regularly and restore after "
               "a push." + note + (f" ({_jerr})" if _jerr else ""))


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
    render_cpr()


CPR_ORDER = ["R3", "R2", "R1", "TC", "P", "BC", "S1", "S2", "S3"]
POS_LABEL = {"ABOVE_TC": "Above TC", "BELOW_BC": "Below BC", "INSIDE": "Inside CPR"}


def render_cpr():
    cp = tech.cpr if tech.ok else None
    if not cp or "P" not in cp:
        return
    price = float(tech.price)
    atr = float(tech.atr) or 1.0
    st.markdown("**CPR — today's levels** (from the previous CME day's H/L/C)")
    c = st.columns(2) + st.columns(2)
    c[0].metric("CPR type", str(cp.get("width_class", "—")).capitalize(),
                f"{cp['width_pct']:.2f}% · ${cp.get('width_abs', cp['TC'] - cp['BC']):,.2f}",
                delta_color="off")
    c[1].metric("Price vs CPR", POS_LABEL.get(cp.get("price_vs_cpr"), "—"))
    c[2].metric("Virgin today", "Yes" if cp.get("virgin_today") else "No",
                "magnet to P" if cp.get("virgin_today") else "CPR tested", delta_color="off")
    rel = cp.get("relationship", "n/a")
    c[3].metric("vs yesterday", rel.split(" (")[0].capitalize(),
                rel.split(" (")[1].rstrip(")") if " (" in rel else None, delta_color="off")
    if cp.get("type_bias"):
        st.caption(cp["type_bias"])

    rows, placed = [], False
    for k in CPR_ORDER:
        lv = float(cp[k])
        if not placed and price > lv:
            rows.append({"Level": "▶ price", "GC=F": f"{price:,.2f}",
                         "Spot": f"{price - basis:,.2f}", "Distance": "—"})
            placed = True
        d = lv - price
        rows.append({"Level": k, "GC=F": f"{lv:,.2f}", "Spot": f"{lv - basis:,.2f}",
                     "Distance": f"{d:+,.2f} ({d / atr:+.1f} ATR)"})
    if not placed:
        rows.append({"Level": "▶ price", "GC=F": f"{price:,.2f}",
                     "Spot": f"{price - basis:,.2f}", "Distance": "—"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=388)

    su = cp.get("setup") or {}
    lines = []
    if su.get("text"):
        lines.append(f"**Setup:** {su['text']}")
    if su.get("direction"):
        z0, z1 = su["zone"]
        lines.append(f"**{su['direction']}** zone {z0:,.2f}–{z1:,.2f} (spot "
                     f"{z0 - basis:,.2f}–{z1 - basis:,.2f}) · target {su['target']:,.2f} · "
                     f"invalid {su['invalid']:,.2f}")
    pb = cp.get("rsi_pullback") or {}
    if pb.get("text"):
        icon = {"CONFIRMED": "✅", "CAUTION": "⚠️", "REJECTED": "⛔", "WAITING": "⏳"}.get(
            pb["state"], "")
        lines.append(f"**RSI pullback check:** {icon} {pb['state']} — {pb['text']}")
    if cp.get("flip"):
        t = cp.get("flip_time")
        when = "" if t is None else f" at {_local(t):%H:%M} EAT"
        lines.append(f"**Flip:** {cp['flip']}{when}")
    if lines:
        if pb.get("state") == "REJECTED" and su.get("direction"):
            box = st.warning
        else:
            box = st.success if su.get("direction") == "BUY" else \
                st.error if su.get("direction") == "SELL" else st.info
        box("  \n".join(lines))
    st.caption("CPR setup is a stand-alone read like the NAS100 scalping engine; "
               "the master signal only uses CPR position (±2 in L5). RSI pullback check: "
               "longs want the dip into TC to hold RSI ≥45 (not <40); shorts want the "
               "bounce into BC to keep RSI ≤55 (not >60). Grey band on the RSI pane = 45–55.")


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
st.divider()
panel("Calendar", render_calendar)

st.divider()
fcol1, fcol2 = st.columns(2)
with fcol1:
    panel("Flows", render_flows)
    panel("COT chart", render_cot_chart)
with fcol2:
    panel("Options", render_options)
    panel("OI chart", render_oi_chart)
with signal_slot.container():
    panel("Master signal", render_signal)
with gate_slot.container():
    panel("Gate banner", render_gate_banner)
with regime_slot.container():
    panel("Regime banner", render_regime_banner)

st.divider()
lcol, rcol = st.columns(2)
with lcol:
    panel("Technicals", render_tech)
with rcol:
    panel("Liquidity", render_liq)

st.divider()
panel("Journal", render_journal)
