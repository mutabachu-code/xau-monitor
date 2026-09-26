"""
xau_flows.py — Layer 4: investment flows and positioning (score ±10).

  ETF flow (±5)     GLD shares-outstanding change over 5 days (true creations/
                    redemptions). If yfinance can't supply share history, a
                    proxy: signed dollar volume of GLD + IAU (up days +, down
                    days −) over 5 days ÷ their average daily dollar volume.
  COT trend (±3)    CFTC managed-money net position, week-on-week change as a
                    % of open interest (weekly: Tuesday data, Friday release).
  COT crowding (±2) Net long vs its 3-year range. Above the 90th percentile is
                    crowded (−2, raises G4); below the 10th is washed out (+2).

Fetchers are cached and never raise; failures return None and aren't cached.
"""
import json
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import xau_config as cfg
import xau_data as xd

try:
    import streamlit as st
    _cot_cache = st.cache_data(ttl=cfg.COT_TTL_SEC, show_spinner=False)
    _etf_cache = st.cache_data(ttl=cfg.ETF_TTL_SEC, show_spinner=False)
except Exception:  # pragma: no cover
    def _cot_cache(fn):
        return fn
    _etf_cache = _cot_cache


# ── Fetchers ─────────────────────────────────────────────────────────────────
def parse_cot(records: List[Dict]) -> pd.DataFrame:
    """CFTC Socrata records → weekly frame (oldest first): long, short, oi, net."""
    rows = []
    for r in records:
        try:
            rows.append({
                "date": pd.Timestamp(r["report_date_as_yyyy_mm_dd"][:10]),
                "long": float(r["m_money_positions_long_all"]),
                "short": float(r["m_money_positions_short_all"]),
                "oi": float(r["open_interest_all"]),
            })
        except (KeyError, TypeError, ValueError):
            continue
    if not rows:
        return pd.DataFrame(columns=["long", "short", "oi", "net"])
    df = pd.DataFrame(rows).drop_duplicates("date").set_index("date").sort_index()
    df["net"] = df["long"] - df["short"]
    return df


@_cot_cache
def _cot_raw() -> pd.DataFrame:
    req = urllib.request.Request(cfg.COT_URL, headers={"User-Agent": "xau-monitor/1.0"})
    with urllib.request.urlopen(req, timeout=cfg.COT_TIMEOUT) as resp:
        records = json.loads(resp.read().decode("utf-8"))
    df = parse_cot(records)
    if df.empty:
        raise ValueError("CFTC returned no usable gold records")
    return df


def fetch_cot() -> Optional[pd.DataFrame]:
    try:
        return _cot_raw()
    except Exception:  # noqa: BLE001
        return None


@_etf_cache
def _etf_daily_raw() -> Dict[str, pd.DataFrame]:
    raw = xd._download(cfg.ETF_TICKERS, "1d", cfg.ETF_DAILY_PERIOD)
    out = {t: xd.extract_ticker(raw, t) for t in cfg.ETF_TICKERS}
    if all(v.empty for v in out.values()):
        raise ValueError("no ETF daily data")
    return out


def fetch_etf_daily() -> Optional[Dict[str, pd.DataFrame]]:
    try:
        return _etf_daily_raw()
    except Exception:  # noqa: BLE001
        return None


@_etf_cache
def _gld_shares_raw() -> pd.Series:
    import yfinance as yf
    start = (pd.Timestamp.now('UTC') - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
    s = yf.Ticker("GLD").get_shares_full(start=start)
    if s is None or len(s) < 5:
        raise ValueError("no GLD share history")
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.astype(float)


def fetch_gld_shares() -> Optional[pd.Series]:
    try:
        return _gld_shares_raw()
    except Exception:  # noqa: BLE001
        return None


# ── Components ───────────────────────────────────────────────────────────────
def shares_flow_pct(shares: Optional[pd.Series], days: int = cfg.ETF_FLOW_DAYS) -> Optional[float]:
    """% change in shares outstanding over the last `days` calendar-distinct points."""
    if shares is None or len(shares) < 2:
        return None
    daily = shares.groupby(pd.DatetimeIndex(shares.index).normalize()).last()
    if len(daily) <= days:
        return None
    a, b = daily.iat[-1 - days], daily.iat[-1]
    return float((b / a - 1) * 100) if a else None


def volume_flow_proxy(etfs: Optional[Dict[str, pd.DataFrame]],
                      days: int = cfg.ETF_FLOW_DAYS) -> Optional[float]:
    """Σ sign(Δclose)·close·volume over `days` ÷ (days × 20-day avg $-volume)."""
    if not etfs:
        return None
    num = den = 0.0
    for df in etfs.values():
        if df is None or len(df) < 25:
            continue
        dv = df["Close"] * df["Volume"]
        sign = np.sign(df["Close"].diff())
        num += float((sign * dv).iloc[-days:].sum())
        den += float(dv.iloc[-25:-days].mean() * days)
    return num / den if den > 0 else None


def cot_stats(cot: Optional[pd.DataFrame]) -> Optional[Dict]:
    if cot is None or len(cot) < 2:
        return None
    last, prev = cot.iloc[-1], cot.iloc[-2]
    window = cot["net"].iloc[-cfg.COT_PCT_WEEKS:]
    pct = float((window < last["net"]).mean() * 100) if len(window) >= 20 else None
    chg = float(last["net"] - prev["net"])
    trend3 = float(last["net"] - cot["net"].iat[-4]) if len(cot) >= 4 else None
    return {"date": cot.index[-1], "net": float(last["net"]), "long": float(last["long"]),
            "short": float(last["short"]), "oi": float(last["oi"]), "week_chg": chg,
            "week_chg_pct_oi": chg / last["oi"] * 100 if last["oi"] else 0.0,
            "trend3": trend3, "percentile": pct, "weeks": len(window),
            "net_pct_oi": float(last["net"] / last["oi"] * 100) if last["oi"] else None}


# ── Report ───────────────────────────────────────────────────────────────────
@dataclass
class FlowReport:
    ok: bool = False
    error: str = ""
    score: float = 0.0
    bias: str = "NEUTRAL"
    components: Dict[str, Optional[float]] = field(default_factory=dict)
    details: Dict[str, str] = field(default_factory=dict)
    etf_source: str = ""
    cot: Optional[Dict] = None
    cot_history: Optional[pd.DataFrame] = None
    flags: Dict[str, bool] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


def _clip(x, m=1.0):
    return float(max(-m, min(m, x)))


def compute(cot: Optional[pd.DataFrame], etfs: Optional[Dict[str, pd.DataFrame]],
            shares: Optional[pd.Series]) -> FlowReport:
    rep = FlowReport()
    P = cfg.FLOW_PTS
    comp, det = {}, {}

    # ETF flow
    sp = shares_flow_pct(shares)
    if sp is not None:
        comp["ETF flow"] = round(P["ETF flow"] * _clip(sp / cfg.ETF_REF_SHARES_PCT), 1)
        det["ETF flow"] = f"GLD shares {sp:+.2f}% over {cfg.ETF_FLOW_DAYS} days " \
                          f"({'creations' if sp > 0 else 'redemptions' if sp < 0 else 'flat'})"
        rep.etf_source = "GLD shares outstanding"
    else:
        px = volume_flow_proxy(etfs)
        if px is not None:
            comp["ETF flow"] = round(P["ETF flow"] * _clip(px / cfg.ETF_REF_PROXY), 1)
            det["ETF flow"] = f"signed $-volume {px:+.2f}× normal over " \
                              f"{cfg.ETF_FLOW_DAYS} days (GLD+IAU proxy)"
            rep.etf_source = "GLD+IAU volume proxy"
        else:
            comp["ETF flow"] = None
            det["ETF flow"] = "ETF data unavailable"

    # COT
    cs = cot_stats(cot)
    rep.cot, rep.cot_history = cs, cot
    if cs:
        comp["COT trend"] = round(P["COT trend"] *
                                  _clip(cs["week_chg_pct_oi"] / cfg.COT_REF_WEEK_CHG_OI), 1)
        det["COT trend"] = f"managed money net {cs['net']:,.0f} " \
                           f"({cs['week_chg']:+,.0f} w/w, {cs['week_chg_pct_oi']:+.1f}% of OI), " \
                           f"as of {cs['date']:%d %b}"
        p = cs["percentile"]
        rep.flags["cot_crowded_long"] = bool(p is not None and p >= cfg.COT_CROWDED_PCT)
        rep.flags["cot_washed_out"] = bool(p is not None and p <= cfg.COT_WASHED_PCT)
        if p is None:
            comp["COT crowding"] = None
            det["COT crowding"] = "need ≥20 weeks of history"
        else:
            comp["COT crowding"] = -P["COT crowding"] if rep.flags["cot_crowded_long"] else \
                P["COT crowding"] if rep.flags["cot_washed_out"] else 0.0
            det["COT crowding"] = f"net long at {p:.0f}th percentile of {cs['weeks']} weeks"
        if rep.flags["cot_crowded_long"]:
            rep.notes.append(f"Managed money is crowded long ({p:.0f}th pct) — "
                             "vulnerable to a flush on hawkish news (G4)")
        if rep.flags["cot_washed_out"]:
            rep.notes.append(f"Managed money positioning washed out ({p:.0f}th pct) — "
                             "limited selling fuel left")
        age = (pd.Timestamp.now('UTC').tz_localize(None) - cs["date"]).days
        if age > 10:
            rep.notes.append(f"COT data is {age} days old")
    else:
        comp["COT trend"] = comp["COT crowding"] = None
        det["COT trend"] = det["COT crowding"] = "CFTC data unavailable"

    present = {k: v for k, v in comp.items() if v is not None}
    rep.components, rep.details = comp, det
    if not present:
        rep.error = "no flow or positioning data reachable"
        return rep
    rep.score = round(_clip(sum(present.values()), cfg.FLOWS_MAX), 1)
    cut = cfg.MACRO_BIAS_FRAC * cfg.FLOWS_MAX
    rep.bias = "LONG" if rep.score >= cut else "SHORT" if rep.score <= -cut else "NEUTRAL"
    rep.ok = True
    return rep


def get_flow_report(cot=None, etfs=None, shares=None, fetch: bool = True) -> FlowReport:
    """fetch=True pulls live data (cached); tests pass data directly."""
    try:
        if fetch:
            cot = cot if cot is not None else fetch_cot()
            shares = shares if shares is not None else fetch_gld_shares()
            etfs = etfs if etfs is not None else (fetch_etf_daily() if shares is None else None)
        return compute(cot, etfs, shares)
    except Exception as e:  # noqa: BLE001
        return FlowReport(error=f"{type(e).__name__}: {e}")
