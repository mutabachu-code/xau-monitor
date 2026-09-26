"""
xau_options.py — Layer 7: options and volatility (score ±10).

GLD options (yfinance) are scaled to GC=F by the live price ratio.
  Skew (±3)      call-minus-put IV at ~105% / ~95% strikes on the first expiry
                 ≥5 days out. Call skew = upside demand (bullish).
  Put/call (±2)  open-interest PCR over the nearest expiries.
  Walls (±3)     biggest call OI strike above spot (resistance / pin) and put
                 OI strike below (support). At a wall = full points against/for.
  Vol (±2)       GVZ day change read against gold's direction: vol up with
                 gold up = upside chase; vol up with gold down = fear selling.
Also: daily expected move (ATM IV) in gold $, how much of it today's range
has used (≥95% raises em_exhausted), and dealer gamma (GEX) sign: positive =
moves dampened / mean-revert, negative = moves amplified.
Most gold options trade on COMEX, not GLD, so this layer carries low weight.
"""
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import xau_config as cfg
import xau_data as xd
import xau_macro as xm
import xau_sessions as xs

try:
    import streamlit as st
    _opt_cache = st.cache_data(ttl=cfg.OPT_TTL_SEC, show_spinner=False)
    _gvz_cache = st.cache_data(ttl=3600, show_spinner=False)
except Exception:  # pragma: no cover
    def _opt_cache(fn):
        return fn
    _gvz_cache = _opt_cache


# ── Fetchers ─────────────────────────────────────────────────────────────────
@_opt_cache
def _chain_raw(ticker: str = cfg.OPT_TICKER, max_expiries: int = 5) -> Dict:
    import yfinance as yf
    t = yf.Ticker(ticker)
    exps = list(t.options or [])[:max_expiries]
    if not exps:
        raise ValueError("no expiries")
    try:
        spot = float(t.fast_info["last_price"])
    except Exception:  # noqa: BLE001
        spot = float(t.history(period="5d")["Close"].iat[-1])
    out = []
    for e in exps:
        oc = t.option_chain(e)
        out.append({"expiry": e, "calls": oc.calls, "puts": oc.puts})
    return {"spot": spot, "expiries": out, "fetched": datetime.now(timezone.utc)}


def fetch_chain() -> Optional[Dict]:
    try:
        return _chain_raw()
    except Exception:  # noqa: BLE001
        return None


@_gvz_cache
def _gvz_daily_raw() -> pd.DataFrame:
    df = xd.extract_ticker(xd._download(["^GVZ"], "1d", cfg.GVZ_DAILY_PERIOD), "^GVZ")
    if df.empty:
        raise ValueError("no GVZ daily")
    return df


def fetch_gvz_daily() -> Optional[pd.DataFrame]:
    try:
        return _gvz_daily_raw()
    except Exception:  # noqa: BLE001
        return None


# ── Math ─────────────────────────────────────────────────────────────────────
def bs_gamma(S: float, K: float, T: float, iv: float, r: float = cfg.OPT_RISK_FREE) -> float:
    if S <= 0 or K <= 0 or T <= 0 or iv <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * iv * iv) * T) / (iv * math.sqrt(T))
    return math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi) / (S * iv * math.sqrt(T))


def days_to_expiry(expiry: str, now: datetime) -> float:
    exp = pd.Timestamp(expiry).tz_localize(cfg.ET) + pd.Timedelta(hours=16)
    return max((exp - pd.Timestamp(now)).total_seconds() / 86400, 0.25)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["strike", "openInterest", "impliedVolatility",
                                     "bid", "ask", "lastPrice", "volume"])
    d = df.copy()
    for c in ("strike", "openInterest", "impliedVolatility", "bid", "ask", "lastPrice", "volume"):
        d[c] = pd.to_numeric(d.get(c, 0), errors="coerce").fillna(0.0)
    return d


def _iv_near(df: pd.DataFrame, strike: float) -> Optional[float]:
    d = df[(df["impliedVolatility"] > 0.01) & ((df["bid"] > 0) | (df["openInterest"] > 0))]
    if d.empty:
        return None
    row = d.iloc[(d["strike"] - strike).abs().argsort().iloc[0]]
    return float(row["impliedVolatility"])


def strike_table(chain: Dict, n_exp: int = cfg.OPT_EXPIRIES) -> pd.DataFrame:
    """OI by strike summed over the nearest n expiries."""
    parts = []
    for e in chain["expiries"][:n_exp]:
        c, p = _clean(e["calls"]), _clean(e["puts"])
        parts.append(pd.DataFrame({"strike": c["strike"], "call_oi": c["openInterest"],
                                   "put_oi": 0.0}))
        parts.append(pd.DataFrame({"strike": p["strike"], "call_oi": 0.0,
                                   "put_oi": p["openInterest"]}))
    if not parts:
        return pd.DataFrame(columns=["strike", "call_oi", "put_oi"])
    return pd.concat(parts).groupby("strike", as_index=False).sum().sort_values("strike")


def wall(side: pd.DataFrame, col: str) -> Optional[float]:
    """Strike with the most OI on this side, only if it clearly stands out."""
    oi = side[col]
    if oi.sum() <= 0:
        return None
    top = side.loc[oi.idxmax()]
    med = float(oi[oi > 0].median())
    return float(top["strike"]) if top[col] >= cfg.OPT_WALL_MIN_MULT * med else None


def dealer_gex(chain: Dict, now: datetime, n_exp: int = cfg.OPT_EXPIRIES) -> float:
    """Dealer $ gamma per 1% move under the standard assumption that call OI
    is dealer-long gamma and put OI dealer-short gamma:
    (Σ call γ·OI − Σ put γ·OI) × 100 × S² × 0.01. Sign is what matters."""
    S = chain["spot"]
    tot = 0.0
    for e in chain["expiries"][:n_exp]:
        T = days_to_expiry(e["expiry"], now) / 365
        for df, sgn in ((_clean(e["calls"]), 1), (_clean(e["puts"]), -1)):
            for k, oi, iv in zip(df["strike"], df["openInterest"], df["impliedVolatility"]):
                if oi > 0 and iv > 0.01:
                    tot += sgn * bs_gamma(S, k, T, iv) * oi
    return tot * 100 * S * S * 0.01


# ── Report ───────────────────────────────────────────────────────────────────
@dataclass
class OptReport:
    ok: bool = False
    error: str = ""
    score: float = 0.0
    bias: str = "NEUTRAL"
    components: Dict[str, Optional[float]] = field(default_factory=dict)
    details: Dict[str, str] = field(default_factory=dict)
    ratio: Optional[float] = None               # GC=F per 1 GLD
    call_wall: Optional[float] = None           # in GC=F terms
    put_wall: Optional[float] = None
    em_daily: Optional[float] = None            # $ gold, 1σ daily
    em_used: Optional[float] = None             # today's range ÷ em_daily
    atm_iv: Optional[float] = None
    pcr: Optional[float] = None
    skew_vol: Optional[float] = None            # call IV − put IV, vol pts
    gex: Optional[float] = None                 # $ per 1% move (GLD units)
    gvz: Optional[float] = None
    gvz_day_pct: Optional[float] = None
    gvz_pctile: Optional[float] = None
    strikes: Optional[pd.DataFrame] = None      # strike_gold, call_oi, put_oi
    flags: Dict[str, bool] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


def _clip(x, m=1.0):
    return float(max(-m, min(m, x)))


def compute(chain: Optional[Dict], gold: pd.DataFrame, gvz_intraday: Optional[pd.DataFrame],
            gvz_daily: Optional[pd.DataFrame], now: Optional[datetime] = None) -> OptReport:
    rep = OptReport()
    now = now or datetime.now(timezone.utc)
    P = cfg.OPT_PTS
    comp, det = {}, {}
    if gold is None or gold.empty:
        rep.error = "no gold data"
        return rep
    g_last = float(gold["Close"].iat[-1])
    gold_day = xm.day_change_pct(gold)

    if chain and chain.get("expiries") and chain.get("spot"):
        S = float(chain["spot"])
        ratio = g_last / S
        rep.ratio = ratio
        exps = chain["expiries"]

        # ATM IV + expected move (first expiry)
        first = exps[0]
        iv_c = _iv_near(_clean(first["calls"]), S)
        iv_p = _iv_near(_clean(first["puts"]), S)
        ivs = [v for v in (iv_c, iv_p) if v]
        if ivs:
            rep.atm_iv = float(np.mean(ivs))
            rep.em_daily = g_last * rep.atm_iv / math.sqrt(252)
            td = xs.trading_day(gold.index[-1].to_pydatetime())
            days = np.asarray(xs.trading_day_index(gold.index))
            today = gold[days == td]
            if len(today):
                rng = float(today["High"].max() - today["Low"].min())
                rep.em_used = rng / rep.em_daily if rep.em_daily else None
            rep.flags["em_exhausted"] = bool(rep.em_used is not None and
                                             rep.em_used >= cfg.OPT_EM_USED_FLAG)
            if rep.flags["em_exhausted"]:
                rep.notes.append(f"Today's range has used {rep.em_used:.0%} of the "
                                 "options-implied daily move — late entries are poor value")

        # Skew
        skew_exp = next((e for e in exps
                         if days_to_expiry(e["expiry"], now) >= cfg.OPT_SKEW_MIN_DAYS), None)
        if skew_exp:
            civ = _iv_near(_clean(skew_exp["calls"]), S * (1 + cfg.OPT_SKEW_OTM_PCT / 100))
            piv = _iv_near(_clean(skew_exp["puts"]), S * (1 - cfg.OPT_SKEW_OTM_PCT / 100))
            if civ and piv:
                rep.skew_vol = (civ - piv) * 100
                comp["Skew"] = round(P["Skew"] * _clip(rep.skew_vol / cfg.OPT_REF_SKEW_VOL), 1)
                det["Skew"] = f"call {civ * 100:.1f} vs put {piv * 100:.1f} IV → " \
                              f"{rep.skew_vol:+.1f} vol pts ({skew_exp['expiry']})"
        if "Skew" not in comp:
            comp["Skew"], det["Skew"] = None, "not enough liquid OTM strikes"

        # Strikes, PCR, walls
        tbl = strike_table(chain)
        lo, hi = S * (1 - cfg.OPT_WALL_RANGE_PCT / 100), S * (1 + cfg.OPT_WALL_RANGE_PCT / 100)
        tbl = tbl[(tbl["strike"] >= lo) & (tbl["strike"] <= hi)]
        tot_c, tot_p = tbl["call_oi"].sum(), tbl["put_oi"].sum()
        if tot_c > 0:
            rep.pcr = float(tot_p / tot_c)
            comp["Put/call"] = round(P["Put/call"] * _clip((1.0 - rep.pcr) / 0.5), 1)
            det["Put/call"] = f"OI put/call {rep.pcr:.2f} (nearest {cfg.OPT_EXPIRIES} expiries)"
        else:
            comp["Put/call"], det["Put/call"] = None, "no open interest"
        above, below = tbl[tbl["strike"] > S], tbl[tbl["strike"] < S]
        cw, pw = wall(above, "call_oi"), wall(below, "put_oi")
        rep.call_wall = None if cw is None else float(cw * ratio)
        rep.put_wall = None if pw is None else float(pw * ratio)
        t2 = tbl.copy()
        t2["strike_gold"] = t2["strike"] * ratio
        rep.strikes = t2
        if rep.em_daily and (cw is not None or pw is not None):
            em = rep.em_daily
            d_c = None if rep.call_wall is None else (rep.call_wall - g_last) / em
            d_p = None if rep.put_wall is None else (g_last - rep.put_wall) / em
            near = cfg.OPT_WALL_NEAR_EM
            rep.flags["near_call_wall"] = bool(d_c is not None and d_c <= near)
            rep.flags["near_put_wall"] = bool(d_p is not None and d_p <= near)
            if rep.flags["near_call_wall"] and rep.flags["near_put_wall"]:
                s = 0.0
                rep.notes.append(f"Pinned between the put wall {rep.put_wall:,.0f} and call "
                                 f"wall {rep.call_wall:,.0f} — expect chop")
            elif rep.flags["near_call_wall"]:
                s = -P["Walls"]
                rep.notes.append(f"Price is pressing the call wall {rep.call_wall:,.0f} — "
                                 "resistance / pin risk")
            elif rep.flags["near_put_wall"]:
                s = P["Walls"]
                rep.notes.append(f"Price is sitting on the put wall {rep.put_wall:,.0f} — support")
            elif d_c is not None and d_p is not None:
                s = 0.5 * P["Walls"] * _clip((d_c - d_p) / (d_c + d_p))
            else:
                s = 0.0
            comp["Walls"] = round(s, 1)
            det["Walls"] = " · ".join(x for x in (
                None if rep.call_wall is None else f"call wall {rep.call_wall:,.0f} "
                f"({d_c:+.1f} EM)",
                None if rep.put_wall is None else f"put wall {rep.put_wall:,.0f} "
                f"(−{d_p:.1f} EM)") if x)
        else:
            comp["Walls"], det["Walls"] = None, "walls need OI and an expected move"

        try:
            rep.gex = dealer_gex(chain, now)
            if rep.gex > 0:
                rep.notes.append("Dealers net long gamma — moves tend to be dampened "
                                 "(mean-revert)")
            elif rep.gex < 0:
                rep.notes.append("Dealers net short gamma — moves tend to be amplified")
        except Exception:  # noqa: BLE001
            rep.gex = None
    else:
        for k in ("Skew", "Put/call", "Walls"):
            comp[k], det[k] = None, "GLD options chain unavailable"

    # GVZ
    if gvz_intraday is not None and not gvz_intraday.empty:
        rep.gvz = float(gvz_intraday["Close"].iat[-1])
        rep.gvz_day_pct = xm.day_change_pct(gvz_intraday)
    if gvz_daily is not None and len(gvz_daily) >= 60:
        lvl = rep.gvz if rep.gvz is not None else float(gvz_daily["Close"].iat[-1])
        rep.gvz = lvl
        rep.gvz_pctile = float((gvz_daily["Close"] < lvl).mean() * 100)
    if rep.gvz_day_pct is not None and gold_day is not None:
        v, g = rep.gvz_day_pct, gold_day
        if v >= cfg.GVZ_SPIKE_PCT:
            s = P["Vol"] if g > 0 else -P["Vol"]
            tag = "upside chase" if g > 0 else "fear selling"
        elif v <= -cfg.GVZ_SPIKE_PCT and g > 0:
            s, tag = 1.0, "calm grind higher"
        elif v <= -cfg.GVZ_SPIKE_PCT and g < 0:
            s, tag = -1.0, "calm drift lower"
        else:
            s, tag = 0.0, "no vol signal"
        comp["Vol"] = s
        det["Vol"] = f"GVZ {rep.gvz:.1f} ({v:+.1f}%)" + \
            ("" if rep.gvz_pctile is None else f", {rep.gvz_pctile:.0f}th pct 1y") + f" — {tag}"
    else:
        comp["Vol"], det["Vol"] = None, "GVZ unavailable"

    present = {k: v for k, v in comp.items() if v is not None}
    rep.components, rep.details = comp, det
    if not present:
        rep.error = "no options or volatility data reachable"
        return rep
    rep.score = round(_clip(sum(present.values()), cfg.OPTIONS_MAX), 1)
    cut = cfg.MACRO_BIAS_FRAC * cfg.OPTIONS_MAX
    rep.bias = "LONG" if rep.score >= cut else "SHORT" if rep.score <= -cut else "NEUTRAL"
    rep.ok = True
    return rep


def get_options_report(gold: pd.DataFrame, gvz_intraday: Optional[pd.DataFrame] = None,
                       chain: Optional[Dict] = None, gvz_daily: Optional[pd.DataFrame] = None,
                       fetch: bool = True, now: Optional[datetime] = None) -> OptReport:
    try:
        if fetch:
            chain = chain if chain is not None else fetch_chain()
            gvz_daily = gvz_daily if gvz_daily is not None else fetch_gvz_daily()
        return compute(chain, gold, gvz_intraday, gvz_daily, now)
    except Exception as e:  # noqa: BLE001
        return OptReport(error=f"{type(e).__name__}: {e}")
