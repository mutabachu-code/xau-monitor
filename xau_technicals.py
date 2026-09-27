"""
xau_technicals.py — Layer 5: technicals on 15m GC=F (score ±25).

Components (max points):
  VWAP position + slope ±6 · EMA stack ±5 · 1h HTF trend ±4 ·
  market structure ±5 · RSI regime/divergence/decay ±3 · CPR position ±2
ADX below ADX_CHOP damps the total (chop), ATR regime is reported for sizing.

No lookahead: swings are only used once confirmed (SWING_N bars later), CPR
for a day comes from the prior trading day, VWAP is cumulative within the day.
get_tech_report() never raises; failures return ok=False with the error.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import xau_config as cfg
import xau_sessions as xs


# ── Primitive indicators ─────────────────────────────────────────────────────
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False).mean()


def rsi(close: pd.Series, n: int = cfg.RSI_N) -> pd.Series:
    d = close.diff()
    up = wilder(d.clip(lower=0), n)
    dn = wilder((-d).clip(lower=0), n)
    out = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    out = out.where(dn != 0, 100.0)
    out = out.where(~((up == 0) & (dn == 0)), 50.0)
    return out


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["Close"].shift()
    return pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(),
                      (df["Low"] - pc).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int = cfg.ATR_N) -> pd.Series:
    return wilder(true_range(df), n)


def adx(df: pd.DataFrame, n: int = cfg.ADX_N) -> pd.DataFrame:
    up, dn = df["High"].diff(), -df["Low"].diff()
    plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    tr_ = wilder(true_range(df), n).replace(0, np.nan)
    pdi, mdi = 100 * wilder(plus, n) / tr_, 100 * wilder(minus, n) / tr_
    dx = (100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)).fillna(0)
    return pd.DataFrame({"adx": wilder(dx, n), "pdi": pdi, "mdi": mdi})


def session_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """VWAP anchored at each CME trading day (18:00 ET) with σ bands.
    Bars with no volume so far in the day fall back to an equal-weight mean."""
    g = np.asarray(xs.trading_day_index(df.index))
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    v = df["Volume"].fillna(0).clip(lower=0)
    cv = v.groupby(g).cumsum()
    vw = (tp * v).groupby(g).cumsum() / cv.replace(0, np.nan)
    var = (tp ** 2 * v).groupby(g).cumsum() / cv.replace(0, np.nan) - vw ** 2
    n = tp.groupby(g).cumcount() + 1
    eq = tp.groupby(g).cumsum() / n
    eq_var = (tp ** 2).groupby(g).cumsum() / n - eq ** 2
    use_eq = cv <= 0
    vwap = vw.where(~use_eq, eq)
    sd = np.sqrt(var.where(~use_eq, eq_var).clip(lower=0))
    return pd.DataFrame({"vwap": vwap, "sd": sd})


# ── Swings and structure ─────────────────────────────────────────────────────
def swing_points(df: pd.DataFrame, n: int = cfg.SWING_N) -> Dict[str, List[int]]:
    """Fractal swing highs/lows (positional indexes). A swing at i is only
    known at bar i+n, callers must respect that confirmation delay."""
    h, l = df["High"].to_numpy(), df["Low"].to_numpy()
    highs, lows = [], []
    for i in range(n, len(df) - n):
        win_h, win_l = h[i - n:i + n + 1], h[i - n:i]
        if h[i] == win_h.max() and h[i] > win_l.max():
            highs.append(i)
        win_l2, left_l = l[i - n:i + n + 1], l[i - n:i]
        if l[i] == win_l2.min() and l[i] < left_l.min():
            lows.append(i)
    return {"highs": highs, "lows": lows}


def market_structure(df: pd.DataFrame, n: int = cfg.SWING_N) -> Dict:
    """Walk closes; a close beyond the last confirmed swing is a break.
    Break in trend direction = BoS, against it = MSS (change of character)."""
    sw = swing_points(df, n)
    close = df["Close"].to_numpy()
    conf_h = {i + n: i for i in sw["highs"]}
    conf_l = {i + n: i for i in sw["lows"]}
    last_h = last_l = None
    trend, events = "none", []
    for i in range(len(df)):
        if i in conf_h:
            last_h = conf_h[i]
        if i in conf_l:
            last_l = conf_l[i]
        hv = df["High"].iat[last_h] if last_h is not None else None
        lv = df["Low"].iat[last_l] if last_l is not None else None
        if hv is not None and close[i] > hv:
            kind = "BoS" if trend == "up" else "MSS"
            events.append({"i": i, "time": df.index[i], "dir": "up", "kind": kind,
                           "level": float(hv)})
            trend, last_h = "up", None
        elif lv is not None and close[i] < lv:
            kind = "BoS" if trend == "down" else "MSS"
            events.append({"i": i, "time": df.index[i], "dir": "down", "kind": kind,
                           "level": float(lv)})
            trend, last_l = "down", None
    last = events[-1] if events else None
    return {"trend": trend, "last": last, "events": events[-10:], "swings": sw,
            "bars_since": (len(df) - 1 - last["i"]) if last else None}


def rsi_divergence(df: pd.DataFrame, r: pd.Series, sw: Dict,
                   lookback: int = cfg.DIVERGENCE_LOOKBACK,
                   n: int = cfg.SWING_N) -> Optional[str]:
    """'bearish' / 'bullish' from the last two confirmed swings in lookback."""
    last_i = len(df) - 1
    ok = lambda i: i + n <= last_i and i >= last_i - lookback
    hs = [i for i in sw["highs"] if ok(i)][-2:]
    ls = [i for i in sw["lows"] if ok(i)][-2:]
    bear = bull = False
    if len(hs) == 2:
        a, b = hs
        bear = df["High"].iat[b] > df["High"].iat[a] and r.iat[b] < r.iat[a]
    if len(ls) == 2:
        a, b = ls
        bull = df["Low"].iat[b] < df["Low"].iat[a] and r.iat[b] > r.iat[a]
    if bear and bull:
        return "bearish" if max(hs) > max(ls) else "bullish"
    return "bearish" if bear else "bullish" if bull else None


def momentum_decay(close: pd.Series, r: pd.Series, bars: int = cfg.RSI_DECAY_BARS,
                   drop: float = cfg.RSI_DECAY_DROP) -> Optional[str]:
    """Price still pushing but RSI fading over the last `bars` bars."""
    if len(close) <= bars:
        return None
    dp = close.iat[-1] - close.iat[-1 - bars]
    dr = r.iat[-1] - r.iat[-1 - bars]
    if dp > 0 and dr <= -drop:
        return "bullish fading"
    if dp < 0 and dr >= drop:
        return "bearish fading"
    return None


# ── CPR ──────────────────────────────────────────────────────────────────────
def daily_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    g = np.asarray(xs.trading_day_index(df.index))
    d = df.groupby(g).agg(Open=("Open", "first"), High=("High", "max"),
                          Low=("Low", "min"), Close=("Close", "last"))
    d.index.name = "trading_day"
    return d


def cpr_from(h: float, l: float, c: float) -> Dict[str, float]:
    p = (h + l + c) / 3
    bc = (h + l) / 2
    tc = 2 * p - bc
    tc, bc = max(tc, bc), min(tc, bc)
    rng = h - l
    return {"P": p, "TC": tc, "BC": bc,
            "R1": 2 * p - l, "S1": 2 * p - h, "R2": p + rng, "S2": p - rng,
            "R3": h + 2 * (p - l), "S3": l - 2 * (h - p),
            "width_pct": (tc - bc) / p * 100 if p else 0.0}


def cpr_table(daily: pd.DataFrame) -> pd.DataFrame:
    """CPR applying to each trading day (built from the prior day's HLC),
    with width class and whether that day left its CPR untouched (virgin)."""
    rows = {}
    days = list(daily.index)
    for k in range(1, len(days)):
        prev, day = daily.iloc[k - 1], daily.iloc[k]
        c = cpr_from(prev.High, prev.Low, prev.Close)
        c["virgin"] = bool(day.Low > c["TC"] or day.High < c["BC"])
        rows[days[k]] = c
    t = pd.DataFrame.from_dict(rows, orient="index")
    if t.empty:
        return t
    hist = t["width_pct"].shift(1).rolling(cfg.CPR_LOOKBACK_DAYS, min_periods=5)
    q25, q75 = hist.quantile(0.25), hist.quantile(0.75)
    narrow = q25.fillna(cfg.CPR_NARROW_PCT)
    wide = q75.fillna(cfg.CPR_WIDE_PCT)
    t["width_class"] = np.where(t["width_pct"] <= narrow, "narrow",
                                np.where(t["width_pct"] >= wide, "wide", "moderate"))
    return t


# ── CPR read (display + setup, same rules as the NAS100 scalping engine) ─────
CPR_ZONE_MULT = 0.3          # entry zone = level ± CPR width × 0.3
CPR_AT_LEVEL_ATR = 0.25      # "at R1/S1" = within 0.25 ATR (or beyond)
CPR_TYPE_BIAS = {
    "narrow": "Trending day likely — strong directional move expected",
    "moderate": "Mixed day — watch for breakout direction from CPR",
    "wide": "Sideways/choppy day likely — fade extremes, avoid breakouts",
}


def cpr_relationship(today: Dict, prev: Optional[Dict]) -> str:
    """Today's CPR against yesterday's (two-day relationship)."""
    if not prev:
        return "n/a"
    t_tc, t_bc, p_tc, p_bc = today["TC"], today["BC"], prev["TC"], prev["BC"]
    if t_bc > p_tc:
        return "higher value (bullish)"
    if t_tc < p_bc:
        return "lower value (bearish)"
    if t_tc <= p_tc and t_bc >= p_bc:
        return "inside (breakout likely)"
    if t_tc >= p_tc and t_bc <= p_bc:
        return "outside (range likely)"
    return "overlapping higher (mild bullish)" if t_tc > p_tc else \
        "overlapping lower (mild bearish)"


def cpr_flip(today: pd.DataFrame, cp: Dict) -> Dict:
    """Most recent close-through of TC (up) or BC (down) today, and whether it held."""
    if today is None or len(today) < 2:
        return {"flip": None, "flip_time": None}
    c = today["Close"].to_numpy()
    tc, bc = cp["TC"], cp["BC"]
    last = None
    for i in range(1, len(c)):
        if c[i - 1] <= tc < c[i]:
            last = ("TC", "up", i)
        elif c[i - 1] >= bc > c[i]:
            last = ("BC", "down", i)
        elif c[i - 1] > tc >= c[i]:
            last = ("TC", "lost", i)
        elif c[i - 1] < bc <= c[i]:
            last = ("BC", "reclaimed", i)
    if last is None:
        return {"flip": None, "flip_time": None}
    lvl, how, i = last
    t = today.index[i]
    text = {"up": "TC flipped to support", "down": "BC flipped to resistance",
            "lost": "lost TC — flip failed, back inside", "reclaimed":
            "reclaimed BC — flip failed, back inside"}[how]
    return {"flip": text, "flip_time": t}


def cpr_setup(cp: Dict, price: float, atr: float, virgin_today: bool) -> Dict:
    """Setup per CPR width, mirroring the NAS100 rules."""
    w = cp["TC"] - cp["BC"]
    z = max(w * CPR_ZONE_MULT, 0.1 * atr)
    pos = "ABOVE_TC" if price > cp["TC"] else "BELOW_BC" if price < cp["BC"] else "INSIDE"
    kind = cp.get("width_class", "moderate")
    at = CPR_AT_LEVEL_ATR * atr
    setup = {"direction": None, "zone": None, "target": None, "invalid": None, "text": ""}
    if kind == "narrow":
        if pos == "ABOVE_TC":
            setup.update(direction="BUY", zone=(cp["TC"] - z, cp["TC"] + z), target=cp["R1"],
                         invalid=cp["BC"], text="Narrow CPR, above TC — buy dips to TC, "
                         "target R1, invalid below BC")
        elif pos == "BELOW_BC":
            setup.update(direction="SELL", zone=(cp["BC"] - z, cp["BC"] + z), target=cp["S1"],
                         invalid=cp["TC"], text="Narrow CPR, below BC — sell bounces to BC, "
                         "target S1, invalid above TC")
        else:
            setup["text"] = "Narrow CPR, inside — wait for the TC/BC break; expect a trend"
    elif kind == "wide":
        if price >= cp["R1"] - at:
            setup.update(direction="SELL", zone=(cp["R1"] - at, cp["R1"] + at), target=cp["P"],
                         invalid=cp["R2"], text="Wide CPR, at R1 — fade with a sell to P; "
                         "avoid new longs")
        elif price <= cp["S1"] + at:
            setup.update(direction="BUY", zone=(cp["S1"] - at, cp["S1"] + at), target=cp["P"],
                         invalid=cp["S2"], text="Wide CPR, at S1 — fade with a buy to P; "
                         "avoid new shorts")
        else:
            setup["text"] = "Wide CPR — range day; fade R1/S1, avoid breakouts"
    else:
        if pos == "INSIDE":
            setup["text"] = "Moderate CPR, inside — wait for TC/BC breakout confirmation"
        elif pos == "ABOVE_TC":
            setup["text"] = "Moderate CPR, above TC — breakout up; lean long on holds of TC"
        else:
            setup["text"] = "Moderate CPR, below BC — breakdown; lean short on rejections of BC"
    if virgin_today:
        setup["text"] += ". CPR untouched today (virgin) — price magnet to P"
    return {"price_vs_cpr": pos, "setup": setup, "width_abs": w}


# ── HTF ──────────────────────────────────────────────────────────────────────
def resample_1h(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("1h", label="left", closed="left").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last",
         "Volume": "sum"}).dropna(subset=["Close"])


def htf_trend(df: pd.DataFrame) -> Dict:
    h = resample_1h(df)
    if len(h) < cfg.HTF_EMA_FAST:
        return {"trend": "unknown", "close": None, "ema_fast": None, "ema_slow": None}
    ef, es = ema(h["Close"], cfg.HTF_EMA_FAST), ema(h["Close"], cfg.HTF_EMA_SLOW)
    c, f, s = h["Close"].iat[-1], ef.iat[-1], es.iat[-1]
    if c > f > s:
        t = "up"
    elif c < f < s:
        t = "down"
    elif c > f:
        t = "up-weak"
    elif c < f:
        t = "down-weak"
    else:
        t = "mixed"
    return {"trend": t, "close": float(c), "ema_fast": float(f), "ema_slow": float(s),
            "reliable": len(h) >= cfg.HTF_EMA_SLOW}


# ── Report ───────────────────────────────────────────────────────────────────
@dataclass
class TechReport:
    ok: bool = False
    error: str = ""
    score: float = 0.0
    bias: str = "NEUTRAL"
    components: Dict[str, float] = field(default_factory=dict)
    details: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    price: Optional[float] = None
    vwap: Optional[float] = None
    vwap_sd: Optional[float] = None
    vwap_z: Optional[float] = None
    rsi: Optional[float] = None
    atr: Optional[float] = None
    atr_ratio: Optional[float] = None
    atr_regime: str = "unknown"
    daily_atr: Optional[float] = None
    adx: Optional[float] = None
    chop: bool = False
    htf: Dict = field(default_factory=dict)
    structure: Dict = field(default_factory=dict)
    divergence: Optional[str] = None
    decay: Optional[str] = None
    cpr: Dict = field(default_factory=dict)
    frame: Optional[pd.DataFrame] = None     # series for charting


def _clamp(x, m):
    return float(max(-m, min(m, x)))


def _atr_regime(ratio):
    for cut, name in cfg.ATR_REGIMES:
        if ratio < cut:
            return name
    return "extreme"


def compute(df: pd.DataFrame) -> TechReport:
    if df is None or len(df) < max(cfg.EMA_SLOW, 60):
        return TechReport(error=f"need ≥60 bars, have {0 if df is None else len(df)}")
    rep = TechReport(ok=True)
    c = df["Close"]
    price = float(c.iat[-1])
    rep.price = price

    vw = session_vwap(df)
    e9, e21, e50 = ema(c, cfg.EMA_FAST), ema(c, cfg.EMA_MID), ema(c, cfg.EMA_SLOW)
    r = rsi(c)
    a = atr(df)
    ad = adx(df)
    frame = pd.DataFrame({"vwap": vw["vwap"], "sd": vw["sd"], "ema9": e9,
                          "ema21": e21, "ema50": e50, "rsi": r, "atr": a,
                          "adx": ad["adx"]})
    rep.frame = frame

    rep.vwap, rep.vwap_sd = float(vw["vwap"].iat[-1]), float(vw["sd"].iat[-1])
    rep.vwap_z = (price - rep.vwap) / rep.vwap_sd if rep.vwap_sd > 0 else 0.0
    rep.rsi, rep.atr, rep.adx = float(r.iat[-1]), float(a.iat[-1]), float(ad["adx"].iat[-1])
    base = a.iloc[-cfg.ATR_REGIME_BARS:].median()
    rep.atr_ratio = float(rep.atr / base) if base and base > 0 else 1.0
    rep.atr_regime = _atr_regime(rep.atr_ratio)
    daily = daily_ohlc(df)
    rep.daily_atr = float((daily["High"] - daily["Low"]).iloc[-15:-1].mean()) \
        if len(daily) > 2 else None
    rep.chop = rep.adx < cfg.ADX_CHOP

    comp, det = {}, {}

    # VWAP (±6)
    slope_bars = min(cfg.VWAP_SLOPE_BARS, len(vw) - 1)
    slope = vw["vwap"].iat[-1] - vw["vwap"].iat[-1 - slope_bars]
    side = 1 if price > rep.vwap else -1 if price < rep.vwap else 0
    sl = 1 if slope > 0 else -1 if slope < 0 else 0
    v = 3 * side + (2 * sl if sl == side else 0)
    if side and sl and sl != side:
        v -= sl  # slope against position trims it
    if abs(rep.vwap_z) > cfg.VWAP_STRETCH_SIGMA:
        v *= 0.5
        rep.notes.append(f"Price {rep.vwap_z:+.1f}σ from VWAP — stretched, "
                         "mean-reversion risk")
    comp["VWAP"] = _clamp(v, 6)
    det["VWAP"] = f"{'above' if side > 0 else 'below' if side < 0 else 'at'} VWAP " \
                  f"{rep.vwap:,.2f} ({rep.vwap_z:+.1f}σ), slope {'up' if sl > 0 else 'down' if sl < 0 else 'flat'}"

    # EMA stack (±5)
    f9, f21, f50 = e9.iat[-1], e21.iat[-1], e50.iat[-1]
    if price > f9 > f21 > f50:
        s = 5
    elif price < f9 < f21 < f50:
        s = -5
    elif f9 > f21 > f50:
        s = 3
    elif f9 < f21 < f50:
        s = -3
    else:
        s = 1.5 if price > f21 else -1.5 if price < f21 else 0
    comp["EMA stack"] = s
    det["EMA stack"] = f"9 {f9:,.1f} · 21 {f21:,.1f} · 50 {f50:,.1f}"

    # HTF (±4)
    rep.htf = htf_trend(df)
    comp["1h trend"] = {"up": 4, "down": -4, "up-weak": 2, "down-weak": -2}.get(
        rep.htf["trend"], 0)
    det["1h trend"] = rep.htf["trend"] + ("" if rep.htf.get("reliable", True)
                                          else " (EMA200 warming up)")

    # Structure (±5)
    st = market_structure(df.iloc[-400:])
    rep.structure = st
    s = {"up": 3, "down": -3}.get(st["trend"], 0)
    if st["last"] is not None and st["bars_since"] <= cfg.STRUCTURE_RECENT_BARS:
        s += 2 if st["last"]["dir"] == "up" else -2
    comp["Structure"] = _clamp(s, 5)
    det["Structure"] = "no breaks" if st["last"] is None else \
        f"{st['last']['kind']} {st['last']['dir']} at {st['last']['level']:,.2f}, " \
        f"{st['bars_since']} bars ago"

    # RSI (±3)
    sub = df.iloc[-(cfg.DIVERGENCE_LOOKBACK + 10):]
    rep.divergence = rsi_divergence(sub, r.loc[sub.index], swing_points(sub))
    rep.decay = momentum_decay(c, r)
    s = 2 if rep.rsi >= 55 else -2 if rep.rsi <= 45 else 0
    if rep.divergence == "bearish":
        s -= 2
    elif rep.divergence == "bullish":
        s += 2
    if rep.decay == "bullish fading":
        s -= 1
    elif rep.decay == "bearish fading":
        s += 1
    comp["RSI"] = _clamp(s, 3)
    extras = [x for x in (rep.divergence and f"{rep.divergence} divergence", rep.decay) if x]
    det["RSI"] = f"{rep.rsi:.1f}" + (f" — {', '.join(extras)}" if extras else "")
    if rep.divergence:
        rep.notes.append(f"RSI {rep.divergence} divergence on last swings")
    if rep.decay:
        rep.notes.append(f"RSI momentum decay: {rep.decay}")

    # CPR (±2)
    ct = cpr_table(daily)
    td = daily.index[-1]
    if not ct.empty and td in ct.index:
        cp = ct.loc[td].to_dict()
        prev_virgin = [d for d in ct.index if d < td and ct.loc[d, "virgin"]]
        cp["virgin_prior"] = None
        if prev_virgin and prev_virgin[-1] == ct.index[ct.index.get_loc(td) - 1]:
            pv = ct.loc[prev_virgin[-1]]
            cp["virgin_prior"] = {"TC": float(pv["TC"]), "BC": float(pv["BC"])}
            rep.notes.append(f"Yesterday's CPR {pv['BC']:,.2f}–{pv['TC']:,.2f} "
                             "was never traded (virgin) — magnet level")
        rep.cpr = cp
        try:                                   # display extras; never affects the score
            days_arr = np.asarray(xs.trading_day_index(df.index))
            today_bars = df[days_arr == td]
            prev_row = ct.iloc[ct.index.get_loc(td) - 1].to_dict() \
                if ct.index.get_loc(td) > 0 else None
            cp["virgin_today"] = bool(cp.get("virgin"))
            cp["type_bias"] = CPR_TYPE_BIAS.get(cp["width_class"], "")
            cp["relationship"] = cpr_relationship(cp, prev_row)
            cp.update(cpr_flip(today_bars, cp))
            cp.update(cpr_setup(cp, price, float(a.iat[-1]), cp["virgin_today"]))
        except Exception:  # noqa: BLE001
            pass
        s = 2 if price > cp["TC"] else -2 if price < cp["BC"] else 0
        comp["CPR"] = s
        det["CPR"] = f"{cp['width_class']} ({cp['width_pct']:.2f}%), price " + \
            ("above TC" if s > 0 else "below BC" if s < 0 else "inside")
        if cp["width_class"] == "narrow":
            rep.notes.append("Narrow CPR — trend-day odds higher")
        elif cp["width_class"] == "wide":
            rep.notes.append("Wide CPR — range/mean-reversion day more likely")
    else:
        comp["CPR"] = 0
        det["CPR"] = "needs a prior trading day"

    total = sum(comp.values())
    if rep.chop:
        total *= cfg.CHOP_DAMPING
        rep.notes.append(f"ADX {rep.adx:.0f} < {cfg.ADX_CHOP} — chop, score damped "
                         f"×{cfg.CHOP_DAMPING}")
    if rep.atr_regime in ("expanding", "extreme"):
        rep.notes.append(f"ATR {rep.atr_regime} ({rep.atr_ratio:.1f}× normal) — "
                         "widen stops, cut size")
    rep.components, rep.details = comp, det
    rep.score = round(_clamp(total, cfg.TECH_MAX), 1)
    t = cfg.TECH_BIAS_THRESHOLD
    rep.bias = "LONG" if rep.score >= t else "SHORT" if rep.score <= -t else "NEUTRAL"
    return rep


def get_tech_report(df: pd.DataFrame) -> TechReport:
    try:
        return compute(df)
    except Exception as e:  # noqa: BLE001
        return TechReport(error=f"{type(e).__name__}: {e}")
