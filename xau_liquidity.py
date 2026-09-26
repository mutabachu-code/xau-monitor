"""
xau_liquidity.py — Layer 6: liquidity levels and sweeps (score ±15).

Liquidity pools: Asian high/low, previous day high/low (PDH/PDL), previous
week high/low (PWH/PWL), equal highs/lows (EQH/EQL) and $50 round numbers.

A sweep = a bar pierces a level by ≥ SWEEP_EPS_ATR·ATR and a close back on the
original side happens within SWEEP_RECLAIM_BARS. Sweeping highs (buy-side
liquidity) is bearish, sweeping lows (sell-side) is bullish. The most recent
valid sweep is scored by level weight, session, time-of-day RVOL, follow-
through and age; a later close beyond the sweep extreme invalidates it.
get_liq_report() never raises.
"""
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import xau_config as cfg
import xau_sessions as xs
import xau_technicals as xt


# ── RVOL ─────────────────────────────────────────────────────────────────────
def time_of_day_rvol(df: pd.DataFrame, days: int = cfg.RVOL_DAYS) -> pd.Series:
    """Bar volume ÷ mean volume of the same 15m slot over prior trading days.
    Falls back to a 20-bar rolling mean where slot history is too thin."""
    v = df["Volume"].astype(float)
    slot = df.index.hour * 60 + df.index.minute
    base = v.groupby(slot).transform(
        lambda s: s.shift(1).rolling(days, min_periods=5).mean())
    fb = v.shift(1).rolling(20, min_periods=5).mean()
    base = base.where(base > 0, fb)
    return (v / base.replace(0, np.nan)).fillna(1.0)


# ── Levels ───────────────────────────────────────────────────────────────────
@dataclass
class Level:
    name: str
    price: float
    side: str               # "buy" = resting above (highs), "sell" = below (lows)
    valid_from: pd.Timestamp  # level exists only from this time
    weight: float = 1.0


def _prev_week(df: pd.DataFrame, td) -> Optional[Dict]:
    days = pd.Series(np.asarray(xs.trading_day_index(df.index)), index=df.index)
    week = pd.Series([d.isocalendar()[:2] for d in days], index=df.index)
    cur = td.isocalendar()[:2]
    prior = sorted({w for w in week if w < cur})
    if not prior:
        return None
    sub = df[week == prior[-1]]
    return {"high": float(sub["High"].max()), "low": float(sub["Low"].min())}


def equal_levels(df: pd.DataFrame, atr_val: float,
                 lookback: int = cfg.EQ_LOOKBACK_BARS,
                 n: int = cfg.SWING_N) -> List[Level]:
    """Clusters of ≥2 confirmed swing highs (lows) within tolerance."""
    sub = df.iloc[-lookback:]
    sw = xt.swing_points(sub, n)
    tol = max(cfg.EQ_TOL_ATR * atr_val, cfg.EQ_MIN_TOL)
    out = []
    for key, side, name, col in (("highs", "buy", "EQH", "High"),
                                 ("lows", "sell", "EQL", "Low")):
        pts = [(i, float(sub[col].iat[i])) for i in sw[key] if i + n < len(sub)]
        used = set()
        for a in range(len(pts)):
            if a in used:
                continue
            cluster = [pts[a]] + [pts[b] for b in range(a + 1, len(pts))
                                  if b not in used and abs(pts[b][1] - pts[a][1]) <= tol]
            if len(cluster) >= 2:
                used.update(b for b in range(a, len(pts)) if pts[b] in cluster)
                price = max(p for _, p in cluster) if side == "buy" \
                    else min(p for _, p in cluster)
                last_i = max(i for i, _ in cluster)
                out.append(Level(name, price, side, sub.index[last_i + n],
                                 cfg.LEVEL_WEIGHTS[name]))
    return out


def build_levels(df: pd.DataFrame, atr_val: float) -> List[Level]:
    td = xs.trading_day(df.index[-1].to_pydatetime())
    day_start = pd.Timestamp(xs.asian_window(td)["start"])
    early = day_start              # prior-day/week levels are today's liquidity
    lv: List[Level] = []
    pdl = xs.prev_day_levels(df, td)
    if pdl:
        lv += [Level("PDH", pdl["high"], "buy", early, cfg.LEVEL_WEIGHTS["PDH"]),
               Level("PDL", pdl["low"], "sell", early, cfg.LEVEL_WEIGHTS["PDL"])]
    pw = _prev_week(df, td)
    if pw:
        lv += [Level("PWH", pw["high"], "buy", early, cfg.LEVEL_WEIGHTS["PWH"]),
               Level("PWL", pw["low"], "sell", early, cfg.LEVEL_WEIGHTS["PWL"])]
    asia = xs.asian_range(df, td)
    if asia and df.index[-1] >= asia["end"]:
        t0 = pd.Timestamp(asia["end"])
        lv += [Level("Asia H", asia["high"], "buy", t0, cfg.LEVEL_WEIGHTS["Asia H"]),
               Level("Asia L", asia["low"], "sell", t0, cfg.LEVEL_WEIGHTS["Asia L"])]
    lv += equal_levels(df, atr_val)
    price = float(df["Close"].iat[-1])
    for r in xs.round_levels(price, cfg.ROUND_NUMBER_STEP, 2):
        lv.append(Level("Round", r, "buy" if r > price else "sell", day_start,
                        cfg.LEVEL_WEIGHTS["Round"]))
    return lv


# ── Sweeps ───────────────────────────────────────────────────────────────────
def detect_sweeps(df: pd.DataFrame, levels: List[Level], atr_val: float,
                  rvol: pd.Series, scan: int = cfg.SWEEP_SCAN_BARS) -> List[Dict]:
    """Sweeps in the last `scan` bars, oldest first, one per level (first hit)."""
    eps = cfg.SWEEP_EPS_ATR * atr_val
    start = max(0, len(df) - scan)
    H, L, C = (df[k].to_numpy() for k in ("High", "Low", "Close"))
    out = []
    for lv in levels:
        for i in range(start, len(df)):
            t = df.index[i]
            if t < lv.valid_from:
                continue
            # level must still be untaken: no close beyond it since it became valid
            since = (df.index >= lv.valid_from) & (np.arange(len(df)) < i)
            prior = C[since]
            if lv.side == "buy" and np.any(prior > lv.price):
                break
            if lv.side == "sell" and np.any(prior < lv.price):
                break
            if lv.side == "buy":
                pierced = H[i] >= lv.price + eps
            else:
                pierced = L[i] <= lv.price - eps
            if not pierced:
                continue
            reclaim_at = None
            for j in range(i, min(i + cfg.SWEEP_RECLAIM_BARS + 1, len(df))):
                back = C[j] < lv.price if lv.side == "buy" else C[j] > lv.price
                if back:
                    reclaim_at = j
                    break
            if reclaim_at is not None:
                ext_end = reclaim_at + 1
                extreme = float(H[i:ext_end].max() if lv.side == "buy" else L[i:ext_end].min())
                out.append({"level": lv.name, "price": lv.price, "side": lv.side,
                            "dir": "bearish" if lv.side == "buy" else "bullish",
                            "i": i, "reclaim_i": reclaim_at, "time": t,
                            "extreme": extreme, "weight": lv.weight,
                            "rvol": float(rvol.iat[i]),
                            "session": xs.session_label(t.to_pydatetime())})
            break  # only the first interaction with each level counts
    return sorted(out, key=lambda s: (s["reclaim_i"], s["i"]))


def score_sweep(sw: Dict, df: pd.DataFrame, atr_val: float) -> Dict:
    last = len(df) - 1
    C = df["Close"].to_numpy()
    after = C[sw["reclaim_i"] + 1:]
    sign = 1 if sw["dir"] == "bullish" else -1
    invalid = bool(np.any(after < sw["extreme"])) if sign > 0 \
        else bool(np.any(after > sw["extreme"]))
    age = last - sw["reclaim_i"]
    pts, parts = 0.0, []
    if not invalid and age <= cfg.SWEEP_VALID_BARS:
        pts = cfg.SWEEP_BASE_PTS * sw["weight"]
        parts.append(f"{sw['level']} ×{sw['weight']}")
        if sw["rvol"] >= cfg.RVOL_CONFIRM:
            pts += cfg.SWEEP_RVOL_PTS
            parts.append(f"RVOL {sw['rvol']:.1f}")
        move = (C[-1] - sw["price"]) * sign
        if move >= cfg.SWEEP_FOLLOW_ATR * atr_val:
            pts += cfg.SWEEP_FOLLOW_PTS
            parts.append("follow-through")
        mult = cfg.SESSION_MULT.get(sw["session"], 0.5)
        decay = 1 - 0.5 * age / max(cfg.SWEEP_VALID_BARS, 1)
        pts *= mult * decay
        parts.append(f"{sw['session']} ×{mult}, age {age} bars ×{decay:.2f}")
    return {**sw, "invalid": invalid, "age": age, "points": round(sign * pts, 1),
            "why": ", ".join(parts) if parts else
            ("invalidated — extreme broken" if invalid else "too old")}


# ── Report ───────────────────────────────────────────────────────────────────
@dataclass
class LiqReport:
    ok: bool = False
    error: str = ""
    score: float = 0.0
    bias: str = "NEUTRAL"
    atr: Optional[float] = None
    rvol_now: Optional[float] = None
    levels: List[Level] = field(default_factory=list)
    sweeps: List[Dict] = field(default_factory=list)
    active: Optional[Dict] = None
    targets: Dict[str, Optional[Dict]] = field(default_factory=dict)
    asian_state: str = "n/a"
    notes: List[str] = field(default_factory=list)


def _asian_state(df: pd.DataFrame, sweeps: List[Dict]) -> str:
    td = xs.trading_day(df.index[-1].to_pydatetime())
    asia = xs.asian_range(df, td)
    if not asia:
        return "n/a"
    if df.index[-1] < asia["end"]:
        return "forming"
    names = {s["level"] for s in sweeps}
    price = float(df["Close"].iat[-1])
    tag = "above" if price > asia["high"] else "below" if price < asia["low"] else "inside"
    swept = [n for n in ("Asia H", "Asia L") if n in names]
    return tag + (f" (swept {', '.join(swept)})" if swept else "")


def compute(df: pd.DataFrame) -> LiqReport:
    if df is None or len(df) < 60:
        return LiqReport(error="need ≥60 bars")
    rep = LiqReport(ok=True)
    a = float(xt.atr(df).iat[-1])
    rep.atr = a
    rvol = time_of_day_rvol(df)
    rep.rvol_now = float(rvol.iat[-1])
    rep.levels = build_levels(df, a)
    raw = detect_sweeps(df, rep.levels, a, rvol)
    rep.sweeps = [score_sweep(s, df, a) for s in raw]
    live = [s for s in rep.sweeps if s["points"] != 0]
    if live:
        rep.active = live[-1]
        # a same-direction sweep stacked on the active one adds a little
        extra = sum(0.25 * s["points"] for s in live[:-1]
                    if np.sign(s["points"]) == np.sign(rep.active["points"]))
        rep.score = round(max(-cfg.LIQ_MAX, min(cfg.LIQ_MAX,
                                                rep.active["points"] + extra)), 1)
        opposing = [s for s in live[:-1]
                    if np.sign(s["points"]) != np.sign(rep.active["points"])]
        if opposing:
            rep.notes.append("Sweeps on both sides recently — two-way liquidity grab")
    t = cfg.LIQ_BIAS_THRESHOLD
    rep.bias = "LONG" if rep.score >= t else "SHORT" if rep.score <= -t else "NEUTRAL"

    price = float(df["Close"].iat[-1])
    taken = {(s["level"], s["price"]) for s in rep.sweeps}
    hi = df["High"].iloc[-cfg.SWEEP_SCAN_BARS:].max()
    lo = df["Low"].iloc[-cfg.SWEEP_SCAN_BARS:].min()
    above = [l for l in rep.levels if l.price > max(price, hi)
             and (l.name, l.price) not in taken]
    below = [l for l in rep.levels if l.price < min(price, lo)
             and (l.name, l.price) not in taken]
    near = lambda ls, f: None if not ls else f(ls, key=lambda l: abs(l.price - price))
    for key, lv in (("above", near(above, min)), ("below", near(below, min))):
        rep.targets[key] = None if lv is None else {
            "name": lv.name, "price": lv.price, "dist": abs(lv.price - price),
            "atr_mult": abs(lv.price - price) / a if a else None}
    rep.asian_state = _asian_state(df, rep.sweeps)
    if rep.rvol_now >= cfg.RVOL_CONFIRM:
        rep.notes.append(f"Current bar RVOL {rep.rvol_now:.1f}× — participation high")
    return rep


def get_liq_report(df: pd.DataFrame) -> LiqReport:
    try:
        return compute(df)
    except Exception as e:  # noqa: BLE001
        return LiqReport(error=f"{type(e).__name__}: {e}")
