"""
xau_zones.py — multi-timeframe supply/demand, order blocks, FVGs and IFVGs.

Timeframes: 15m (native), 1h and 4h (resampled from 15m; 4h bins aligned to
the CME day). For each timeframe:

  Supply / demand (sd)  a small base candle (range ≤ 1 ATR) followed within
                        3 bars by a displacement of ≥ 1.5 ATR away from it.
                        Zone = base low–high.
  Order block (ob)      the same base when it is an opposite-colour candle and
                        the displacement broke the prior 20-bar high/low (BoS).
  Fair value gap (fvg)  3-candle gap: low[j] > high[j−2] (bullish) or
                        high[j] < low[j−2] (bearish), gap ≥ 0.1 ATR.
  Inverse FVG (ifvg)    an FVG that price closed straight through; it flips role
                        (a broken bullish FVG becomes resistance and vice versa).

State after creation: touches (separate visits), fresh (never revisited) and
broken (a close through the far edge → removed; a broken FVG becomes an IFVG).
Weights: timeframe (15m 1, 1h 2, 4h 3) × kind × freshness. Nothing here raises.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import xau_config as cfg
import xau_technicals as xt

SUPPORT, RESIST = "support", "resistance"
KIND_LABEL = {"sd": "S/D", "ob": "Order block", "fvg": "FVG", "ifvg": "IFVG"}


def resample(df: pd.DataFrame, rule: Optional[str]) -> pd.DataFrame:
    if rule is None:
        return df
    kw = {"offset": cfg.ZONE_4H_OFFSET} if rule == "4h" else {}
    return df.resample(rule, label="left", closed="left", **kw).agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last",
         "Volume": "sum"}).dropna(subset=["Close"])


def _track(side: str, lo: float, hi: float, start: int, L, H, C):
    """Touches and break index after `start` (exclusive)."""
    if start + 1 >= len(C):
        return 0, None
    seg_l, seg_h, seg_c = L[start + 1:], H[start + 1:], C[start + 1:]
    if side == SUPPORT:
        broken = np.nonzero(seg_c < lo)[0]
        inside = seg_l <= hi
    else:
        broken = np.nonzero(seg_c > hi)[0]
        inside = seg_h >= lo
    b = int(broken[0]) if len(broken) else None
    upto = b if b is not None else len(seg_c)
    ins = inside[:upto]
    touches = int(np.sum(ins & ~np.r_[False, ins[:-1]])) if upto else 0
    return touches, (None if b is None else start + 1 + b)


def _make(tf, kind, side, lo, hi, created, touches, a, tfw, ts) -> Dict:
    h = hi - lo
    maxh = cfg.ZONE_MAX_HEIGHT_ATR * a
    if h > maxh:                               # keep the proximal part of tall zones
        if side == SUPPORT:
            lo = hi - maxh
        else:
            hi = lo + maxh
    fresh = touches == 0
    fw = cfg.FRESH_BONUS if fresh else (cfg.TESTED_PENALTY if touches >= 2 else 1.0)
    return {"tf": tf, "kind": kind, "side": side, "lo": float(lo), "hi": float(hi),
            "created": ts, "touches": touches, "fresh": fresh,
            "weight": round(tfw * cfg.KIND_WEIGHT[kind] * fw, 2),
            "label": f"{tf} {KIND_LABEL[kind]}"}


def detect(df: pd.DataFrame, tf: str, tfw: float) -> List[Dict]:
    n = len(df)
    if n < 25:
        return []
    O, H, L, C = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
    A = xt.atr(df).to_numpy(float)
    A = np.where(np.isfinite(A) & (A > 0), A, np.nanmean(A))
    idx = df.index
    zones: List[Dict] = []

    # supply / demand and order blocks
    raw = []
    for i in range(2, n - 1):
        a = A[i]
        if H[i] - L[i] > cfg.BASE_MAX_ATR * a:
            continue
        w_end = min(n, i + 1 + cfg.DISP_BARS)
        fc = C[i + 1:w_end]
        if not len(fc):
            continue
        up, dn = fc.max() - H[i], L[i] - fc.min()
        prior_hi = H[max(0, i - 20):i].max() if i > 0 else H[i]
        prior_lo = L[max(0, i - 20):i].min() if i > 0 else L[i]
        if up >= cfg.DISP_ATR * a:
            k = i + 1 + int(np.argmax(fc))
            ob = C[i] < O[i] and H[i + 1:k + 1].max() > prior_hi
            raw.append(("ob" if ob else "sd", SUPPORT, L[i], H[i], i, k))
        if dn >= cfg.DISP_ATR * a:
            k = i + 1 + int(np.argmin(fc))
            ob = C[i] > O[i] and L[i + 1:k + 1].min() < prior_lo
            raw.append(("ob" if ob else "sd", RESIST, L[i], H[i], i, k))
    # one zone per move: keep the base closest to the displacement
    raw.sort(key=lambda r: (r[1], r[5], r[4]))
    kept = []
    for r in raw:
        if kept and kept[-1][1] == r[1] and kept[-1][5] == r[5]:
            kept[-1] = r
        else:
            kept.append(r)
    for kind, side, lo, hi, i, k in kept:
        touches, broken = _track(side, lo, hi, k, L, H, C)
        if broken is None:
            zones.append(_make(tf, kind, side, lo, hi, k, touches, A[i], tfw, idx[i]))

    # FVG / IFVG
    for j in range(2, n):
        a = A[j]
        if L[j] > H[j - 2] and L[j] - H[j - 2] >= cfg.FVG_MIN_ATR * a:
            side, lo, hi, flip = SUPPORT, H[j - 2], L[j], RESIST
        elif H[j] < L[j - 2] and L[j - 2] - H[j] >= cfg.FVG_MIN_ATR * a:
            side, lo, hi, flip = RESIST, H[j], L[j - 2], SUPPORT
        else:
            continue
        touches, broken = _track(side, lo, hi, j, L, H, C)
        if broken is None:
            zones.append(_make(tf, "fvg", side, lo, hi, j, touches, a, tfw, idx[j - 1]))
        else:                                             # inverse FVG from the break
            t2, b2 = _track(flip, lo, hi, broken, L, H, C)
            if b2 is None:
                zones.append(_make(tf, "ifvg", flip, lo, hi, broken, t2, a, tfw, idx[broken]))
    return zones


@dataclass
class ZoneReport:
    ok: bool = False
    error: str = ""
    zones: List[Dict] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)
    atr: Dict[str, float] = field(default_factory=dict)
    nearest: Dict[str, Optional[Dict]] = field(default_factory=dict)


def compute(gold: pd.DataFrame) -> ZoneReport:
    rep = ZoneReport()
    if gold is None or len(gold) < 60:
        rep.error = "need ≥60 bars"
        return rep
    price = float(gold["Close"].iat[-1])
    for tf, spec in cfg.ZONE_TFS.items():
        d = resample(gold, spec["rule"]).iloc[-spec["bars"]:]
        z = detect(d, tf, spec["weight"])
        rep.zones.extend(z)
        rep.counts[tf] = len(z)
        if len(d) > 15:
            rep.atr[tf] = float(xt.atr(d).iat[-1])
    sup = [z for z in rep.zones if z["side"] == SUPPORT and z["lo"] <= price]
    res = [z for z in rep.zones if z["side"] == RESIST and z["hi"] >= price]
    rep.nearest["support"] = max(sup, key=lambda z: z["hi"]) if sup else None
    rep.nearest["resistance"] = min(res, key=lambda z: z["lo"]) if res else None
    rep.ok = True
    return rep


def get_zone_report(gold: pd.DataFrame) -> ZoneReport:
    try:
        return compute(gold)
    except Exception as e:  # noqa: BLE001
        return ZoneReport(error=f"{type(e).__name__}: {e}")
