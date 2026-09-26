"""
xau_crossasset.py — Layer 3: cross-asset agreement (score ±20).

1. Implied move (±8): gold's 15m returns are regressed (no intercept) on the
   drivers — 10y yield bp change (ZN=F), DXY return (EUR/USD flipped if
   missing) and oil return — over the prior 20 trading days. Today's actual
   move minus the move the drivers imply is the residual; its z-score says
   whether gold is stronger (hidden bid) or weaker than its drivers justify.
2. Structural bid/offer (±5, G7): gold up while yields AND the dollar are up
   is buying that ignores the usual headwinds; the mirror is a structural offer.
3. Silver (±4, G3): same-direction silver confirms; gold at a fresh session
   high/low that silver doesn't match is a non-confirmation warning.
4. Metals context (±3): falling gold/silver ratio and rising copper = broad
   metals bid; rising ratio + falling copper = gold-only safe-haven move.
Also raises the oil-spike flag (G8) for the master signal.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import xau_config as cfg
import xau_macro as xm
import xau_sessions as xs
import xau_technicals as xt


# ── Implied-move model ───────────────────────────────────────────────────────
def driver_frame(bundle: Dict) -> pd.DataFrame:
    """Aligned per-bar returns: g (gold) plus available drivers, gaps removed."""
    gold = bundle["gold"]
    cross = bundle.get("cross", {})
    cols = {"g": gold["Close"]}
    zn = cross.get("ZN=F")
    dxy, eur, oil = cross.get("DX-Y.NYB"), cross.get("EURUSD=X"), cross.get("CL=F")
    if zn is not None and not zn.empty:
        cols["yield_bp"] = zn["Close"]
    if dxy is not None and not dxy.empty:
        cols["dxy"] = dxy["Close"]
    elif eur is not None and not eur.empty:
        cols["dxy"] = 1 / eur["Close"]                 # USD strength proxy
    if oil is not None and not oil.empty:
        cols["oil"] = oil["Close"]
    j = pd.concat(cols, axis=1, join="inner").dropna()
    if len(j) < 3:
        return pd.DataFrame()
    out = pd.DataFrame(index=j.index)
    out["g"] = j["g"].pct_change()
    if "yield_bp" in j:
        out["yield_bp"] = -j["yield_bp"].pct_change() * 1e4 / cfg.FUT_DURATION["ZN=F"]
    for c in ("dxy", "oil"):
        if c in j:
            out[c] = j[c].pct_change()
    gap = j.index.to_series().diff() > pd.Timedelta(minutes=30)
    return out[~gap].dropna()


def fit_implied(frame: pd.DataFrame, today) -> Optional[Dict]:
    """OLS without intercept on the prior IMPLIED_TRAIN_DAYS trading days."""
    if frame.empty or frame.shape[1] < 2:
        return None
    days = np.asarray(xs.trading_day_index(frame.index))
    prior = sorted({d for d in days if d < today})[-cfg.IMPLIED_TRAIN_DAYS:]
    train = frame[np.isin(days, prior)]
    if len(train) < cfg.IMPLIED_MIN_TRAIN:
        return None
    drivers = [c for c in frame.columns if c != "g"]
    X, y = train[drivers].to_numpy(), train["g"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_tot = float(np.sum(y ** 2))
    r2 = 1 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else 0.0
    return {"beta": dict(zip(drivers, beta)), "resid_sd": float(resid.std()),
            "r2": r2, "n": len(train), "drivers": drivers}


def implied_window(frame: pd.DataFrame, model: Dict, today) -> Optional[Dict]:
    """Actual vs implied cumulative gold return over today's bars
    (at least IMPLIED_MIN_WINDOW bars, reaching back if the day is young)."""
    days = np.asarray(xs.trading_day_index(frame.index))
    n_today = int((days == today).sum())
    n = max(n_today, cfg.IMPLIED_MIN_WINDOW)
    w = frame.iloc[-n:]
    if len(w) < 2:
        return None
    X = w[model["drivers"]].to_numpy()
    b = np.array([model["beta"][d] for d in model["drivers"]])
    imp_bar = X @ b
    actual = float(w["g"].sum() * 100)
    implied = float(imp_bar.sum() * 100)
    resid = actual - implied
    sd = model["resid_sd"] * np.sqrt(len(w)) * 100
    path = pd.DataFrame({"actual": w["g"].cumsum() * 100,
                         "implied": pd.Series(imp_bar, index=w.index).cumsum() * 100})
    contrib = {d: float((w[d].to_numpy() * model["beta"][d]).sum() * 100)
               for d in model["drivers"]}
    return {"actual_pct": actual, "implied_pct": implied, "resid_pct": resid,
            "resid_z": resid / sd if sd > 0 else 0.0, "bars": len(w),
            "today_bars": n_today, "path": path, "contrib": contrib}


# ── Confirmation helpers ─────────────────────────────────────────────────────
def session_slice(df: pd.DataFrame, today) -> pd.DataFrame:
    days = np.asarray(xs.trading_day_index(df.index))
    return df[days == today]


def silver_check(gold: pd.DataFrame, silver: pd.DataFrame, today, atr_val: float) -> Dict:
    g, s = session_slice(gold, today), session_slice(silver, today)
    if len(g) < cfg.NEW_EXTREME_BARS + 1 or len(s) < cfg.NEW_EXTREME_BARS + 1:
        return {"state": "n/a"}
    g_last, s_last = float(g["Close"].iat[-1]), float(s["Close"].iat[-1])
    tol = cfg.SILVER_HIGH_TOL_ATR * atr_val
    recent_hi = g["High"].iloc[-cfg.NEW_EXTREME_BARS:].max() >= g["High"].max() - 1e-9
    recent_lo = g["Low"].iloc[-cfg.NEW_EXTREME_BARS:].min() <= g["Low"].min() + 1e-9
    at_hi = recent_hi and g_last >= g["High"].max() - tol
    at_lo = recent_lo and g_last <= g["Low"].min() + tol
    s_off_hi = (s["High"].max() - s_last) / s["High"].max() * 100
    s_off_lo = (s_last - s["Low"].min()) / s["Low"].min() * 100
    if at_hi and s_off_hi >= cfg.SILVER_LAG_PCT:
        return {"state": "not confirming high", "silver_off_pct": s_off_hi}
    if at_lo and s_off_lo >= cfg.SILVER_LAG_PCT:
        return {"state": "not confirming low", "silver_off_pct": s_off_lo}
    if at_hi:
        return {"state": "confirming high"}
    if at_lo:
        return {"state": "confirming low"}
    return {"state": "mid-range"}


# ── Report ───────────────────────────────────────────────────────────────────
@dataclass
class XAssetReport:
    ok: bool = False
    error: str = ""
    score: float = 0.0
    bias: str = "NEUTRAL"
    components: Dict[str, Optional[float]] = field(default_factory=dict)  # points
    details: Dict[str, str] = field(default_factory=dict)
    model: Optional[Dict] = None
    implied: Optional[Dict] = None
    move_type: str = "n/a"
    gs_ratio: Optional[float] = None
    gs_ratio_chg: Optional[float] = None
    readings: Dict[str, Optional[float]] = field(default_factory=dict)
    flags: Dict[str, bool] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


def _sgn(x, eps=0.0):
    return 0 if x is None or abs(x) <= eps else (1 if x > 0 else -1)


def compute(bundle: Dict) -> XAssetReport:
    rep = XAssetReport()
    gold = bundle.get("gold")
    cross = bundle.get("cross", {})
    if gold is None or len(gold) < 60:
        rep.error = "need ≥60 gold bars"
        return rep
    today = xs.trading_day(gold.index[-1].to_pydatetime())
    atr_val = float(xt.atr(gold).iat[-1])
    P = cfg.XASSET_PTS
    comp, det, rd = {}, {}, {}

    gold_day = xm.day_change_pct(gold)
    rd["gold_day_pct"] = gold_day

    # 1. implied move
    frame = driver_frame(bundle)
    model = fit_implied(frame, today)
    rep.model = model
    if model:
        imp = implied_window(frame, model, today)
        rep.implied = imp
    else:
        imp = None
    if imp:
        z = imp["resid_z"]
        comp["Residual"] = round(P["Residual"] * xm.clip1(z / cfg.RESID_Z_FULL), 1)
        det["Residual"] = (f"actual {imp['actual_pct']:+.2f}% vs implied "
                           f"{imp['implied_pct']:+.2f}% over {imp['bars']} bars "
                           f"(residual {imp['resid_pct']:+.2f}%, z {z:+.1f}, "
                           f"R² {model['r2']:.2f})")
        if z >= 1.5:
            rep.notes.append("Gold is stronger than yields/dollar/oil explain — "
                             "hidden bid")
        elif z <= -1.5:
            rep.notes.append("Gold is weaker than its drivers explain — hidden supply")
        if model["r2"] < 0.05:
            rep.notes.append(f"Drivers explain little of gold lately (R² "
                             f"{model['r2']:.2f}) — residual is mostly gold's own flow")
    else:
        comp["Residual"] = None
        det["Residual"] = "not enough overlapping driver history to fit"

    # 2. structural bid / offer (G7)
    zn, dxy, eur = cross.get("ZN=F"), cross.get("DX-Y.NYB"), cross.get("EURUSD=X")
    ybp = xm.fut_pct_to_bp(xm.day_change_pct(zn), cfg.FUT_DURATION["ZN=F"]) \
        if xm.usable(zn, gold) else None
    if xm.usable(dxy, gold):
        dpct = xm.day_change_pct(dxy)
    elif xm.usable(eur, gold):
        e = xm.day_change_pct(eur)
        dpct = None if e is None else -e
    else:
        dpct = None
    rd["yield_day_bp"], rd["dxy_day_pct"] = ybp, dpct
    rep.flags["structural_bid"] = bool(
        gold_day is not None and ybp is not None and dpct is not None
        and gold_day >= cfg.STRUCT_GOLD_PCT and ybp >= cfg.STRUCT_YIELD_BP
        and dpct >= cfg.STRUCT_DXY_PCT)
    rep.flags["structural_offer"] = bool(
        gold_day is not None and ybp is not None and dpct is not None
        and gold_day <= -cfg.STRUCT_GOLD_PCT and ybp <= -cfg.STRUCT_YIELD_BP
        and dpct <= -cfg.STRUCT_DXY_PCT)
    if rep.flags["structural_bid"]:
        comp["Structural"] = P["Structural"]
        det["Structural"] = "BID — gold up despite higher yields and a stronger dollar"
        rep.notes.append("Structural bid (G7): gold rising against yields and the "
                         "dollar — high-quality long context")
    elif rep.flags["structural_offer"]:
        comp["Structural"] = -P["Structural"]
        det["Structural"] = "OFFER — gold down despite lower yields and a weaker dollar"
        rep.notes.append("Structural offer: gold falling even with yields and the "
                         "dollar helping — sellers in control")
    else:
        comp["Structural"] = 0 if (ybp is not None and dpct is not None) else None
        det["Structural"] = "none" if comp["Structural"] is not None else "needs yields + dollar"
    if ybp is not None and dpct is not None and gold_day is not None:
        det["Structural"] += (f" (gold {gold_day:+.2f}%, 10y {ybp:+.1f} bp, "
                              f"USD {dpct:+.2f}%)")

    # 3. silver
    si = cross.get("SI=F")
    if xm.usable(si, gold):
        s_day = xm.day_change_pct(si)
        rd["silver_day_pct"] = s_day
        chk = silver_check(gold, si, today, atr_val)
        g_s, s_s = _sgn(gold_day, 0.05), _sgn(s_day, 0.05)
        pts = 0.0
        if g_s and s_s == g_s:
            pts = 2 * g_s
        elif g_s and s_s == -g_s:
            pts = -1 * g_s
        rep.flags["silver_nonconfirm_high"] = chk["state"] == "not confirming high"
        rep.flags["silver_nonconfirm_low"] = chk["state"] == "not confirming low"
        if rep.flags["silver_nonconfirm_high"]:
            pts -= 3
            rep.notes.append(f"Gold at a session high that silver isn't confirming "
                             f"(silver {chk['silver_off_pct']:.2f}% off its high) — G3")
        elif rep.flags["silver_nonconfirm_low"]:
            pts += 3
            rep.notes.append(f"Gold at a session low that silver isn't confirming "
                             f"(silver {chk['silver_off_pct']:.2f}% off its low)")
        elif chk["state"].startswith("confirming"):
            pts += 1 if chk["state"].endswith("high") else -1
        comp["Silver"] = float(max(-P["Silver"], min(P["Silver"], pts)))
        det["Silver"] = f"silver {s_day:+.2f}% on day, {chk['state']}" \
            if s_day is not None else chk["state"]
    else:
        comp["Silver"] = None
        det["Silver"] = "silver data stale / missing"

    # 4. metals context: gold/silver ratio + copper
    ctx, bits = 0.0, []
    if si is not None and not si.empty:
        j = pd.concat([gold["Close"], si["Close"]], axis=1, join="inner").dropna()
        if len(j):
            ratio = j.iloc[:, 0] / j.iloc[:, 1]
            rep.gs_ratio = float(ratio.iat[-1])
            tmp = pd.DataFrame({"Open": ratio, "High": ratio, "Low": ratio, "Close": ratio})
            ch = xm.day_change_pct(tmp)
            rep.gs_ratio_chg = ch
            if ch is not None:
                bits.append(f"G/S ratio {rep.gs_ratio:.1f} ({ch:+.2f}%)")
    hg = cross.get("HG=F")
    cu = xm.day_change_pct(hg) if xm.usable(hg, gold) else None
    rd["copper_day_pct"] = cu
    if cu is not None:
        bits.append(f"copper {cu:+.2f}%")
    g_s = _sgn(gold_day, 0.05)
    if g_s and rep.gs_ratio_chg is not None and cu is not None:
        broad = rep.gs_ratio_chg < 0 and cu > 0          # silver + copper leading
        haven = rep.gs_ratio_chg > 0 and cu < 0          # gold alone, industrials off
        if g_s > 0:
            rep.move_type = "broad metals bid" if broad else \
                "safe-haven (gold-only)" if haven else "mixed"
            ctx = 3 if broad else 1.5 if haven else 0.5
        else:
            rep.move_type = "broad metals selloff" if (rep.gs_ratio_chg > 0 and cu < 0) \
                else "gold-specific selling" if (rep.gs_ratio_chg < 0 and cu > 0) else "mixed"
            ctx = -3 if rep.move_type == "broad metals selloff" else \
                -1.5 if rep.move_type == "gold-specific selling" else -0.5
        comp["Metals context"] = ctx
    else:
        comp["Metals context"] = None if not bits else 0.0
    det["Metals context"] = (rep.move_type + " — " if rep.move_type != "n/a" else "") + \
        (", ".join(bits) if bits else "no silver/copper data")

    # oil (G8 flag, not scored here — the master decides by regime)
    cl = cross.get("CL=F")
    oil = xm.day_change_pct(cl) if xm.usable(cl, gold) else None
    rd["oil_day_pct"] = oil
    rep.flags["oil_spike_up"] = bool(oil is not None and oil >= cfg.OIL_SPIKE_PCT)
    rep.flags["oil_spike_down"] = bool(oil is not None and oil <= -cfg.OIL_SPIKE_PCT)
    if rep.flags["oil_spike_up"]:
        rep.notes.append(f"Oil {oil:+.1f}% — in the current hike-risk regime this "
                         "pressures gold via Fed expectations (G8)")

    present = {k: v for k, v in comp.items() if v is not None}
    if not present:
        rep.error = "no usable cross-asset inputs"
        return rep
    rep.components, rep.details, rep.readings = comp, det, rd
    rep.score = round(max(-cfg.XASSET_MAX, min(cfg.XASSET_MAX, sum(present.values()))), 1)
    cut = cfg.MACRO_BIAS_FRAC * cfg.XASSET_MAX
    rep.bias = "LONG" if rep.score >= cut else "SHORT" if rep.score <= -cut else "NEUTRAL"
    rep.ok = True
    return rep


def get_xasset_report(bundle: Dict) -> XAssetReport:
    try:
        return compute(bundle)
    except Exception as e:  # noqa: BLE001
        return XAssetReport(error=f"{type(e).__name__}: {e}")
