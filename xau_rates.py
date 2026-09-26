"""
xau_rates.py — Layer 1: US rates (score ±20 × correlation weight).

Inputs (all oriented so rising yields = bearish gold):
  10y day   ZN=F futures % move → bp (fallback ^TNX level change)
  2y day    ZT=F futures % move → bp (fallback ^FVX 5y level change)
  30-min    short-end move over the last 2 bars; ≥5bp up raises the G1 flag
  Real 10y  FRED DFII10, 5-observation change (daily, lags a day)

Treasury futures are used for intraday because ^TNX only updates in US cash
hours. The whole layer is scaled by how inversely gold has been trading
against 10y yields over 5 and 20 days: when the link breaks the layer fades.
"""
from typing import Dict, Optional

import pandas as pd

import xau_config as cfg
import xau_macro as xm


def _yield_series_from_fut(df: pd.DataFrame, duration: float):
    return lambda s: -s.pct_change() * 1e4 / duration


def compute(bundle: Dict, fred: Optional[Dict[str, pd.Series]] = None) -> xm.MacroReport:
    rep = xm.MacroReport(layer="Rates", max_pts=cfg.RATES_MAX)
    gold = bundle.get("gold")
    cross = bundle.get("cross", {})
    if gold is None or gold.empty:
        rep.error = "no gold data"
        return rep

    zn, zt = cross.get("ZN=F"), cross.get("ZT=F")
    tnx = xm.normalize_yield(cross.get("^TNX"))
    fvx = xm.normalize_yield(cross.get("^FVX"))
    comp, det, rd = {}, {}, {}

    # 10y day
    bp10, src10 = None, None
    if xm.usable(zn, gold):
        bp10, src10 = xm.fut_pct_to_bp(xm.day_change_pct(zn), cfg.FUT_DURATION["ZN=F"]), "ZN=F"
    elif xm.usable(tnx, gold, stale_min=24 * 60):
        ch = xm.day_change_abs(tnx)
        bp10, src10 = (None if ch is None else ch * 100), "^TNX"
    comp["10y day"] = None if bp10 is None else xm.clip1(-bp10 / cfg.REF_10Y_DAY_BP)
    det["10y day"] = "no data" if bp10 is None else f"{bp10:+.1f} bp ({src10})"
    rd["10y_day_bp"] = bp10

    # 2y day
    bp2, src2, short_df, short_d = None, None, None, None
    if xm.usable(zt, gold):
        bp2, src2 = xm.fut_pct_to_bp(xm.day_change_pct(zt), cfg.FUT_DURATION["ZT=F"]), "ZT=F"
        short_df, short_d = zt, cfg.FUT_DURATION["ZT=F"]
    elif xm.usable(fvx, gold, stale_min=24 * 60):
        ch = xm.day_change_abs(fvx)
        bp2, src2 = (None if ch is None else ch * 100), "^FVX (5y proxy)"
    comp["2y day"] = None if bp2 is None else xm.clip1(-bp2 / cfg.REF_2Y_DAY_BP)
    det["2y day"] = "no data" if bp2 is None else f"{bp2:+.1f} bp ({src2})"
    rd["2y_day_bp"] = bp2

    # 30-min spike (short end preferred, 10y futures as fallback)
    if short_df is None and xm.usable(zn, gold):
        short_df, short_d = zn, cfg.FUT_DURATION["ZN=F"]
    spike = None
    if short_df is not None:
        spike = xm.fut_pct_to_bp(xm.recent_change_pct(short_df, cfg.SPIKE_BARS), short_d)
    comp["30-min spike"] = None if spike is None else xm.clip1(-spike / cfg.REF_SPIKE_BP)
    det["30-min spike"] = "no data" if spike is None else f"{spike:+.1f} bp"
    rd["spike_bp"] = spike
    rep.flags["yield_spike_up"] = bool(spike is not None and spike >= cfg.SPIKE_FLAG_BP)
    rep.flags["yield_spike_down"] = bool(spike is not None and spike <= -cfg.SPIKE_FLAG_BP)
    if rep.flags["yield_spike_up"]:
        rep.notes.append(f"Yields spiked {spike:+.1f} bp in 30 min — longs at risk (G1)")
    if rep.flags["yield_spike_down"]:
        rep.notes.append(f"Yields dropped {spike:+.1f} bp in 30 min — tailwind for gold")

    # Real yield (daily FRED)
    fred = fred or {}
    ry = fred.get(cfg.FRED_REAL_10Y)
    be = fred.get(cfg.FRED_BREAKEVEN_10Y)
    if ry is not None and len(ry) >= 6:
        lvl, chg = float(ry.iat[-1]), float((ry.iat[-1] - ry.iat[-6]) * 100)
        comp["Real yield"] = xm.clip1(-chg / cfg.REF_REAL_5D_BP)
        det["Real yield"] = f"{lvl:.2f}% ({chg:+.0f} bp over 5 obs, as of " \
                            f"{ry.index[-1]:%d %b})"
        rd["real_10y"], rd["real_5d_bp"] = lvl, chg
        if lvl >= cfg.REAL_YIELD_HIGH:
            rep.notes.append(f"10y real yield {lvl:.2f}% ≥ {cfg.REAL_YIELD_HIGH}% — "
                             "high opportunity cost for gold")
    else:
        comp["Real yield"] = None
        det["Real yield"] = "FRED unavailable"
    if be is not None and len(be):
        rd["breakeven_10y"] = float(be.iat[-1])
    if tnx is not None and not tnx.empty:
        rd["tnx_level"] = float(tnx["Close"].iat[-1])

    # Correlation: gold returns vs 10y yield changes (bp)
    if zn is not None and not zn.empty:
        r = xm.aligned_returns(gold, zn, _yield_series_from_fut(zn, cfg.FUT_DURATION["ZN=F"]))
        rep.corr_source = "gold vs 10y yield (ZN=F)"
    elif tnx is not None and not tnx.empty:
        r = xm.aligned_returns(gold, tnx, lambda s: s.diff() * 100)
        rep.corr_source = "gold vs 10y yield (^TNX, US hours only)"
    else:
        r = pd.DataFrame(columns=["g", "o"])
    cs = xm.corr_stats(r)
    rep.corr_short, rep.corr_long, rep.corr_series = cs["short"], cs["long"], cs["series"]
    rep.corr_expected = "negative"
    inv = lambda c: None if c is None else -c
    rep.weight = xm.dynamic_weight(inv(cs["short"]), inv(cs["long"]))
    if rep.weight is not None and rep.weight < 0.25:
        rep.notes.append("Gold is barely trading off yields right now — rates layer "
                         "muted (structural bid or other driver in control)")
    if cs["long"] is not None and cs["long"] > 0.1:
        rep.notes.append(f"Gold–yield correlation is positive ({cs['long']:+.2f}) — "
                         "textbook relationship inverted")

    rep.components, rep.details, rep.readings = comp, det, rd
    return xm.finish(rep, cfg.RATES_WEIGHTS, fallback_weight=0.5)


def get_rates_report(bundle: Dict, fred: Optional[Dict] = None) -> xm.MacroReport:
    try:
        return compute(bundle, fred)
    except Exception as e:  # noqa: BLE001
        return xm.MacroReport(layer="Rates", max_pts=cfg.RATES_MAX,
                              error=f"{type(e).__name__}: {e}")


def load_fred() -> Dict[str, Optional[pd.Series]]:
    return {sid: xm.fetch_fred(sid) for sid in (cfg.FRED_REAL_10Y, cfg.FRED_BREAKEVEN_10Y)}
