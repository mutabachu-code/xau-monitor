"""
xau_dollar.py — Layer 2: US dollar (score ±15 × correlation weight).

Inputs (oriented so a stronger dollar = bearish gold):
  DXY day       DX-Y.NYB % vs previous CME close
  EUR/USD day   EURUSD=X (sign flipped — EUR up means USD down)
  USD/JPY day   JPY=X
  DXY 1h        last 4 bars; EUR/USD (flipped) if DXY is stale

Weight comes from the 5d/20d correlation of gold vs DXY returns (vs EUR/USD,
flipped, when DXY is missing).
"""
from typing import Dict

import pandas as pd

import xau_config as cfg
import xau_macro as xm


def compute(bundle: Dict) -> xm.MacroReport:
    rep = xm.MacroReport(layer="Dollar", max_pts=cfg.DOLLAR_MAX)
    gold = bundle.get("gold")
    cross = bundle.get("cross", {})
    if gold is None or gold.empty:
        rep.error = "no gold data"
        return rep
    dxy, eur, jpy = cross.get("DX-Y.NYB"), cross.get("EURUSD=X"), cross.get("JPY=X")
    ok_dxy, ok_eur, ok_jpy = (xm.usable(d, gold) for d in (dxy, eur, jpy))
    comp, det, rd = {}, {}, {}

    def day(name, df, ok, ref, sign, label):
        pct = xm.day_change_pct(df) if ok else None
        comp[name] = None if pct is None else xm.clip1(sign * pct / ref)
        det[name] = ("stale / no data" if not ok else "no prior close") if pct is None \
            else f"{pct:+.2f}% ({label})"
        rd[name] = pct
        return pct

    dxy_pct = day("DXY day", dxy, ok_dxy, cfg.REF_DXY_DAY_PCT, -1, "DX-Y.NYB")
    day("EUR/USD day", eur, ok_eur, cfg.REF_EUR_DAY_PCT, +1, "EURUSD=X")
    day("USD/JPY day", jpy, ok_jpy, cfg.REF_JPY_DAY_PCT, -1, "JPY=X")

    h1 = None
    if ok_dxy:
        h1 = xm.recent_change_pct(dxy, 4)
        src = "DXY"
    elif ok_eur:
        e = xm.recent_change_pct(eur, 4)
        h1, src = (None if e is None else -e), "EUR/USD inverted"
    comp["DXY 1h"] = None if h1 is None else xm.clip1(-h1 / cfg.REF_DXY_1H_PCT)
    det["DXY 1h"] = "no data" if h1 is None else f"{h1:+.2f}% ({src})"
    rd["DXY 1h"] = h1

    if dxy_pct is not None and dxy_pct >= 0.3:
        rep.flags["dollar_strong_up"] = True
        rep.notes.append(f"Dollar up {dxy_pct:+.2f}% on the day — headwind for gold")
    if dxy_pct is not None and dxy_pct <= -0.3:
        rep.flags["dollar_strong_down"] = True
        rep.notes.append(f"Dollar down {dxy_pct:+.2f}% on the day — tailwind for gold")
    if not ok_dxy and dxy is not None and not dxy.empty:
        rep.notes.append("DXY feed is stale — using EUR/USD for the short-term read")

    # correlation
    if dxy is not None and not dxy.empty:
        r = xm.aligned_returns(gold, dxy, lambda s: s.pct_change())
        rep.corr_source, flip = "gold vs DXY", -1
    elif eur is not None and not eur.empty:
        r = xm.aligned_returns(gold, eur, lambda s: s.pct_change())
        rep.corr_source, flip = "gold vs EUR/USD", 1
    else:
        r, flip = pd.DataFrame(columns=["g", "o"]), -1
    cs = xm.corr_stats(r)
    rep.corr_short, rep.corr_long, rep.corr_series = cs["short"], cs["long"], cs["series"]
    rep.corr_expected = "negative" if flip < 0 else "positive"
    orient = lambda c: None if c is None else flip * c
    rep.weight = xm.dynamic_weight(orient(cs["short"]), orient(cs["long"]))
    if rep.weight is not None and rep.weight < 0.25:
        rep.notes.append("Gold is barely trading off the dollar right now — "
                         "dollar layer muted")

    rep.components, rep.details, rep.readings = comp, det, rd
    return xm.finish(rep, cfg.DOLLAR_WEIGHTS, fallback_weight=0.5)


def get_dollar_report(bundle: Dict) -> xm.MacroReport:
    try:
        return compute(bundle)
    except Exception as e:  # noqa: BLE001
        return xm.MacroReport(layer="Dollar", max_pts=cfg.DOLLAR_MAX,
                              error=f"{type(e).__name__}: {e}")
