"""
xau_regime.py — Layer 8: which driver is in control (score ±15).

Each candidate regime gets a strength 0..1 from the other layers' outputs:
  rates-led  max(L1 weight, L2 weight) × macro activity (|raw| of L1/L2) —
             gold is trading off yields/dollar AND they are moving today
  flow-led   (1 − link) × |L3 residual z|/2, or 1.0 on a structural bid/offer
  risk-off   ES sell-off + VIX jump (level or % change)
  trend      ADX between TREND_ADX_START and TREND_ADX_FULL
  chop       ADX below ADX_CHOP (stronger the lower it is; compressed ATR adds)
The strongest wins; if none reaches REGIME_MIN_STRENGTH the regime is "mixed".

Direction comes from the winner's own driver, and the score is
REGIME_MAX × strength × direction. Chop and mixed score 0 — their job is to
tell the master signal how to trade (style), not which way.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import xau_config as cfg
import xau_macro as xm


def _c(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


@dataclass
class RegimeReport:
    ok: bool = False
    error: str = ""
    regime: str = "mixed"
    label: str = cfg.REGIME_LABELS["mixed"]
    sub: str = ""                       # e.g. "haven bid" / "liquidation"
    strength: float = 0.0
    direction: float = 0.0              # −1..+1
    score: float = 0.0
    bias: str = "NEUTRAL"
    style: str = cfg.REGIME_STYLE["mixed"]
    playbook: str = cfg.REGIME_PLAYBOOK["mixed"]
    candidates: Dict[str, float] = field(default_factory=dict)
    why: Dict[str, str] = field(default_factory=dict)
    readings: Dict[str, Optional[float]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


def _ok(rep) -> bool:
    return bool(rep is not None and getattr(rep, "ok", False))


def risk_off_inputs(bundle: Dict) -> Dict[str, Optional[float]]:
    gold = bundle.get("gold")
    cross = bundle.get("cross", {})
    es, vix = cross.get("ES=F"), cross.get("^VIX")
    es_pct = xm.day_change_pct(es) if xm.usable(es, gold) else None
    vix_ok = xm.usable(vix, gold, stale_min=24 * 60)
    vix_pct = xm.day_change_pct(vix) if vix_ok else None
    vix_lvl = float(vix["Close"].iat[-1]) if vix_ok else None
    gold_pct = xm.day_change_pct(gold) if gold is not None and not gold.empty else None
    return {"es_day_pct": es_pct, "vix_day_pct": vix_pct, "vix_level": vix_lvl,
            "gold_day_pct": gold_pct}


def compute(bundle: Dict, tech=None, rates=None, dollar=None, xasset=None) -> RegimeReport:
    rep = RegimeReport()
    cand, why = {}, {}

    # rates-led
    wr = rates.weight if _ok(rates) and rates.weight is not None else None
    wd = dollar.weight if _ok(dollar) and dollar.weight is not None else None
    link = max([w for w in (wr, wd) if w is not None], default=None)
    if link is not None:
        raws = [abs(r.raw) for r in (rates, dollar) if _ok(r)]
        activity = _c(max(raws, default=0.0) / cfg.RATES_LED_ACTIVE_RAW)
        cand["rates-led"] = round(link * activity, 2)
        why["rates-led"] = f"link {link:.0%} (yields {0 if wr is None else wr:.0%}, " \
                           f"USD {0 if wd is None else wd:.0%}) × macro activity " \
                           f"{activity:.0%}"

    # flow-led
    if _ok(xasset):
        z = xasset.implied["resid_z"] if xasset.implied else None
        structural = xasset.flags.get("structural_bid") or xasset.flags.get("structural_offer")
        if structural:
            cand["flow-led"] = 1.0
            why["flow-led"] = "structural bid/offer against yields and the dollar"
        elif z is not None:
            decouple = 1 - (link if link is not None else 0.5)
            cand["flow-led"] = round(_c(decouple * abs(z) / cfg.RESID_Z_FULL), 2)
            why["flow-led"] = f"residual z {z:+.1f}, decoupling {decouple:.0%}"

    # risk-off
    ro = risk_off_inputs(bundle)
    rep.readings.update(ro)
    if ro["es_day_pct"] is not None:
        es_part = _c(ro["es_day_pct"] / cfg.RISKOFF_ES_PCT)       # both negative → positive
        vix_part = 0.0
        if ro["vix_day_pct"] is not None:
            vix_part = _c(ro["vix_day_pct"] / cfg.RISKOFF_VIX_CHG_PCT)
        if ro["vix_level"] is not None and ro["vix_level"] >= cfg.RISKOFF_VIX_LEVEL:
            vix_part = max(vix_part, 1.0)
        # both legs needed: an equity dip without a vol jump isn't risk-off
        cand["risk-off"] = round(min(es_part, vix_part) if ro["vix_day_pct"] is not None
                                 or ro["vix_level"] is not None else es_part * 0.5, 2)
        why["risk-off"] = f"ES {ro['es_day_pct']:+.2f}%" + (
            "" if ro["vix_day_pct"] is None else f", VIX {ro['vix_day_pct']:+.1f}%") + (
            "" if ro["vix_level"] is None else f" (level {ro['vix_level']:.1f})")

    # trend / chop from technicals
    if _ok(tech):
        adx = tech.adx
        rep.readings["adx"] = adx
        span = cfg.TREND_ADX_FULL - cfg.TREND_ADX_START
        cand["trend"] = round(_c((adx - cfg.TREND_ADX_START) / span), 2)
        why["trend"] = f"ADX {adx:.0f}"
        if adx < cfg.ADX_CHOP:
            chop = 0.5 + 0.5 * (cfg.ADX_CHOP - adx) / cfg.ADX_CHOP
            if tech.atr_regime == "compressed":
                chop = min(1.0, chop + 0.15)
            cand["chop"] = round(_c(chop), 2)
            why["chop"] = f"ADX {adx:.0f} < {cfg.ADX_CHOP}, ATR {tech.atr_regime}"
        else:
            cand["chop"] = 0.0
            why["chop"] = f"ADX {adx:.0f} ≥ {cfg.ADX_CHOP}"

    if not cand:
        rep.error = "no layer inputs available"
        return rep
    rep.candidates, rep.why = cand, why

    best = max(cand, key=cand.get)
    strength = cand[best]
    regime = best if strength >= cfg.REGIME_MIN_STRENGTH else "mixed"

    # direction from the winning driver
    d = 0.0
    if regime == "rates-led":
        pts = [(r.score, r.eff_max) for r in (rates, dollar) if _ok(r)]
        cap = sum(m for _, m in pts)
        d = max(-1.0, min(1.0, sum(s for s, _ in pts) / (0.5 * cap))) if cap else 0.0
    elif regime == "flow-led":
        if xasset.flags.get("structural_bid"):
            d = 1.0
        elif xasset.flags.get("structural_offer"):
            d = -1.0
        else:
            d = max(-1.0, min(1.0, xasset.implied["resid_z"] / cfg.RESID_Z_FULL))
    elif regime == "risk-off":
        g = ro["gold_day_pct"]
        if g is not None and g >= cfg.RISKOFF_HAVEN_GOLD_PCT:
            d, rep.sub = 1.0, "haven bid"
        elif g is not None and g < 0:
            d, rep.sub = -1.0, "liquidation"
            rep.notes.append("Gold being sold alongside equities — margin-call "
                             "liquidation; havens usually recover once it ends")
        else:
            d, rep.sub = 0.0, "gold not reacting yet"
    elif regime == "trend":
        c = tech.components
        d = max(-1.0, min(1.0, (c.get("1h trend", 0) + c.get("Structure", 0)) / 9))

    rep.regime, rep.strength, rep.direction = regime, round(strength, 2), round(d, 2)
    rep.label = cfg.REGIME_LABELS[regime] + (f" — {rep.sub}" if rep.sub else "")
    rep.style = cfg.REGIME_STYLE[regime]
    rep.playbook = cfg.REGIME_PLAYBOOK[regime]
    rep.score = round(cfg.REGIME_MAX * strength * d, 1) if regime != "mixed" else 0.0
    cut = cfg.MACRO_BIAS_FRAC * cfg.REGIME_MAX
    rep.bias = "LONG" if rep.score >= cut else "SHORT" if rep.score <= -cut else "NEUTRAL"

    runner_up = sorted(cand.items(), key=lambda kv: -kv[1])
    if len(runner_up) > 1 and runner_up[0][1] - runner_up[1][1] < 0.1 \
            and runner_up[1][1] >= cfg.REGIME_MIN_STRENGTH:
        rep.notes.append(f"Close call between {runner_up[0][0]} and {runner_up[1][0]} "
                         "— regime may flip")
    if regime == "chop":
        rep.notes.append("Chop regime — momentum/breakout signals should be blocked (G6)")
    rep.ok = True
    return rep


def get_regime_report(bundle: Dict, tech=None, rates=None, dollar=None,
                      xasset=None) -> RegimeReport:
    try:
        return compute(bundle, tech, rates, dollar, xasset)
    except Exception as e:  # noqa: BLE001
        return RegimeReport(error=f"{type(e).__name__}: {e}")
