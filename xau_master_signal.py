"""
xau_master_signal.py — combine L1–L8 into one ±100 signal with conflict rules,
gates and a trade plan (Phase 7).

Score: Σ layer scores (±130 max) × 100/130. Direction from the sign.
Tier by |score|: A ≥60 · B ≥40 · C ≥25 · else none. Rules then move the tier,
block the trade or cut size:

  G1  yields spiked ≥5bp/30m against the trade ............ downgrade
  G2  yields + dollar both against, gold not moving with ... block
  G3  silver not confirming gold's new high/low ............ downgrade
  G4  COT crowded long + FOMC/CPI within 7 days (longs) .... size ×0.5
  G5  failed Asian-range breakout / Asia-session entry ..... block / downgrade
  G6  chop regime: only fades from beyond ±1σ VWAP ......... block
  G7  structural bid/offer in the trade's direction ........ upgrade
  G8  oil spike while the yield link is live (longs) ....... downgrade
  G9  GVZ spike with price at the option wall ahead ........ block
  G10 ≥6 of 8 layers agree ................................. upgrade
  M1  mixed regime: C-tier trades are skipped
Gates: BLOCKED → WAIT; CAUTION → size ×0.5. ATR expanding/extreme cuts size.

Plan (GC=F, and spot via the sidebar basis): entry zone, stop beyond the last
confirmed swing (or active sweep extreme) clamped to 1–3 ATR, TP1 at 1.5R or a
liquidity level between 1R and 2.5R, TP2 at 3R or front-running the option wall.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import xau_config as cfg
import xau_sessions as xs
import xau_technicals as xt


def _ok(r) -> bool:
    return bool(r is not None and getattr(r, "ok", False))


def _tier_idx(abs_score: float) -> int:
    for i, (cut, _) in enumerate(cfg.TIERS):
        if abs_score >= cut:
            return len(cfg.TIERS) - i          # A=3, B=2, C=1
    return 0


TIER_NAMES = {3: "A", 2: "B", 1: "C", 0: "NONE"}


@dataclass
class MasterSignal:
    ok: bool = False
    error: str = ""
    raw_total: float = 0.0
    score: float = 0.0                 # ±100
    direction: str = "NEUTRAL"         # LONG | SHORT | NEUTRAL
    tier: str = "NONE"
    action: str = "WAIT"               # LONG | SHORT | WAIT
    size_mult: float = 0.0
    layers: List[Dict] = field(default_factory=list)
    agree: int = 0
    active: int = 0
    rules: List[Dict] = field(default_factory=list)
    blocked_by: List[str] = field(default_factory=list)
    plan: Optional[Dict] = None
    notes: List[str] = field(default_factory=list)


# ── Plan ─────────────────────────────────────────────────────────────────────
def last_swing(df: pd.DataFrame, direction: str, price: float, atr: float) -> Optional[float]:
    sub = df.iloc[-cfg.SWING_LOOKBACK_BARS:]
    sw = xt.swing_points(sub, cfg.SWING_N)
    n = cfg.SWING_N
    if direction == "LONG":
        pts = [float(sub["Low"].iat[i]) for i in sw["lows"] if i + n < len(sub)]
        pts = [p for p in pts if p < price - 0.3 * atr]
    else:
        pts = [float(sub["High"].iat[i]) for i in sw["highs"] if i + n < len(sub)]
        pts = [p for p in pts if p > price + 0.3 * atr]
    return pts[-1] if pts else None


def build_plan(direction: str, gold: pd.DataFrame, tech, liq, opts, basis: float) -> Optional[Dict]:
    if direction not in ("LONG", "SHORT") or not _ok(tech):
        return None
    s = 1 if direction == "LONG" else -1
    price = float(gold["Close"].iat[-1])
    atr = float(tech.atr)
    if not atr or atr <= 0:
        return None

    # stop
    swing = last_swing(gold, direction, price, atr)
    stop_src = "swing"
    sl = None if swing is None else swing - s * cfg.STOP_BUFFER_ATR * atr
    if _ok(liq) and liq.active and np.sign(liq.active["points"]) == s:
        sweep_sl = liq.active["extreme"] - s * cfg.STOP_BUFFER_ATR * atr
        if sl is None or s * (price - sweep_sl) > s * (price - sl):
            sl, stop_src = sweep_sl, "sweep extreme"
    dist = None if sl is None else s * (price - sl)
    if dist is None or dist > cfg.STOP_MAX_ATR * atr:
        sl, stop_src = price - s * cfg.STOP_DEFAULT_ATR * atr, f"{cfg.STOP_DEFAULT_ATR} ATR"
    elif dist < cfg.STOP_MIN_ATR * atr:
        sl, stop_src = price - s * cfg.STOP_MIN_ATR * atr, f"min {cfg.STOP_MIN_ATR} ATR"
    R = abs(price - sl)

    # TP1
    tp1, tp1_src = price + s * cfg.TP1_R * R, f"{cfg.TP1_R}R"
    tgt = None
    if _ok(liq):
        tgt = liq.targets.get("above" if s > 0 else "below")
    if tgt:
        d = abs(tgt["price"] - price)
        lo, hi = cfg.TP1_TARGET_R
        if lo * R <= d <= hi * R:
            tp1 = tgt["price"] - s * cfg.TARGET_BUFFER_ATR * atr
            tp1_src = f"{tgt['name']} liquidity"

    # TP2
    tp2, tp2_src = price + s * cfg.TP2_R * R, f"{cfg.TP2_R}R"
    wall = None
    if _ok(opts):
        wall = opts.call_wall if s > 0 else opts.put_wall
    if wall is not None and s * (wall - tp1) > 0 and s * (wall - tp2) < 0:
        tp2 = wall - s * cfg.TARGET_BUFFER_ATR * atr
        tp2_src = "option wall"
    if s * (tp2 - tp1) <= 0:
        tp2, tp2_src = tp1 + s * R, "TP1 + 1R"

    zone = sorted([price, price - s * cfg.ENTRY_ZONE_ATR * atr])
    r = lambda x: round(float(x), 2)
    return {
        "direction": direction, "entry": r(price), "zone": (r(zone[0]), r(zone[1])),
        "sl": r(sl), "tp1": r(tp1), "tp2": r(tp2), "risk": r(R), "atr": r(atr),
        "rr1": round(abs(tp1 - price) / R, 2), "rr2": round(abs(tp2 - price) / R, 2),
        "stop_src": stop_src, "tp1_src": tp1_src, "tp2_src": tp2_src, "basis": basis,
        "spot": {k: r(v - basis) for k, v in (("entry", price), ("sl", sl), ("tp1", tp1),
                                              ("tp2", tp2))},
    }


# ── Master ───────────────────────────────────────────────────────────────────
def compute(layers: Dict, gold: pd.DataFrame, gate=None, basis: float = 0.0,
            now=None) -> MasterSignal:
    """layers: {"L1": rates, "L2": dollar, "L3": xasset, "L4": flows,
                "L5": tech, "L6": liq, "L7": opts, "L8": regime}"""
    sig = MasterSignal()
    if gold is None or gold.empty:
        sig.error = "no gold data"
        return sig
    rates, dollar, xasset, flows = (layers.get(k) for k in ("L1", "L2", "L3", "L4"))
    tech, liq, opts, regime = (layers.get(k) for k in ("L5", "L6", "L7", "L8"))

    total = 0.0
    for k, mx in cfg.LAYER_MAX.items():
        rep = layers.get(k)
        ok = _ok(rep)
        sc = float(rep.score) if ok else 0.0
        total += sc
        sig.layers.append({"key": k, "name": cfg.LAYER_NAMES[k], "score": sc, "max": mx,
                           "ok": ok, "bias": getattr(rep, "bias", "—") if ok else "n/a"})
    sig.raw_total = round(total, 1)
    sig.score = round(total * cfg.MASTER_SCALE, 1)
    s = 1 if sig.score > 0 else -1 if sig.score < 0 else 0
    tier = _tier_idx(abs(sig.score))
    sig.direction = ("LONG" if s > 0 else "SHORT") if tier > 0 else "NEUTRAL"
    D = sig.direction
    size = 1.0
    block: List[str] = []

    def rule(code, name, effect, detail):
        sig.rules.append({"code": code, "rule": name, "effect": effect, "detail": detail})

    # agreement
    for L in sig.layers:
        if L["ok"] and abs(L["score"]) >= cfg.AGREE_FRAC * L["max"]:
            sig.active += 1
            if np.sign(L["score"]) == s:
                sig.agree += 1

    if D != "NEUTRAL":
        gold_day = xasset.readings.get("gold_day_pct") if _ok(xasset) else None
        # G1
        if _ok(rates):
            if (D == "LONG" and rates.flags.get("yield_spike_up")) or \
                    (D == "SHORT" and rates.flags.get("yield_spike_down")):
                tier -= 1
                rule("G1", "Yield spike against the trade", "downgrade",
                     f"30-min short-end move {rates.readings.get('spike_bp', 0):+.1f} bp")
        # G2
        if _ok(rates) and _ok(dollar) and gold_day is not None:
            ybp = rates.readings.get("10y_day_bp")
            dxy = dollar.readings.get("DXY day")
            if dxy is None and dollar.readings.get("EUR/USD day") is not None:
                dxy = -dollar.readings["EUR/USD day"]
            if ybp is not None and dxy is not None:
                if D == "LONG" and ybp > 0 and dxy > 0 and gold_day <= 0.05:
                    block.append("G2")
                    rule("G2", "Yields and dollar both against the long", "block",
                         f"10y {ybp:+.1f} bp, USD {dxy:+.2f}%, gold {gold_day:+.2f}%")
                if D == "SHORT" and ybp < 0 and dxy < 0 and gold_day >= -0.05:
                    block.append("G2")
                    rule("G2", "Yields and dollar both against the short", "block",
                         f"10y {ybp:+.1f} bp, USD {dxy:+.2f}%, gold {gold_day:+.2f}%")
        # G3
        if _ok(xasset):
            if (D == "LONG" and xasset.flags.get("silver_nonconfirm_high")) or \
                    (D == "SHORT" and xasset.flags.get("silver_nonconfirm_low")):
                tier -= 1
                rule("G3", "Silver not confirming", "downgrade",
                     "gold at a session extreme silver isn't matching")
        # G4
        if D == "LONG" and _ok(flows) and flows.flags.get("cot_crowded_long") and gate is not None \
                and getattr(gate, "mins_to_next_high", None) is not None \
                and gate.mins_to_next_high <= cfg.G4_EVENT_DAYS * 1440:
            size *= 0.5
            rule("G4", "Crowded COT into a major event", "size ×0.5",
                 f"{gate.next_high['name']} within {cfg.G4_EVENT_DAYS} days")
        # G5
        if _ok(liq):
            st_ = liq.asian_state or ""
            if (D == "LONG" and st_.startswith("inside") and "Asia H" in st_) or \
                    (D == "SHORT" and st_.startswith("inside") and "Asia L" in st_):
                block.append("G5")
                rule("G5", "Failed Asian-range breakout", "block",
                     f"Asian range: {st_}")
        if now is not None and xs.session_label(now) == "Asia":
            tier -= 1
            rule("G5", "Asia-session entry — needs London confirmation", "downgrade",
                 "thin liquidity; breakouts often reverse at the London open")
        # G6
        if _ok(regime) and regime.regime == "chop" and _ok(tech):
            z = tech.vwap_z or 0.0
            fade_ok = (D == "LONG" and z <= -cfg.G6_MR_SIGMA) or \
                      (D == "SHORT" and z >= cfg.G6_MR_SIGMA)
            if not fade_ok:
                block.append("G6")
                rule("G6", "Chop regime — only fade the extremes", "block",
                     f"price {z:+.1f}σ from VWAP; needs ≤−{cfg.G6_MR_SIGMA}σ for longs / "
                     f"≥+{cfg.G6_MR_SIGMA}σ for shorts")
        # G7
        if _ok(xasset):
            if (D == "LONG" and xasset.flags.get("structural_bid")) or \
                    (D == "SHORT" and xasset.flags.get("structural_offer")):
                tier += 1
                rule("G7", "Structural bid/offer with the trade", "upgrade",
                     "gold moving against its usual headwinds")
        # G8
        if D == "LONG" and _ok(xasset) and xasset.flags.get("oil_spike_up") and _ok(rates) \
                and (rates.weight or 0) >= 0.5:
            tier -= 1
            rule("G8", "Oil spike while the yield link is live", "downgrade",
                 f"oil {xasset.readings.get('oil_day_pct', 0):+.1f}% → hike risk")
        # G9
        if _ok(opts) and opts.gvz_day_pct is not None and opts.gvz_day_pct >= cfg.G9_GVZ_SPIKE_PCT:
            if (D == "LONG" and opts.flags.get("near_call_wall")) or \
                    (D == "SHORT" and opts.flags.get("near_put_wall")):
                block.append("G9")
                rule("G9", "Vol spike into the option wall", "block",
                     f"GVZ {opts.gvz_day_pct:+.1f}% at the wall — mean-reversion only")
        # G10
        if sig.agree >= cfg.AGREE_UPGRADE:
            tier += 1
            rule("G10", "Broad layer agreement", "upgrade", f"{sig.agree} of 8 layers agree")
        # M1
        if _ok(regime) and regime.regime == "mixed" and tier == 1:
            tier = 0
            rule("M1", "Mixed regime skips C-tier trades", "downgrade", "no clear driver")

    tier = max(0, min(3, tier))
    sig.tier = TIER_NAMES[tier]
    if tier > 0:
        size *= cfg.TIER_SIZE[sig.tier]
    else:
        size = 0.0

    # gates
    if gate is not None and getattr(gate, "ok", False):
        if gate.state == "BLOCKED":
            block.append("GATE")
            sig.notes.append("Entry gate BLOCKED: " + "; ".join(gate.reasons))
        elif gate.state == "CAUTION":
            size *= cfg.CAUTION_SIZE
            sig.notes.append("Entry gate CAUTION: " + "; ".join(gate.cautions))
    if _ok(tech) and tech.atr_regime in cfg.ATR_SIZE:
        size *= cfg.ATR_SIZE[tech.atr_regime]
        sig.notes.append(f"ATR {tech.atr_regime} — size ×{cfg.ATR_SIZE[tech.atr_regime]}")

    sig.blocked_by = block
    if D != "NEUTRAL" and tier > 0 and not block:
        sig.action = D
        sig.size_mult = round(size, 2)
    else:
        sig.action = "WAIT"
        sig.size_mult = 0.0
    sig.plan = build_plan(D, gold, tech, liq, opts, basis) if D != "NEUTRAL" else None
    if sig.plan and _ok(opts) and opts.em_daily and _ok(tech):
        sig.plan["em_daily"] = round(opts.em_daily, 2)
    sig.ok = True
    return sig


def get_master_signal(layers: Dict, gold: pd.DataFrame, gate=None, basis: float = 0.0,
                      now=None) -> MasterSignal:
    try:
        return compute(layers, gold, gate, basis, now)
    except Exception as e:  # noqa: BLE001
        return MasterSignal(error=f"{type(e).__name__}: {e}")
