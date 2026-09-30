"""
xau_entry.py — where to enter: confluence of zones, liquidity and gamma, plus a
momentum / chase mode. Replaces "enter at market when the score crosses a tier"
with "enter at the best level in the signal's direction".

1. Candidates on the pullback side of price (below for longs, above for shorts),
   within 3 × 15m ATR:
     MTF zones (S/D, order blocks, FVG, IFVG) · liquidity pools (entry just past
     the pool, i.e. buy the sweep) · gamma levels (weight by greeks strength and
     sign) · option wall · VWAP · CPR TC/P/BC.
2. Overlapping candidates merge into clusters; score = Σ weights, reduced 10%
   per ATR of distance. Best cluster ≥ CLUSTER_MIN_SCORE is the entry location.
3. Limit entry 25% into the cluster from its proximal edge. Stop beyond the
   distal edge — and beyond any liquidity pool sitting just past it, so the
   stop isn't in the sweep. Risk kept between 0.6 and 3 ATR.
4. Targets: next opposing zones / liquidity / gamma / walls. TP1 ≥ 1R, TP2 ≥ 2R
   (else 2.5R). An opposing zone closer than 1R flags a poor-RR "obstacle".
5. Momentum (6 checks: 3-bar displacement, RVOL, ADX, beyond ±1σ VWAP, fresh
   FVG, 3 closes in a row). STRONG momentum with no nearby zone → CHASE:
   enter the fresh FVG retest (or market) at half size.
6. Quality grade A / B / C → strength STRONG / MODERATE / WEAK and a size factor.

Everything is in GC=F prices; `basis` converts to spot.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import xau_config as cfg

GAMMA_W = {"STRONG": 1.5, "MODERATE": 1.0, "WEAK": 0.5}


def _ok(r) -> bool:
    return bool(r is not None and getattr(r, "ok", False))


# ── Momentum ─────────────────────────────────────────────────────────────────
def momentum(direction: str, gold: pd.DataFrame, tech, liq, zones) -> Dict:
    s = 1 if direction == "LONG" else -1
    C = gold["Close"].to_numpy(float)
    atr = float(getattr(tech, "atr", 0) or 0) or 1.0
    checks = {}
    checks["displacement"] = bool(len(C) >= 4 and s * (C[-1] - C[-4]) >= cfg.MOM_DISP_ATR * atr)
    checks["rvol"] = bool(_ok(liq) and (getattr(liq, "rvol_now", 0) or 0) >= cfg.MOM_RVOL)
    checks["adx"] = bool(_ok(tech) and (getattr(tech, "adx", 0) or 0) >= cfg.MOM_ADX)
    checks["beyond VWAP 1σ"] = bool(_ok(tech) and s * (getattr(tech, "vwap_z", 0) or 0) >= 1.0)
    fresh_fvg = None
    if _ok(zones):
        side = "support" if s > 0 else "resistance"
        recent = gold.index[-cfg.CHASE_FVG_BARS] if len(gold) >= cfg.CHASE_FVG_BARS else gold.index[0]
        cands = [z for z in getattr(zones, "zones", []) if z["tf"] == "15m" and z["kind"] == "fvg"
                 and z["side"] == side and z["created"] >= recent]
        if cands:
            fresh_fvg = max(cands, key=lambda z: z["created"])
    checks["fresh FVG"] = fresh_fvg is not None
    checks["3 closes"] = bool(len(C) >= 3 and s * (C[-1] - C[-2]) > 0 and s * (C[-2] - C[-3]) > 0)
    n = sum(checks.values())
    grade = "STRONG" if n >= cfg.MOM_STRONG else "MODERATE" if n >= cfg.MOM_MODERATE else "WEAK"
    return {"score": n, "of": len(checks), "grade": grade, "checks": checks,
            "fresh_fvg": fresh_fvg}


# ── Candidates ───────────────────────────────────────────────────────────────
def _cand(lo, hi, kind, label, weight, tf="", fresh=False, strength=None):
    return {"lo": float(min(lo, hi)), "hi": float(max(lo, hi)), "kind": kind, "label": label,
            "weight": float(weight), "tf": tf, "fresh": fresh, "strength": strength}


def candidates(direction: str, price: float, atr: float, tech, liq, zones, gamma,
               opts, pullback: bool) -> List[Dict]:
    """pullback=True → entry side (below price for longs); False → target side."""
    s = 1 if direction == "LONG" else -1
    want_side = ("support" if s > 0 else "resistance") if pullback else \
        ("resistance" if s > 0 else "support")
    on_side = (lambda x: s * (price - x) >= -0.1 * atr) if pullback else \
        (lambda x: s * (x - price) > 0)
    out: List[Dict] = []
    if _ok(zones):
        for z in getattr(zones, "zones", []):
            if z["side"] != want_side:
                continue
            prox = z["hi"] if want_side == "support" else z["lo"]
            distal = z["lo"] if want_side == "support" else z["hi"]
            # entry side: qualify by the far edge so a zone price is already inside counts
            if on_side(distal if pullback else prox):
                out.append(_cand(z["lo"], z["hi"], z["kind"], z["label"], z["weight"],
                                 z["tf"], z["fresh"]))
    if _ok(liq):
        for lv in getattr(liq, "levels", []) or []:
            x = lv.price
            if not on_side(x):
                continue
            w = cfg.KIND_WEIGHT["liq"] * lv.weight
            stop_side_pool = (lv.side == "sell") if s > 0 else (lv.side == "buy")
            if pullback and stop_side_pool:              # enter just past the pool
                out.append(_cand(x, x - s * 0.3 * atr, "liq", f"{lv.name} pool", w))
            elif pullback:
                out.append(_cand(x - 0.15 * atr, x + 0.15 * atr, "liq", f"{lv.name} (flipped)",
                                 0.6 * w))
            else:
                out.append(_cand(x, x, "liq", f"{lv.name} liquidity", w))
    if _ok(gamma):
        for g in getattr(gamma, "levels", []) or []:
            x = g["price"]
            if not on_side(x):
                continue
            w = cfg.KIND_WEIGHT["gamma"] * GAMMA_W[g["grade"]] * (1.2 if g["gex"] > 0 else 0.8)
            out.append(_cand(x - 0.15 * atr, x + 0.15 * atr, "gamma",
                             f"γ {g['strike']:.0f} {g['grade']}", w, strength=g["grade"]))
    if _ok(opts):
        cw, pw = getattr(opts, "call_wall", None), getattr(opts, "put_wall", None)
        wall = (pw if s > 0 else cw) if pullback else (cw if s > 0 else pw)
        if wall and on_side(wall):
            out.append(_cand(wall - 0.15 * atr, wall + 0.15 * atr, "wall",
                             "put wall" if (s > 0) == pullback else "call wall",
                             cfg.KIND_WEIGHT["wall"]))
    if _ok(tech):
        vw = getattr(tech, "vwap", None)
        if vw and on_side(vw):
            out.append(_cand(vw - 0.1 * atr, vw + 0.1 * atr, "vwap", "VWAP",
                             cfg.KIND_WEIGHT["vwap"]))
        cp = getattr(tech, "cpr", None) or {}
        for k in ("TC", "P", "BC", "R1", "S1"):
            if k in cp and on_side(cp[k]):
                out.append(_cand(cp[k] - 0.1 * atr, cp[k] + 0.1 * atr, "cpr", f"CPR {k}",
                                 cfg.KIND_WEIGHT["cpr"]))
    return out


def clusters(direction: str, price: float, atr: float, cands: List[Dict]) -> List[Dict]:
    """Merge overlapping pullback candidates; nearest first."""
    s = 1 if direction == "LONG" else -1
    depth = lambda x: s * (price - x)                    # ≥0 = further into the pullback
    items = []
    for c in cands:
        d1, d2 = sorted([depth(c["hi"] if s > 0 else c["lo"]),
                         depth(c["lo"] if s > 0 else c["hi"])])
        items.append({**c, "d_prox": d1, "d_dist": d2})
    items.sort(key=lambda c: c["d_prox"])
    out: List[Dict] = []
    for c in items:
        if out and c["d_prox"] <= out[-1]["d_dist"] + cfg.CLUSTER_GAP_ATR * atr:
            cl = out[-1]
            cl["d_dist"] = max(cl["d_dist"], c["d_dist"])
            cl["members"].append(c)
        else:
            out.append({"d_prox": c["d_prox"], "d_dist": c["d_dist"], "members": [c]})
    for cl in out:
        cl["score"] = round(sum(m["weight"] for m in cl["members"]), 2)
        dist_atr = max(0.0, cl["d_prox"]) / atr
        cl["adj"] = round(cl["score"] * max(0.2, 1 - 0.1 * dist_atr), 2)
        cl["dist_atr"] = round(dist_atr, 2)
        cl["prox"] = price - s * cl["d_prox"]
        cl["distal"] = price - s * cl["d_dist"]
        cl["lo"], cl["hi"] = sorted([cl["prox"], cl["distal"]])
        cl["labels"] = [m["label"] for m in sorted(cl["members"], key=lambda m: -m["weight"])]
        cl["htf_fresh"] = any(m["fresh"] and m["tf"] in ("1h", "4h") for m in cl["members"])
        cl["gamma_strong"] = any(m["strength"] == "STRONG" for m in cl["members"])
    return out


# ── Plan ─────────────────────────────────────────────────────────────────────
@dataclass
class EntryPlan:
    ok: bool = False
    error: str = ""
    plan: Optional[Dict] = None
    clusters: List[Dict] = field(default_factory=list)
    momentum: Dict = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


def _targets(direction, entry, R, atr, tech, liq, zones, gamma, opts) -> List[Dict]:
    s = 1 if direction == "LONG" else -1
    cands = candidates(direction, entry, atr, tech, liq, zones, gamma, opts, pullback=False)
    out = []
    for c in cands:
        x = c["lo"] if s > 0 else c["hi"]                # proximal edge of the obstacle
        h = s * (x - entry)
        if h > 0.2 * R:
            out.append({**c, "price": x, "r": h / R})
    return sorted(out, key=lambda t: t["r"])


def build(direction: str, gold: pd.DataFrame, tech, liq, zones, gamma, opts,
          basis: float = 0.0) -> EntryPlan:
    ep = EntryPlan()
    if direction not in ("LONG", "SHORT") or gold is None or gold.empty or not _ok(tech):
        ep.error = "no direction or data"
        return ep
    s = 1 if direction == "LONG" else -1
    price = float(gold["Close"].iat[-1])
    atr = float(tech.atr) or 1.0
    mom = momentum(direction, gold, tech, liq, zones)
    ep.momentum = mom
    maxd = cfg.ENTRY_MAX_DIST_ATR * atr
    cands = [c for c in candidates(direction, price, atr, tech, liq, zones, gamma, opts, True)
             if max(0.0, s * (price - (c["hi"] if s > 0 else c["lo"]))) <= maxd]
    cls = clusters(direction, price, atr, cands)
    ep.clusters = sorted(cls, key=lambda c: -c["adj"])
    best = next((c for c in ep.clusters if c["adj"] >= cfg.CLUSTER_MIN_SCORE), None)
    regime = getattr(gamma, "regime", "unknown") if _ok(gamma) else "unknown"

    chase = mom["grade"] == "STRONG" and (best is None or best["dist_atr"] > cfg.CHASE_MAX_DIST_ATR)
    pools = []
    if _ok(liq):
        pools = [lv.price for lv in (getattr(liq, "levels", []) or [])
                 if (lv.side == "sell" if s > 0 else lv.side == "buy")]

    if chase:
        f = mom["fresh_fvg"]
        if f is not None and s * (price - (f["hi"] if s > 0 else f["lo"])) <= cfg.CHASE_FVG_MAX_ATR * atr:
            entry = (f["lo"] + f["hi"]) / 2
            entry = min(entry, price) if s > 0 else max(entry, price)
            etype, zone_lo, zone_hi = "chase", f["lo"], f["hi"]
            distal = f["lo"] if s > 0 else f["hi"]
            src = "fresh FVG retest (CE)"
        else:
            entry, etype = price, "chase"
            lows = gold["Low"].iloc[-4:].min() if s > 0 else gold["High"].iloc[-4:].max()
            distal, zone_lo, zone_hi = float(lows), *sorted([price, float(lows)])
            src = "market — displacement origin stop"
        label = ["momentum chase"] + ([f["label"]] if f is not None else [])
        cl_score = None
    elif best is not None:
        h = best["hi"] - best["lo"]
        entry = best["prox"] - s * cfg.ENTRY_DEPTH * h
        if s * (price - entry) < 0:                     # price already deeper → take market
            entry = price
        etype = "limit" if s * (price - entry) > 0.05 * atr else "market"
        zone_lo, zone_hi, distal = best["lo"], best["hi"], best["distal"]
        src = " + ".join(best["labels"][:3])
        label, cl_score = best["labels"], best["adj"]
    else:
        ep.plan = {"direction": direction, "entry_type": "none", "grade": "C",
                   "strength": "WEAK", "size_factor": 0.0, "momentum": mom,
                   "gamma_regime": regime, "confluence": [],
                   "reason": "no confluence zone within 3 ATR and momentum not strong — "
                             "wait for price to reach a level"}
        ep.ok = True
        return ep

    # stop: beyond distal edge, then past any pool sitting just beyond it
    sl = distal - s * cfg.STOP_BEYOND_ATR * atr
    stop_src = "beyond zone"
    for p in pools:
        if 0 <= s * (distal - p) <= cfg.LIQ_STOP_GUARD_ATR * atr:
            cand = p - s * cfg.STOP_BEYOND_ATR * atr
            if s * (sl - cand) > 0:
                sl, stop_src = cand, "beyond liquidity pool"
    R = s * (entry - sl)
    if R < cfg.ENTRY_RISK_MIN_ATR * atr:
        sl, R, stop_src = entry - s * cfg.ENTRY_RISK_MIN_ATR * atr, cfg.ENTRY_RISK_MIN_ATR * atr, \
            f"min {cfg.ENTRY_RISK_MIN_ATR} ATR"
    if R > cfg.ENTRY_RISK_MAX_ATR * atr:                # zone too tall → enter deeper
        entry = sl + s * cfg.ENTRY_RISK_MAX_ATR * atr
        R = cfg.ENTRY_RISK_MAX_ATR * atr
        src += f" (entry deepened to cap risk at {cfg.ENTRY_RISK_MAX_ATR} ATR)"
        if etype == "market":
            etype = "limit"

    tg = _targets(direction, entry, R, atr, tech, liq, zones, gamma, opts)
    obstacle = next((t for t in tg if t["r"] < cfg.OBSTACLE_MIN_R and t["kind"] in
                     ("sd", "ob", "fvg", "ifvg") and t["weight"] >= 2), None)
    t1 = next((t for t in tg if t["r"] >= cfg.TP1_MIN_R), None)
    tp1 = (t1["price"] - s * 0.1 * atr) if t1 else entry + s * 1.5 * R
    tp1_src = t1["label"] if t1 else "1.5R"
    t2 = next((t for t in tg if t["r"] >= cfg.TP2_MIN_R and s * (t["price"] - tp1) > 0.3 * R), None)
    tp2 = (t2["price"] - s * 0.1 * atr) if t2 else entry + s * cfg.TP2_DEFAULT_R * R
    tp2_src = t2["label"] if t2 else f"{cfg.TP2_DEFAULT_R}R"
    if s * (tp2 - tp1) <= 0:
        tp2, tp2_src = tp1 + s * R, "TP1 + 1R"

    # quality
    if etype == "chase":
        pts = mom["score"] / 2 + (1.0 if regime == "negative" else 0.0)
    else:
        pts = min(cl_score / 2, 3.0) + (1.0 if best["htf_fresh"] else 0.0) + \
            (1.0 if best["gamma_strong"] else 0.0) + (0.5 if regime == "positive" else 0.0) + \
            (0.5 if mom["grade"] != "WEAK" else 0.0)
    if obstacle:
        pts -= 1.0
        ep.notes.append(f"Opposing {obstacle['label']} at {obstacle['price']:,.2f} is only "
                        f"{obstacle['r']:.1f}R away — expect a reaction there")
    grade = "A" if pts >= 4 else "B" if pts >= 2.5 else "C"
    strength = {"A": "STRONG", "B": "MODERATE", "C": "WEAK"}[grade]
    size = {"A": 1.0, "B": 0.75, "C": 0.5}[grade] * (cfg.CHASE_SIZE if etype == "chase" else 1.0)

    r_ = lambda x: round(float(x), 2)
    ep.plan = {
        "direction": direction, "entry_type": etype, "entry": r_(entry),
        "zone": (r_(zone_lo), r_(zone_hi)), "sl": r_(sl), "tp1": r_(tp1), "tp2": r_(tp2),
        "risk": r_(R), "atr": r_(atr), "rr1": round(abs(tp1 - entry) / R, 2),
        "rr2": round(abs(tp2 - entry) / R, 2), "stop_src": stop_src, "tp1_src": tp1_src,
        "tp2_src": tp2_src, "basis": basis, "entry_src": src, "confluence": label,
        "cluster_score": cl_score, "grade": grade, "strength": strength,
        "size_factor": round(size, 2), "momentum": mom, "gamma_regime": regime,
        "obstacle": None if not obstacle else {"label": obstacle["label"],
                                               "price": r_(obstacle["price"]),
                                               "r": round(obstacle["r"], 2)},
        "distance_atr": round(s * (price - entry) / atr, 2), "price": r_(price),
        "spot": {k: r_(v - basis) for k, v in (("entry", entry), ("sl", sl), ("tp1", tp1),
                                               ("tp2", tp2))},
    }
    if etype == "limit":
        ep.notes.append(f"Limit {'buy' if s > 0 else 'sell'} at {entry:,.2f} "
                        f"({ep.plan['distance_atr']:.1f} ATR from price) — wait for the fill")
    if regime == "negative" and etype == "limit":
        ep.notes.append("Negative gamma: pullbacks may overshoot the zone — keep the stop "
                        "past the pool")
    ep.ok = True
    return ep


def get_entry_plan(direction, gold, tech, liq, zones, gamma, opts, basis=0.0) -> EntryPlan:
    try:
        return build(direction, gold, tech, liq, zones, gamma, opts, basis)
    except Exception as e:  # noqa: BLE001
        return EntryPlan(error=f"{type(e).__name__}: {e}")
