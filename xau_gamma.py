"""
xau_gamma.py — dealer gamma levels from the GLD options chain, with greeks.

For each strike within ±8% on the nearest 3 expiries, Black-Scholes greeks are
computed from that strike's own IV: delta, gamma and theta, then aggregated
with open interest:

  net GEX      Σ(call γ·OI − put γ·OI) × 100 × S² × 1%   (dealer $ gamma per 1%)
  theta·OI     $ time decay per day held at the strike (pin pressure into expiry)
  delta        call delta at the strike (0.5 = at the money, where gamma peaks)
  IV           OI-weighted implied vol at the strike

Gamma flip = the price where total dealer gamma changes sign. Above it dealers
are long gamma (they sell rallies / buy dips → mean reversion, pinning); below
it they are short gamma (they chase → moves accelerate).

Level strength (0..1) = 50% |GEX| + 30% OI + 20% theta, each as the strike's share of
the chain total (10% of all gamma, 8% of OI / theta = full marks),
× IV factor (IV below the chain median = firmer level, above = softer)
× expiry factor (≤2 days 1.3, ≤7 days 1.15) × ATM factor (|Δ−0.5| < 0.15: 1.1).
STRONG ≥ 0.60 · MODERATE ≥ 0.35 · else WEAK. Prices are converted to GC=F.
"""
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import xau_config as cfg
import xau_options as xo

SQRT2PI = math.sqrt(2 * math.pi)


def _ncdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def greeks(S: float, K: float, T: float, iv: float, r: float = cfg.OPT_RISK_FREE) -> Dict:
    """Black-Scholes delta/gamma/theta (per day) for a call and a put."""
    if S <= 0 or K <= 0 or T <= 0 or iv <= 0:
        return {"call_delta": 0.0, "put_delta": 0.0, "gamma": 0.0,
                "call_theta": 0.0, "put_theta": 0.0}
    sq = iv * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * iv * iv) * T) / sq
    d2 = d1 - sq
    pdf = math.exp(-0.5 * d1 * d1) / SQRT2PI
    gamma = pdf / (S * sq)
    common = -S * pdf * iv / (2 * math.sqrt(T))
    call_theta = (common - r * K * math.exp(-r * T) * _ncdf(d2)) / 365
    put_theta = (common + r * K * math.exp(-r * T) * _ncdf(-d2)) / 365
    return {"call_delta": _ncdf(d1), "put_delta": _ncdf(d1) - 1, "gamma": gamma,
            "call_theta": call_theta, "put_theta": put_theta}


def weight_mode(chain: Dict) -> Dict:
    """Yahoo often blanks GLD open interest intraday. Use OI when it is there,
    otherwise today's volume as a proxy, otherwise nothing."""
    S = float(chain["spot"])
    lo, hi = S * (1 - cfg.GAMMA_RANGE_PCT / 100), S * (1 + cfg.GAMMA_RANGE_PCT / 100)
    oi = vol = 0.0
    n = n_oi = 0
    for e in chain["expiries"][:cfg.GAMMA_EXPIRIES]:
        for df in (xo._clean(e["calls"]), xo._clean(e["puts"])):
            d = df[(df["strike"] >= lo) & (df["strike"] <= hi)]
            oi += float(d["openInterest"].sum())
            vol += float(d["volume"].sum())
            n += len(d)
            n_oi += int((d["openInterest"] > 0).sum())
    cov = n_oi / n if n else 0.0
    if oi >= cfg.GAMMA_MIN_WEIGHT:
        return {"col": "openInterest", "label": "open interest", "total": oi, "coverage": cov}
    if vol >= cfg.GAMMA_MIN_WEIGHT:
        return {"col": "volume", "label": "volume proxy (Yahoo OI missing)", "total": vol,
                "coverage": cov}
    return {"col": None, "label": "none", "total": max(oi, vol), "coverage": cov}


def strike_greeks(chain: Dict, now: datetime, wcol: str = "openInterest") -> pd.DataFrame:
    """One row per (expiry, strike) with weight (OI or volume), IV and greeks."""
    S = float(chain["spot"])
    rows = []
    for e in chain["expiries"][:cfg.GAMMA_EXPIRIES]:
        dte = xo.days_to_expiry(e["expiry"], now)
        T = dte / 365
        c, p = xo._clean(e["calls"]), xo._clean(e["puts"])
        c = c.rename(columns={wcol: "w"}) if wcol != "w" else c
        p = p.rename(columns={wcol: "w"}) if wcol != "w" else p
        m = pd.merge(c[["strike", "w", "impliedVolatility"]],
                     p[["strike", "w", "impliedVolatility"]],
                     on="strike", how="outer", suffixes=("_c", "_p")).fillna(0.0)
        lo, hi = S * (1 - cfg.GAMMA_RANGE_PCT / 100), S * (1 + cfg.GAMMA_RANGE_PCT / 100)
        m = m[(m["strike"] >= lo) & (m["strike"] <= hi)]
        for _, r in m.iterrows():
            ivc, ivp = r["impliedVolatility_c"], r["impliedVolatility_p"]
            ivs = [v for v in (ivc, ivp) if v > 0.01]
            if not ivs:
                continue
            gc = greeks(S, r["strike"], T, ivc) if ivc > 0.01 else greeks(S, r["strike"], T, ivs[0])
            gp = greeks(S, r["strike"], T, ivp) if ivp > 0.01 else greeks(S, r["strike"], T, ivs[0])
            oc, op = r["w_c"], r["w_p"]
            if oc + op <= 0:
                continue                                  # empty strike: never a level
            rows.append({
                "expiry": e["expiry"], "dte": dte, "strike": float(r["strike"]),
                "call_oi": oc, "put_oi": op,
                "iv": (ivc * oc + ivp * op) / (oc + op) if oc + op > 0 else float(np.mean(ivs)),
                "call_delta": gc["call_delta"], "put_delta": gp["put_delta"],
                "gex": (gc["gamma"] * oc - gp["gamma"] * op) * 100 * S * S * 0.01,
                "gross": (gc["gamma"] * oc + gp["gamma"] * op) * 100 * S * S * 0.01,
                "theta_oi": abs(gc["call_theta"] * oc + gp["put_theta"] * op) * 100,
            })
    return pd.DataFrame(rows)


def _chain_arrays(chain: Dict, now: datetime, wcol: str = "openInterest"):
    ks, ois, ivs, sg, Ts = [], [], [], [], []
    for e in chain["expiries"][:cfg.GAMMA_EXPIRIES]:
        T = xo.days_to_expiry(e["expiry"], now) / 365
        for df, sgn in ((xo._clean(e["calls"]), 1.0), (xo._clean(e["puts"]), -1.0)):
            ok = (df[wcol] > 0) & (df["impliedVolatility"] > 0.01)
            d = df[ok]
            ks.append(d["strike"].to_numpy(float))
            ois.append(d[wcol].to_numpy(float))
            ivs.append(d["impliedVolatility"].to_numpy(float))
            sg.append(np.full(len(d), sgn))
            Ts.append(np.full(len(d), T))
    cat = lambda xs: np.concatenate(xs) if xs else np.array([])
    return cat(ks), cat(ois), cat(ivs), cat(sg), cat(Ts)


def total_gex_curve(chain: Dict, prices: np.ndarray, now: datetime,
                    wcol: str = "openInterest") -> np.ndarray:
    """Total dealer $ gamma per 1% at each hypothetical price (vectorised)."""
    K, OI, IV, SG, T = _chain_arrays(chain, now, wcol)
    if not len(K):
        return np.zeros(len(prices))
    P = np.asarray(prices, float)[:, None]
    sq = IV * np.sqrt(T)
    d1 = (np.log(P / K) + (cfg.OPT_RISK_FREE + 0.5 * IV * IV) * T) / sq
    gamma = np.exp(-0.5 * d1 * d1) / SQRT2PI / (P * sq)
    return (gamma * OI * SG).sum(axis=1) * 100 * prices * prices * 0.01


def total_gex_at(chain: Dict, price: float, now: datetime, wcol: str = "openInterest") -> float:
    return float(total_gex_curve(chain, np.array([price]), now, wcol)[0])


def gamma_flip(chain: Dict, now: datetime, wcol: str = "openInterest") -> Optional[float]:
    S = float(chain["spot"])
    grid = np.linspace(S * 0.94, S * 1.06, cfg.GAMMA_FLIP_GRID)
    vals = total_gex_curve(chain, grid, now, wcol)
    sgn = np.sign(vals)
    cross = np.nonzero(sgn[:-1] * sgn[1:] < 0)[0]
    if not len(cross):
        return None
    flips = [grid[i] + (grid[i + 1] - grid[i]) * (-vals[i] / (vals[i + 1] - vals[i]))
             for i in cross]
    return float(min(flips, key=lambda f: abs(f - S)))


_CACHE: Dict = {"key": None}


@dataclass
class GammaReport:
    ok: bool = False
    error: str = ""
    ratio: Optional[float] = None
    flip: Optional[float] = None               # GC=F terms
    regime: str = "unknown"                    # positive | negative
    total_gex: Optional[float] = None
    levels: List[Dict] = field(default_factory=list)
    weight_mode: str = ""                      # "open interest" | "volume proxy …"
    oi_coverage: Optional[float] = None        # share of strikes with OI > 0
    notes: List[str] = field(default_factory=list)


def classify(x: float) -> str:
    return "STRONG" if x >= cfg.GAMMA_STRONG else "MODERATE" if x >= cfg.GAMMA_MODERATE \
        else "WEAK"


def compute(chain: Optional[Dict], gold_last: float, now: Optional[datetime] = None) -> GammaReport:
    rep = GammaReport()
    now = now or datetime.now(timezone.utc)
    if not chain or not chain.get("expiries") or not chain.get("spot"):
        rep.error = "GLD options chain unavailable"
        return rep
    S = float(chain["spot"])
    ratio = gold_last / S
    rep.ratio = ratio
    wm = weight_mode(chain)
    rep.weight_mode, rep.oi_coverage = wm["label"], wm["coverage"]
    if wm["col"] is None:
        rep.error = (f"GLD open interest and volume both missing from Yahoo "
                     f"({wm['total']:,.0f} contracts in range, OI on {wm['coverage']:.0%} of "
                     "strikes) — gamma levels suppressed")
        return rep
    key = (id(chain), chain.get("fetched"), now.strftime("%Y%m%d%H"), wm["col"])
    if _CACHE.get("key") == key:
        g, flip_gld, tot = _CACHE["g"], _CACHE["flip"], _CACHE["tot"]
    else:
        g = strike_greeks(chain, now, wm["col"])
        try:
            tot = total_gex_at(chain, S, now, wm["col"])
            flip_gld = gamma_flip(chain, now, wm["col"])
        except Exception:  # noqa: BLE001
            tot, flip_gld = None, None
        _CACHE.update(key=key, g=g, flip=flip_gld, tot=tot)
    if g.empty:
        rep.error = "no strikes with open interest / volume in range"
        return rep
    per = g.groupby("strike").agg(
        gex=("gex", "sum"), call_oi=("call_oi", "sum"), put_oi=("put_oi", "sum"),
        theta_oi=("theta_oi", "sum"), gross=("gross", "sum"), dte=("dte", "min"),
        iv=("iv", "mean"), call_delta=("call_delta", "first")).reset_index()
    per["oi"] = per["call_oi"] + per["put_oi"]
    _sum = lambda s: float(s.abs().sum()) or 1.0
    F = cfg.GAMMA_SHARE_FULL
    share = lambda col, key: np.clip(per[col].abs() / _sum(per[col]) / F[key], 0, 1)
    base = 0.5 * share("gex", "gex") + 0.3 * share("oi", "oi") + 0.2 * share("theta_oi", "theta")
    med_iv = float(per["iv"].median()) or 0.2
    ivf = np.clip(med_iv / per["iv"].replace(0, med_iv), 0.7, 1.3)
    expf = np.where(per["dte"] <= 2, 1.3, np.where(per["dte"] <= 7, 1.15, 1.0))
    atmf = np.where((per["call_delta"] - 0.5).abs() < 0.15, 1.1, 1.0)
    per["strength"] = np.clip(base * ivf * expf * atmf, 0, 1)
    # a level needs contracts and meaningful gamma (deep ITM/OTM strikes carry ~none)
    per = per[(per["oi"] > 0) & (per["gross"] > 1e-3 * float(per["gross"].max() or 1.0))]
    if per.empty:
        rep.error = "no strikes with meaningful gamma exposure"
        return rep
    # rank by gross gamma: a strike with heavy calls AND puts still pins even if net ≈ 0
    top = per.reindex(per["gross"].sort_values(ascending=False).index).head(cfg.GAMMA_TOP_N)
    for _, r in top.iterrows():
        pos = r["gex"] > 0
        rep.levels.append({
            "strike": float(r["strike"]), "price": float(r["strike"] * ratio),
            "gex": float(r["gex"]), "call_oi": float(r["call_oi"]), "put_oi": float(r["put_oi"]),
            "iv": float(r["iv"]), "delta": float(r["call_delta"]),
            "theta_day": float(r["theta_oi"]), "dte": float(r["dte"]),
            "strength": round(float(r["strength"]), 2), "grade": classify(float(r["strength"])),
            "role": "dealer long γ — magnet / fades moves" if pos else
                    "dealer short γ — accelerates through",
            "side": "support" if r["strike"] < S else "resistance",
        })
    rep.levels.sort(key=lambda l: l["price"])
    rep.total_gex = tot
    rep.flip = None if flip_gld is None else float(flip_gld * ratio)
    if rep.flip is not None:
        rep.regime = "positive" if gold_last > rep.flip else "negative"
    elif rep.total_gex is not None:
        rep.regime = "positive" if rep.total_gex > 0 else "negative"
    if wm["col"] == "volume":
        rep.notes.append("Yahoo's GLD open interest is missing right now — levels are "
                         "weighted by today's options volume instead (less reliable)")
    rep.notes.append("Positive gamma: dealers fade moves — favour limit entries at zones"
                     if rep.regime == "positive" else
                     "Negative gamma: dealers chase — moves extend, favour momentum entries"
                     if rep.regime == "negative" else "Gamma regime unknown")
    rep.ok = True
    return rep


def get_gamma_report(chain: Optional[Dict], gold_last: float,
                     now: Optional[datetime] = None) -> GammaReport:
    try:
        return compute(chain, gold_last, now)
    except Exception as e:  # noqa: BLE001
        return GammaReport(error=f"{type(e).__name__}: {e}")
