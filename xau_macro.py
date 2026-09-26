"""
xau_macro.py — shared helpers for the macro layers (L1 rates, L2 dollar).

- aligned_returns / corr_stats: gold vs another asset on common 15m bars,
  ignoring returns that span a gap (weekend, daily break).
- dynamic_weight: turns the blended 5d/20d inverse correlation into a 0..1
  layer weight, so a layer fades out when gold stops trading off it.
- fetch_fred: daily FRED CSV with a 6h cache; failures return None and are
  not cached, so a transient outage doesn't stick.
- MacroReport: common report shape for L1 and L2.
"""
import io
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

import xau_config as cfg
import xau_data as xd

try:
    import streamlit as st
    _fred_cache = st.cache_data(ttl=cfg.FRED_TTL_SEC, show_spinner=False)
except Exception:  # pragma: no cover
    def _fred_cache(fn):
        return fn


# ── Small helpers ────────────────────────────────────────────────────────────
def clip1(x: float) -> float:
    return float(max(-1.0, min(1.0, x)))


def usable(df: Optional[pd.DataFrame], gold: pd.DataFrame,
           stale_min: float = cfg.MACRO_STALE_MIN) -> bool:
    """Has data and its last bar is not much older than gold's last bar."""
    if df is None or df.empty or gold is None or gold.empty:
        return False
    lag = (gold.index[-1] - df.index[-1]).total_seconds() / 60
    return lag <= stale_min


def day_change_pct(df: pd.DataFrame) -> Optional[float]:
    ch = xd.change_vs_prev_close(df)
    return None if not ch or ch["pct"] is None else float(ch["pct"])


def day_change_abs(df: pd.DataFrame) -> Optional[float]:
    ch = xd.change_vs_prev_close(df)
    return None if not ch or ch["change"] is None else float(ch["change"])


def recent_change_pct(df: pd.DataFrame, bars: int) -> Optional[float]:
    if df is None or len(df) <= bars:
        return None
    a, b = df["Close"].iat[-1 - bars], df["Close"].iat[-1]
    return float((b / a - 1) * 100) if a else None


def normalize_yield(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance has quoted ^TNX-style indexes both as 4.95 and 49.5."""
    if df is None or df.empty:
        return df
    if df["Close"].median() > 20:
        df = df.copy()
        for c in ("Open", "High", "Low", "Close"):
            df[c] = df[c] / 10
    return df


def fut_pct_to_bp(pct: Optional[float], duration: float) -> Optional[float]:
    """Treasury futures % move → approximate yield change in bp (sign flipped)."""
    return None if pct is None else -pct * 100 / duration


# ── Correlation ──────────────────────────────────────────────────────────────
def aligned_returns(gold: pd.DataFrame, other: pd.DataFrame,
                    transform: Callable[[pd.Series], pd.Series]) -> pd.DataFrame:
    """Columns g (gold % return) and o (transformed other change), common bars,
    gap-spanning returns removed."""
    j = pd.concat([gold["Close"].rename("g"), other["Close"].rename("o")],
                  axis=1, join="inner").dropna()
    if len(j) < 3:
        return pd.DataFrame(columns=["g", "o"])
    out = pd.DataFrame({"g": j["g"].pct_change(), "o": transform(j["o"])}, index=j.index)
    gap = j.index.to_series().diff() > pd.Timedelta(minutes=30)
    return out[~gap].dropna()


def corr_stats(r: pd.DataFrame) -> Dict:
    """Correlation over the last 5 and 20 trading days of bars + rolling series."""
    short_n = cfg.CORR_SHORT_DAYS * cfg.BARS_PER_DAY
    long_n = cfg.CORR_LONG_DAYS * cfg.BARS_PER_DAY
    out = {"short": None, "long": None, "n": len(r), "series": None}
    if len(r) >= cfg.CORR_MIN_POINTS:
        s = r.iloc[-short_n:]
        l = r.iloc[-long_n:]
        if s["o"].std() > 0 and s["g"].std() > 0:
            out["short"] = float(s["g"].corr(s["o"]))
        if l["o"].std() > 0 and l["g"].std() > 0:
            out["long"] = float(l["g"].corr(l["o"]))
        roll = r["g"].rolling(short_n, min_periods=cfg.CORR_MIN_POINTS).corr(r["o"])
        out["series"] = roll.dropna()
    return out


def dynamic_weight(inv_short: Optional[float], inv_long: Optional[float]) -> Optional[float]:
    """inv_* are correlations oriented so positive = the textbook relationship
    (e.g. -corr(gold, yields)). Returns 0..1, or None when no correlation."""
    vals = [(inv_long, cfg.CORR_BLEND_LONG), (inv_short, 1 - cfg.CORR_BLEND_LONG)]
    vals = [(v, w) for v, w in vals if v is not None and not np.isnan(v)]
    if not vals:
        return None
    blended = sum(v * w for v, w in vals) / sum(w for _, w in vals)
    return float(max(0.0, min(1.0, blended / cfg.CORR_FULL)))


# ── FRED ─────────────────────────────────────────────────────────────────────
@_fred_cache
def _fred_raw(sid: str) -> pd.Series:
    url = cfg.FRED_CSV.format(sid=sid)
    req = urllib.request.Request(url, headers={"User-Agent": "xau-monitor/1.0"})
    with urllib.request.urlopen(req, timeout=cfg.FRED_TIMEOUT) as resp:
        text = resp.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(text))
    date_col = df.columns[0]
    s = pd.to_numeric(df[sid], errors="coerce")
    s.index = pd.to_datetime(df[date_col])
    s = s.dropna()
    if s.empty:
        raise ValueError(f"FRED {sid} returned no data")
    return s


def fetch_fred(sid: str) -> Optional[pd.Series]:
    try:
        return _fred_raw(sid)
    except Exception:  # noqa: BLE001 — failure is not cached
        return None


# ── Report ───────────────────────────────────────────────────────────────────
@dataclass
class MacroReport:
    layer: str
    max_pts: int
    ok: bool = False
    error: str = ""
    score: float = 0.0
    raw: float = 0.0                 # −1..+1, gold-bullish positive, before weighting
    weight: Optional[float] = None   # 0..1 from correlation
    eff_max: float = 0.0
    bias: str = "NEUTRAL"
    components: Dict[str, float] = field(default_factory=dict)   # −1..+1 each
    details: Dict[str, str] = field(default_factory=dict)
    corr_short: Optional[float] = None
    corr_long: Optional[float] = None
    corr_source: str = ""
    corr_expected: str = ""
    corr_series: Optional[pd.Series] = None
    readings: Dict[str, Optional[float]] = field(default_factory=dict)
    flags: Dict[str, bool] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


def finish(rep: MacroReport, weights: Dict[str, float], fallback_weight: float) -> MacroReport:
    """Blend available components, apply correlation weight, set score and bias."""
    present = {k: v for k, v in rep.components.items() if v is not None}
    if not present:
        rep.ok, rep.error = False, "no usable inputs"
        return rep
    tot_w = sum(weights[k] for k in present)
    rep.raw = float(sum(weights[k] * v for k, v in present.items()) / tot_w)
    w = rep.weight
    if w is None:
        w = fallback_weight
        rep.notes.append(f"Not enough overlapping bars for correlation — using "
                         f"fallback weight {fallback_weight:.1f}")
    rep.eff_max = round(rep.max_pts * w, 1)
    rep.score = round(max(-rep.max_pts, min(rep.max_pts, rep.raw * rep.max_pts * w)), 1)
    cut = cfg.MACRO_BIAS_FRAC * rep.max_pts
    rep.bias = "LONG" if rep.score >= cut else "SHORT" if rep.score <= -cut else "NEUTRAL"
    rep.ok = True
    return rep
