"""
xau_data.py — single batched yfinance download with TTL cache and fallbacks.

One yf.download call fetches gold plus every cross asset. If GC=F comes back
empty, micro gold (MGC=F) is tried on its own. All indexes are returned as
tz-aware UTC. Nothing here raises to the UI: failures land in bundle["errors"].
"""
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd

import xau_bg as xb
import xau_config as cfg
import xau_sessions as xs

try:  # keep the module importable in tests without Streamlit
    import streamlit as st
    _cache = st.cache_data(ttl=cfg.CACHE_TTL_SEC, show_spinner=False)
except Exception:  # pragma: no cover
    def _cache(fn):
        return fn

OHLC = ["Open", "High", "Low", "Close", "Volume"]


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=OHLC, index=pd.DatetimeIndex([], tz="UTC"))


def extract_ticker(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Pull one ticker's OHLCV out of any yfinance column layout, UTC index."""
    if raw is None or raw.empty:
        return _empty()
    df = raw
    if isinstance(raw.columns, pd.MultiIndex):
        lv0 = raw.columns.get_level_values(0)
        lv1 = raw.columns.get_level_values(1)
        if ticker in lv0:
            df = raw[ticker]
        elif ticker in lv1:
            df = raw.xs(ticker, axis=1, level=1)
        else:
            return _empty()
    df = df.rename(columns=lambda c: str(c).title()).copy()
    if "Close" not in df.columns:
        return _empty()
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    df = df[[c for c in OHLC if c in df.columns]].dropna(subset=["Close"])
    idx = pd.DatetimeIndex(df.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    df.index = idx
    return df.sort_index()


def _download(tickers: List[str], interval: str, period: str) -> pd.DataFrame:
    import yfinance as yf
    return yf.download(tickers=tickers, interval=interval, period=period,
                       group_by="ticker", auto_adjust=False, prepost=True,
                       progress=False, threads=True)


def fetch_bundle(interval: str = cfg.SIGNAL_INTERVAL,
                 period: str = cfg.SIGNAL_PERIOD) -> Dict:
    """{'gold': df, 'primary': ticker, 'cross': {ticker: df}, 'errors': [...],
    'fetched_at': utc datetime}"""
    errors: List[str] = []
    extra = list(getattr(cfg, "EXTRA_TICKERS", {}))
    tickers = [cfg.PRIMARY] + list(cfg.CROSS_ASSETS) + extra
    try:
        raw = _download(tickers, interval, period)
    except Exception as e:
        raw = None
        errors.append(f"batch download failed: {type(e).__name__}: {e}")

    gold, primary = extract_ticker(raw, cfg.PRIMARY), cfg.PRIMARY
    if gold.empty:
        for fb in cfg.PRIMARY_FALLBACKS:
            try:
                gold = extract_ticker(_download([fb], interval, period), fb)
            except Exception as e:
                errors.append(f"{fb} fallback failed: {e}")
            if not gold.empty:
                primary = fb
                errors.append(f"{cfg.PRIMARY} empty; using {fb}")
                break
    if gold.empty:
        errors.append("no gold data available")

    cross = {}
    for t in list(cfg.CROSS_ASSETS) + extra:
        d = extract_ticker(raw, t)
        if d.empty:
            errors.append(f"{t} empty")
        cross[t] = d

    return {"gold": gold, "primary": primary, "cross": cross, "errors": errors,
            "fetched_at": datetime.now(timezone.utc)}


# ── Helpers ──────────────────────────────────────────────────────────────────
def to_spot(price: Optional[float], basis: float) -> Optional[float]:
    """Convert a GC=F futures price to broker XAUUSD spot using the basis."""
    return None if price is None else price - basis


def change_vs_prev_close(df: pd.DataFrame) -> Optional[Dict]:
    """Last price and change vs the previous CME trading day's close."""
    if df is None or df.empty:
        return None
    last = float(df["Close"].iloc[-1])
    td = xs.trading_day(df.index[-1].to_pydatetime())
    prev = xs.prev_day_levels(df, td)
    if prev is None:
        return {"last": last, "prev_close": None, "change": None, "pct": None}
    ch = last - prev["close"]
    return {"last": last, "prev_close": prev["close"], "change": ch,
            "pct": ch / prev["close"] * 100 if prev["close"] else None}


def bar_age_minutes(df: pd.DataFrame, now: datetime) -> Optional[float]:
    """Minutes since the last bar opened."""
    if df is None or df.empty:
        return None
    return (now - df.index[-1].to_pydatetime()).total_seconds() / 60


# ── Background, incremental bundle ───────────────────────────────────────────
def _merge_frame(old: pd.DataFrame, new: pd.DataFrame, keep_days: int) -> pd.DataFrame:
    if new is None or new.empty:
        return old
    if old is None or old.empty:
        out = new
    else:
        out = pd.concat([old, new])
        out = out[~out.index.duplicated(keep="last")].sort_index()
    cutoff = out.index[-1] - pd.Timedelta(days=keep_days)
    return out[out.index >= cutoff]


def merge_bundle(old: Dict, new: Dict, keep_days: int = cfg.BUNDLE_KEEP_DAYS) -> Dict:
    """Overlay a short recent download onto the cached history. Newer bars win
    (the last, still-forming bar gets updated)."""
    cross = dict(old.get("cross", {}))
    for t, df in new.get("cross", {}).items():
        cross[t] = _merge_frame(cross.get(t), df, keep_days)
    return {"gold": _merge_frame(old["gold"], new["gold"], keep_days),
            "primary": old["primary"], "cross": cross,
            "errors": [e for e in new.get("errors", []) if "empty" not in e],
            "fetched_at": new.get("fetched_at", datetime.now(timezone.utc)),
            "full_at": old.get("full_at"), "mode": "incremental"}


def _bundle_job() -> Dict:
    import time as _t
    prev = xb.peek("bundle")
    full_due = prev is None or (_t.time() - (prev.get("full_at") or 0)) > cfg.BUNDLE_FULL_REFRESH_SEC
    if not full_due:
        inc = fetch_bundle(cfg.SIGNAL_INTERVAL, cfg.BUNDLE_INCR_PERIOD)
        if not inc["gold"].empty and inc["primary"] == prev["primary"]:
            return merge_bundle(prev, inc)
    full = fetch_bundle(cfg.SIGNAL_INTERVAL, cfg.SIGNAL_PERIOD)
    if full["gold"].empty:
        raise RuntimeError("; ".join(full["errors"]) or "no gold data")
    full["full_at"] = _t.time()
    full["mode"] = "full"
    return full


def get_bundle(wait: float = cfg.BUNDLE_WAIT_FIRST) -> Optional[Dict]:
    """Last good bundle immediately; refreshed in the background every
    CACHE_TTL_SEC (incremental), with a full reload every 6 h."""
    return xb.swr("bundle", _bundle_job, ttl=cfg.CACHE_TTL_SEC, neg_ttl=30, wait=wait)
