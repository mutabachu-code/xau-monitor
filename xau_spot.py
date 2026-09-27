"""
xau_spot.py — live XAUUSD spot quote and the GC=F − spot basis.

Sources, tried in order (no API keys):
  1. Swissquote public quote feed — live bid/ask
  2. gold-api.com — single price (can lag, e.g. it holds Friday's close)
Each quote carries its source and age; anything older than SPOT_STALE_SEC is
flagged stale.

Basis = GC=F last − spot mid, measured only while the market is open and the
GC=F bar is fresh. A process-wide store keeps the last BASIS_HISTORY readings
and the median is used, which smooths the timing gap between the two feeds and
survives weekends and new browser sessions. Nothing here raises.
"""
import json
import statistics
import urllib.request
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd

import xau_config as cfg
import xau_sessions as xs

try:
    import streamlit as st
    _spot_cache = st.cache_data(ttl=cfg.SPOT_TTL_SEC, show_spinner=False)
    _store = st.cache_resource(show_spinner=False)
except Exception:  # pragma: no cover
    def _spot_cache(fn):
        return fn

    def _store(fn):
        return fn


# ── Parsers ──────────────────────────────────────────────────────────────────
def parse_swissquote(data) -> Optional[Dict]:
    """[{topo:{platform,server}, spreadProfilePrices:[{spreadProfile,bid,ask}], ts}]"""
    quotes = []
    for item in data or []:
        prices = item.get("spreadProfilePrices") or []
        pick = next((p for p in prices if p.get("spreadProfile") == "prime"), None) or \
            (prices[0] if prices else None)
        if not pick:
            continue
        try:
            bid, ask = float(pick["bid"]), float(pick["ask"])
        except (KeyError, TypeError, ValueError):
            continue
        if bid <= 0 or ask <= 0 or ask < bid:
            continue
        ts = datetime.fromtimestamp(float(item.get("ts", 0)) / 1000, tz=timezone.utc)
        quotes.append({"bid": bid, "ask": ask, "ts": ts,
                       "platform": (item.get("topo") or {}).get("platform", "")})
    if not quotes:
        return None
    q = max(quotes, key=lambda x: x["ts"])          # freshest platform
    return {"bid": q["bid"], "ask": q["ask"], "mid": (q["bid"] + q["ask"]) / 2,
            "spread": q["ask"] - q["bid"], "ts": q["ts"], "source": "Swissquote"}


def parse_goldapi(data) -> Optional[Dict]:
    try:
        px = float(data["price"])
        ts = pd.Timestamp(data["updatedAt"]).to_pydatetime()
    except (KeyError, TypeError, ValueError):
        return None
    if px <= 0:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return {"bid": None, "ask": None, "mid": px, "spread": None, "ts": ts,
            "source": "gold-api"}


PARSERS = {"Swissquote": parse_swissquote, "gold-api": parse_goldapi}


# ── Fetch ────────────────────────────────────────────────────────────────────
def _get_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "xau-monitor/1.0"})
    with urllib.request.urlopen(req, timeout=cfg.SPOT_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


@_spot_cache
def _spot_raw() -> Dict:
    errors = []
    for name, url in cfg.SPOT_SOURCES:
        try:
            q = PARSERS[name](_get_json(url))
            if q:
                q["errors"] = errors
                return q
            errors.append(f"{name}: no usable quote")
        except Exception as e:  # noqa: BLE001
            errors.append(f"{name}: {type(e).__name__}")
    raise RuntimeError("; ".join(errors) or "no spot source")


def fetch_spot(now: Optional[datetime] = None) -> Optional[Dict]:
    """Latest spot quote with age / stale flags, or None."""
    try:
        q = dict(_spot_raw())
    except Exception:  # noqa: BLE001 — failures are not cached
        return None
    now = now or datetime.now(timezone.utc)
    q["age_sec"] = max(0.0, (now - q["ts"]).total_seconds())
    q["stale"] = q["age_sec"] > cfg.SPOT_STALE_SEC
    return q


# ── Basis ────────────────────────────────────────────────────────────────────
def live_basis(gold: Optional[pd.DataFrame], spot: Optional[Dict],
               now: Optional[datetime] = None) -> Dict:
    """One basis reading (GC=F last − spot mid) and whether it can be trusted."""
    now = now or datetime.now(timezone.utc)
    out = {"basis": None, "gc_last": None, "gc_age_min": None, "usable": False, "reason": ""}
    if gold is None or gold.empty:
        out["reason"] = "no GC=F data"
        return out
    out["gc_last"] = float(gold["Close"].iat[-1])
    out["gc_age_min"] = (now - gold.index[-1].to_pydatetime()).total_seconds() / 60
    if spot is None:
        out["reason"] = "no spot quote"
        return out
    b = out["gc_last"] - spot["mid"]
    out["basis"] = round(b, 2)
    if not xs.market_status(now)["open"]:
        out["reason"] = "market closed"
    elif spot.get("stale"):
        out["reason"] = f"spot quote {spot['age_sec']:.0f}s old"
    elif out["gc_age_min"] > cfg.BASIS_GC_MAX_AGE_MIN:
        out["reason"] = f"GC=F bar {out['gc_age_min']:.0f} min old"
    elif not (cfg.BASIS_BOUNDS[0] <= b <= cfg.BASIS_BOUNDS[1]):
        out["reason"] = f"basis {b:+.2f} outside sanity range"
    else:
        out["usable"] = True
    return out


@_store
def basis_store() -> Dict:
    """Shared across sessions in this server process."""
    return {"hist": [], "updated": None}


def record_basis(store: Dict, reading: Dict, now: Optional[datetime] = None) -> Optional[float]:
    """Add a usable reading; return the median of the kept history (or None)."""
    hist: List[float] = store.setdefault("hist", [])
    if reading.get("usable") and reading.get("basis") is not None:
        hist.append(float(reading["basis"]))
        del hist[:-cfg.BASIS_HISTORY]
        store["updated"] = now or datetime.now(timezone.utc)
    return round(statistics.median(hist), 2) if hist else None


def effective_basis(mode: str, manual: float, auto: Optional[float]) -> Dict:
    if mode == "Auto" and auto is not None:
        return {"basis": auto, "source": "live (median of recent readings)"}
    if mode == "Auto":
        return {"basis": manual, "source": "manual fallback (no live reading yet)"}
    return {"basis": manual, "source": "manual"}
