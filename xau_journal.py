"""
xau_journal.py — forward-test ledger for the master signal (Phase 7).

Opening: one trade per 15m bar when the master action is LONG/SHORT and no
trade in that direction is open. An opposite signal closes the open trade at
market ("reversed").

Management (per 15m bar after entry, conservative): stop first if a bar
touches both stop and target. At TP1 half the position is taken and the stop
moves to breakeven; the rest runs to TP2 or back to entry. After
JOURNAL_EXPIRY_BARS the remainder closes at market.

R-multiples: SL = −1R; TP1 then BE = 0.5·rr1; TP1 then TP2 = 0.5·rr1 + 0.5·rr2.
Win = TP1 reached before the stop (the ">75%" forward-test metric); net R
is reported alongside because a high win rate with small wins can still lose.

Persistence: CSV next to the app. Streamlit Cloud storage is wiped on reboot
or redeploy, so the dashboard offers download / restore of the file.
"""
import io
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import numpy as np
import pandas as pd

import xau_config as cfg

COLUMNS = ["id", "opened_utc", "bar_utc", "direction", "tier", "score", "entry", "sl",
           "tp1", "tp2", "risk", "rr1", "rr2", "regime", "status", "tp1_hit",
           "closed_utc", "exit", "r_mult", "order", "grade", "filled_utc", "entry_src"]
OPEN = "open"
PENDING = "pending"
LIVE = (OPEN, PENDING)
NOT_TRADES = ("cancelled", "missed", "replaced")     # never filled → excluded from stats


def default_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg.JOURNAL_CSV)


def empty() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


def load(path: Optional[str] = None) -> pd.DataFrame:
    path = path or default_path()
    try:
        df = pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return empty()
    return normalize(df)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[COLUMNS].copy()
    for c in ("opened_utc", "bar_utc", "closed_utc", "filled_utc"):
        df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    df["tp1_hit"] = df["tp1_hit"].astype(str).str.lower().isin(["true", "1", "1.0"])
    df["order"] = df["order"].fillna("market").astype(str)
    for c in ("grade", "entry_src"):
        df[c] = df[c].astype(object)
    for c in ("score", "entry", "sl", "tp1", "tp2", "risk", "rr1", "rr2", "exit", "r_mult"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def save(df: pd.DataFrame, path: Optional[str] = None) -> bool:
    try:
        df.to_csv(path or default_path(), index=False)
        return True
    except Exception:  # noqa: BLE001
        return False


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def from_csv_bytes(data: bytes) -> pd.DataFrame:
    return normalize(pd.read_csv(io.BytesIO(data)))


# ── Trade life-cycle ─────────────────────────────────────────────────────────
def _r(direction: str, entry: float, price: float, risk: float) -> float:
    s = 1 if direction == "LONG" else -1
    return s * (price - entry) / risk if risk else 0.0


def _close(df, i, status, when, exit_px, r_mult):
    df.at[i, "status"] = status
    df.at[i, "closed_utc"] = when
    df.at[i, "exit"] = np.nan if exit_px is None else round(float(exit_px), 2)
    df.at[i, "r_mult"] = np.nan if r_mult is None else round(float(r_mult), 2)


def update(df: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    """Replay every live order from its signal bar (deterministic each run).

    Pending limit: fills when price trades through the entry within
    PENDING_BARS; 'missed' if TP1 prints first; 'cancelled' on timeout.
    On the fill bar only the stop is checked (conservative). Filled trades:
    stop first if a bar touches both; half at TP1 with the stop to entry;
    rest to TP2, back to entry, or market close after JOURNAL_EXPIRY_BARS."""
    if df.empty or gold is None or gold.empty:
        return df
    df = df.copy()
    part = cfg.JOURNAL_TP1_PART
    for i in df.index[df["status"].isin(LIVE)]:
        t = df.loc[i]
        s = 1 if t["direction"] == "LONG" else -1
        bars = gold[gold.index > t["bar_utc"]]
        filled = t["order"] != "limit"
        df.at[i, "status"] = OPEN if filled else PENDING
        df.at[i, "tp1_hit"] = False
        if not filled:
            df.at[i, "filled_utc"] = pd.NaT
        tp1_hit, since_fill = False, 0
        for n, (ts, b) in enumerate(bars.iterrows(), start=1):
            hi, lo = float(b["High"]), float(b["Low"])
            adverse = lo if s > 0 else hi
            favour = hi if s > 0 else lo
            if not filled:
                if n > cfg.PENDING_BARS:
                    _close(df, i, "cancelled", ts, None, None)
                    break
                if (adverse <= t["entry"]) if s > 0 else (adverse >= t["entry"]):
                    filled = True
                    df.at[i, "status"] = OPEN
                    df.at[i, "filled_utc"] = ts
                    if (adverse <= t["sl"]) if s > 0 else (adverse >= t["sl"]):
                        _close(df, i, "sl", ts, t["sl"], -1.0)
                        break
                    continue
                if (favour >= t["tp1"]) if s > 0 else (favour <= t["tp1"]):
                    _close(df, i, "missed", ts, None, None)
                    break
                continue
            since_fill += 1
            stop = t["entry"] if tp1_hit else t["sl"]
            if (adverse <= stop) if s > 0 else (adverse >= stop):   # stop first
                if tp1_hit:
                    _close(df, i, "tp1+be", ts, stop, part * t["rr1"])
                else:
                    _close(df, i, "sl", ts, stop, -1.0)
                break
            if not tp1_hit and ((favour >= t["tp1"]) if s > 0 else (favour <= t["tp1"])):
                tp1_hit = True
                df.at[i, "tp1_hit"] = True
            if tp1_hit and ((favour >= t["tp2"]) if s > 0 else (favour <= t["tp2"])):
                _close(df, i, "tp2", ts, t["tp2"], part * t["rr1"] + (1 - part) * t["rr2"])
                break
            if since_fill >= cfg.JOURNAL_EXPIRY_BARS:
                px = float(b["Close"])
                rr = _r(t["direction"], t["entry"], px, t["risk"])
                r = part * t["rr1"] + (1 - part) * rr if tp1_hit else rr
                _close(df, i, "expired", ts, px, r)
                break
    return df


def maybe_open(df: pd.DataFrame, sig, gold: pd.DataFrame, regime: str = "",
               now: Optional[datetime] = None) -> pd.DataFrame:
    """Record the master signal if it is actionable and new.

    Limit plans become pending orders. A pending order in the same direction is
    replaced only if the new entry is > 0.5R away; opposite signals reverse open
    trades and cancel opposite pending orders."""
    if sig is None or not getattr(sig, "ok", False) or sig.action not in ("LONG", "SHORT") \
            or not sig.plan or gold is None or gold.empty:
        return df
    p = sig.plan
    if p.get("entry_type") == "none":
        return df
    df = df.copy()
    bar = gold.index[-1]
    now = now or datetime.now(timezone.utc)
    price = float(gold["Close"].iat[-1])
    same = df[(df["status"].isin(LIVE)) & (df["direction"] == sig.action)]
    if ((same["status"] == OPEN)).any():
        return df                                           # already in this direction
    if ((df["bar_utc"] == bar) & (df["direction"] == sig.action)).any():
        return df                                           # already logged this bar
    for i in same.index:                                    # pending, same direction
        if abs(df.at[i, "entry"] - p["entry"]) <= 0.5 * df.at[i, "risk"]:
            return df
        _close(df, i, "replaced", bar, None, None)
    part = cfg.JOURNAL_TP1_PART
    for i in df.index[(df["status"].isin(LIVE)) & (df["direction"] != sig.action)]:
        t = df.loc[i]
        if t["status"] == PENDING:
            _close(df, i, "cancelled", bar, None, None)
            continue
        rr = _r(t["direction"], t["entry"], price, t["risk"])
        r = part * t["rr1"] + (1 - part) * rr if bool(t["tp1_hit"]) else rr
        _close(df, i, "reversed", bar, price, r)
    s = 1 if sig.action == "LONG" else -1
    is_limit = p.get("entry_type") in ("limit", "chase") and s * (price - p["entry"]) > 1e-9
    row = {"id": f"{bar:%Y%m%d%H%M}-{sig.action[0]}", "opened_utc": now, "bar_utc": bar,
           "direction": sig.action, "tier": sig.tier, "score": sig.score,
           "entry": p["entry"], "sl": p["sl"], "tp1": p["tp1"], "tp2": p["tp2"],
           "risk": p["risk"], "rr1": p["rr1"], "rr2": p["rr2"], "regime": regime,
           "status": PENDING if is_limit else OPEN, "tp1_hit": False,
           "closed_utc": pd.NaT, "exit": np.nan, "r_mult": np.nan,
           "order": "limit" if is_limit else "market", "grade": p.get("grade", ""),
           "filled_utc": pd.NaT if is_limit else bar, "entry_src": p.get("entry_src", "")}
    new = pd.DataFrame([row])
    return normalize(new if df.empty else pd.concat([df, new], ignore_index=True))


def stats(df: pd.DataFrame, now: Optional[datetime] = None) -> Dict:
    now = now or datetime.now(timezone.utc)
    closed = df[~df["status"].isin(LIVE + NOT_TRADES)]
    n = len(closed)
    wins = int(closed["tp1_hit"].sum()) if n else 0
    first = df["opened_utc"].min() if len(df) else None
    days = 0 if first is None or pd.isna(first) else max(0, (now - first.to_pydatetime()).days)
    by_tier, by_grade = {}, {}
    for key, out in (("tier", by_tier), ("grade", by_grade)):
        for t, g in closed.groupby(key):
            out[t] = {"n": len(g), "win_rate": float(g["tp1_hit"].mean()),
                      "net_r": float(g["r_mult"].sum())}
    limits = df[df["order"] == "limit"]
    resolved = limits[~limits["status"].isin(LIVE)]
    fill_rate = float((~resolved["status"].isin(NOT_TRADES)).mean()) if len(resolved) else None
    return {"closed": n, "open": int((df["status"] == OPEN).sum()),
            "pending": int((df["status"] == PENDING).sum()),
            "unfilled": int(df["status"].isin(NOT_TRADES).sum()), "fill_rate": fill_rate,
            "wins": wins, "win_rate": wins / n if n else None,
            "net_r": float(closed["r_mult"].sum()) if n else 0.0,
            "avg_r": float(closed["r_mult"].mean()) if n else None,
            "days": days, "days_target": cfg.FORWARD_TEST_DAYS,
            "target": cfg.WIN_RATE_TARGET, "by_tier": by_tier, "by_grade": by_grade,
            "on_track": bool(n >= 20 and wins / n >= cfg.WIN_RATE_TARGET)}


def step(sig, gold: pd.DataFrame, regime: str = "", path: Optional[str] = None,
         df: Optional[pd.DataFrame] = None, now: Optional[datetime] = None):
    """Load → update open trades → maybe open a new one → save. Never raises."""
    try:
        df = load(path) if df is None else df
        df = update(df, gold)
        df = maybe_open(df, sig, gold, regime, now)
        saved = save(df, path)
        return df, saved, ""
    except Exception as e:  # noqa: BLE001
        return (df if df is not None else empty()), False, f"{type(e).__name__}: {e}"
