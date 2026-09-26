"""
xau_sessions.py — DST-aware session clock for gold.

Every venue time is built in its home timezone (London, New York, Shanghai)
and converted to UTC / EAT, so the schedule shifts automatically when the UK
(last Sunday of October) and the US (first Sunday of November) change clocks.

CME gold trading day: 18:00 ET (previous calendar day) -> 17:00 ET.
"""
from datetime import date, datetime, time, timedelta
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd

import xau_config as cfg

UTC = ZoneInfo("UTC")
ET = ZoneInfo(cfg.ET)
LDN = ZoneInfo(cfg.LONDON)
DISP = ZoneInfo(cfg.DISPLAY_TZ)


def _at(d: date, t: time, tz: ZoneInfo) -> datetime:
    """Aware datetime for local date/time in tz (zoneinfo applies correct DST)."""
    return datetime.combine(d, t, tzinfo=tz)


def _utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        raise ValueError("naive datetime; pass timezone-aware timestamps")
    return ts.astimezone(UTC)


# ── Trading day ──────────────────────────────────────────────────────────────
def trading_day(ts: datetime) -> date:
    """CME trading-day label: shift ET time +6h so 18:00 ET rolls to next date."""
    return (_utc(ts).astimezone(ET) + timedelta(hours=6)).date()


def trading_day_index(idx: pd.DatetimeIndex) -> pd.Index:
    """Vectorised trading_day for a tz-aware index."""
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    shifted = idx.tz_convert(cfg.ET) + pd.Timedelta(hours=6)
    return pd.Index(shifted.date, name="trading_day")


# ── Schedule ─────────────────────────────────────────────────────────────────
def schedule_for_display_date(d: date) -> List[Dict]:
    """All session events whose start falls on EAT calendar date d, sorted."""
    out = []
    for ev in cfg.SESSION_EVENTS:
        tz = ZoneInfo(ev.tz)
        for offset in (-1, 0, 1):
            home = d + timedelta(days=offset)
            if home.weekday() not in ev.days:
                continue
            start = _at(home, ev.start, tz).astimezone(DISP)
            if start.date() != d:
                continue
            end = _at(home, ev.end, tz).astimezone(DISP) if ev.end else None
            out.append({
                "key": ev.key, "label": ev.label, "kind": ev.kind,
                "start": start, "end": end, "note": ev.note,
            })
    return sorted(out, key=lambda e: e["start"])


def next_event(now: datetime) -> Optional[Dict]:
    """First scheduled event strictly after now (looks up to 3 EAT days ahead)."""
    now_disp = _utc(now).astimezone(DISP)
    for add in range(4):
        for ev in schedule_for_display_date(now_disp.date() + timedelta(days=add)):
            if ev["start"] > now_disp:
                return ev
    return None


# ── Live state ───────────────────────────────────────────────────────────────
def market_status(now: datetime) -> Dict:
    """open/closed state for CME gold at time now."""
    et = _utc(now).astimezone(ET)
    wd, t = et.weekday(), et.time()
    if wd == 5 or (wd == 4 and t >= time(17)) or (wd == 6 and t < time(18)):
        return {"open": False, "state": "Closed (weekend)"}
    if wd in (0, 1, 2, 3) and time(17) <= t < time(18):
        return {"open": False, "state": "Closed (CME daily break)"}
    return {"open": True, "state": "Open"}


def _in_window(now_utc: datetime, tz: ZoneInfo, window) -> bool:
    local = now_utc.astimezone(tz)
    if local.weekday() >= 5:
        return False
    start = _at(local.date(), window[0], tz)
    end = _at(local.date(), window[1], tz)
    return start <= now_utc < end


def session_label(now: datetime) -> str:
    """Asia / London / London–NY overlap / New York, or Closed."""
    now_utc = _utc(now)
    status = market_status(now_utc)
    if not status["open"]:
        return "Closed"
    in_ldn = _in_window(now_utc, LDN, cfg.LONDON_WINDOW)
    in_ny = _in_window(now_utc, ET, cfg.NY_WINDOW)
    if in_ldn and in_ny:
        return "London–NY overlap"
    if in_ldn:
        return "London"
    if in_ny:
        return "New York"
    return "Asia"


def session_windows(start_utc: datetime, end_utc: datetime) -> List[Dict]:
    """London, New York and overlap windows (UTC) intersecting [start, end]."""
    out = []
    d = _utc(start_utc).astimezone(ET).date() - timedelta(days=1)
    last = _utc(end_utc).astimezone(ET).date() + timedelta(days=1)
    while d <= last:
        if d.weekday() < 5:
            l0, l1 = _at(d, cfg.LONDON_WINDOW[0], LDN), _at(d, cfg.LONDON_WINDOW[1], LDN)
            n0, n1 = _at(d, cfg.NY_WINDOW[0], ET), _at(d, cfg.NY_WINDOW[1], ET)
            o0, o1 = max(l0, n0), min(l1, n1)
            for name, a, b in (("London", l0, o0), ("London–NY overlap", o0, o1),
                               ("New York", o1, n1)):
                a, b = _utc(a), _utc(b)
                if b > a and b > start_utc and a < end_utc:
                    out.append({"name": name, "start": a, "end": b})
        d += timedelta(days=1)
    return out


# ── Levels ───────────────────────────────────────────────────────────────────
def asian_window(td: date) -> Dict:
    """Asian range window for trading day td: 18:00 ET prior day -> London open."""
    start = _at(td - timedelta(days=1), time(18, 0), ET)
    end = _at(td, cfg.LONDON_WINDOW[0], LDN)
    return {"start": _utc(start), "end": _utc(end)}


def asian_range(df: pd.DataFrame, td: date) -> Optional[Dict]:
    """High/low of bars inside the Asian window of trading day td."""
    if df is None or df.empty:
        return None
    w = asian_window(td)
    sub = df[(df.index >= w["start"]) & (df.index < w["end"])]
    if sub.empty:
        return None
    return {"high": float(sub["High"].max()), "low": float(sub["Low"].min()),
            "bars": int(len(sub)), **w}


def prev_day_levels(df: pd.DataFrame, td: date) -> Optional[Dict]:
    """High/low/close of the last complete trading day before td."""
    if df is None or df.empty:
        return None
    days = trading_day_index(df.index)
    prior = sorted({x for x in days if x < td})
    if not prior:
        return None
    pd_day = prior[-1]
    sub = df[days == pd_day]
    return {"day": pd_day, "high": float(sub["High"].max()),
            "low": float(sub["Low"].min()), "close": float(sub["Close"].iloc[-1])}


def round_levels(price: float, step: float = cfg.ROUND_NUMBER_STEP,
                 count: int = cfg.ROUND_NUMBER_COUNT) -> List[float]:
    """Round-number levels around price (count above and below)."""
    base = (price // step) * step
    below = [base - step * i for i in range(count)]
    above = [base + step * (i + 1) for i in range(count)]
    return sorted(below + above)
