"""
xau_calendar.py — event calendar and trading gates (Phase 6).

Gate states for new entries:
  BLOCKED  market closed · CME rollover window · first 30 min after the weekly
           reopen · inside an event window (FOMC −60/+45 min, high −30/+15,
           medium −15/+10) · today's range has used the options expected move
  CAUTION  high-impact event within 2h · US holiday / early-close day · late
           Friday (weekend gap risk)
  OPEN     none of the above

Events come from the verified list in xau_config, rule-based recurring
releases (weekly claims, ISM), and an optional xau_events.csv
(date,time_et,name,impact) that adds or overrides entries without code.
All event times are US Eastern and converted per date, so DST is exact.
"""
import csv
import os
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import xau_config as cfg
import xau_sessions as xs

ET = ZoneInfo(cfg.ET)
UTC = ZoneInfo("UTC")
DISP = ZoneInfo(cfg.DISPLAY_TZ)
IMPACT_ORDER = {"fomc": 3, "high": 2, "medium": 1, "low": 0}


def _t(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))


def _event(d: str, t: str, name: str, impact: str, source: str) -> Dict:
    local = datetime.combine(date.fromisoformat(d), _t(t), tzinfo=ET)
    impact = impact.strip().lower()
    if impact not in cfg.GATE_WINDOWS:
        impact = "medium"
    return {"when": local.astimezone(UTC), "name": name.strip(), "impact": impact,
            "source": source, "date_et": d, "time_et": t}


# ── Event sources ────────────────────────────────────────────────────────────
def _business_days(year: int, month: int) -> List[date]:
    d, out = date(year, month, 1), []
    while d.month == month:
        if d.weekday() < 5 and d.isoformat() not in cfg.US_HOLIDAYS:
            out.append(d)
        d += timedelta(days=1)
    return out


def recurring_events(start: date, end: date) -> List[Dict]:
    out = []
    d = start
    while d <= end:                               # weekly jobless claims
        if d.weekday() == 3:
            ds = d.isoformat()
            ds = cfg.CLAIMS_MOVED.get(ds, ds)
            if ds not in cfg.US_HOLIDAYS:
                out.append(_event(ds, cfg.CLAIMS_TIME, "Jobless claims", "medium", "rule"))
        d += timedelta(days=1)
    y, m = start.year, start.month
    while date(y, m, 1) <= end:                   # ISM manufacturing / services
        bd = _business_days(y, m)
        if len(bd) >= 3:
            for day, tm, name in ((bd[0], cfg.ISM_MFG_TIME, "ISM manufacturing"),
                                  (bd[2], cfg.ISM_SVC_TIME, "ISM services")):
                if start <= day <= end:
                    out.append(_event(day.isoformat(), tm, name, "medium", "rule"))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def load_csv_events(path: Optional[str] = None) -> List[Dict]:
    """Optional user file: date (YYYY-MM-DD), time_et (HH:MM), name, impact."""
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg.EVENTS_CSV)
    if not os.path.exists(path):
        return []
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                out.append(_event(row["date"].strip(), row["time_et"].strip(),
                                  row["name"], row.get("impact", "high"), "csv"))
            except (KeyError, ValueError, AttributeError):
                continue
    return out


def all_events(start: date, end: date, csv_path: Optional[str] = None) -> List[Dict]:
    """Built-in + recurring + CSV, CSV overriding same date + name. Sorted."""
    base = [_event(d, t, n, i, "official") for d, t, n, i in cfg.CALENDAR_EVENTS]
    merged: Dict[tuple, Dict] = {}
    for ev in base + recurring_events(start, end) + load_csv_events(csv_path):
        merged[(ev["date_et"], ev["name"].lower())] = ev
    lo = datetime.combine(start, time(0), tzinfo=ET).astimezone(UTC)
    hi = datetime.combine(end + timedelta(days=1), time(0), tzinfo=ET).astimezone(UTC)
    return sorted((e for e in merged.values() if lo <= e["when"] < hi),
                  key=lambda e: (e["when"], -IMPACT_ORDER[e["impact"]]))


# ── Gate ─────────────────────────────────────────────────────────────────────
@dataclass
class GateReport:
    ok: bool = False
    error: str = ""
    state: str = "OPEN"                       # OPEN | CAUTION | BLOCKED
    reasons: List[str] = field(default_factory=list)
    cautions: List[str] = field(default_factory=list)
    active_event: Optional[Dict] = None
    resume_at: Optional[datetime] = None      # when the current block lifts (UTC)
    next_high: Optional[Dict] = None
    mins_to_next_high: Optional[float] = None
    upcoming: List[Dict] = field(default_factory=list)
    today: List[Dict] = field(default_factory=list)
    stale_calendar: bool = False
    notes: List[str] = field(default_factory=list)


def _block(rep: GateReport, reason: str, until: Optional[datetime] = None):
    rep.reasons.append(reason)
    if until is not None and (rep.resume_at is None or until > rep.resume_at):
        rep.resume_at = until


def compute(now: datetime, em_exhausted: bool = False, csv_path: Optional[str] = None,
            horizon_days: int = 45) -> GateReport:
    rep = GateReport()
    now = now.astimezone(UTC)
    et = now.astimezone(ET)
    events = all_events(et.date() - timedelta(days=1), et.date() + timedelta(days=horizon_days),
                        csv_path)

    # market / session gates
    st_ = xs.market_status(now)
    if not st_["open"]:
        _block(rep, f"Market {st_['state'].lower()}")
    wd, t = et.weekday(), et.time()
    r0, r1 = _t(cfg.ROLLOVER_BLOCK_ET[0]), _t(cfg.ROLLOVER_BLOCK_ET[1])
    if wd in (0, 1, 2, 3) and r0 <= t < r1:
        until = datetime.combine(et.date(), r1, tzinfo=ET).astimezone(UTC)
        _block(rep, "CME rollover window — spreads wide", until)
    if wd == 6 and time(18, 0) <= t < (datetime.combine(et.date(), time(18, 0)) +
                                       timedelta(minutes=cfg.WEEKLY_OPEN_BLOCK_MIN)).time():
        until = datetime.combine(et.date(), time(18, 0), tzinfo=ET).astimezone(UTC) + \
            timedelta(minutes=cfg.WEEKLY_OPEN_BLOCK_MIN)
        _block(rep, "Weekly open — gap and thin liquidity", until)
    if wd == 4 and t >= _t(cfg.FRIDAY_CAUTION_ET):
        rep.cautions.append("Late Friday — weekend gap risk; avoid new swing entries")

    # event windows
    for ev in events:
        pre, post = cfg.GATE_WINDOWS[ev["impact"]]
        a = ev["when"] - timedelta(minutes=pre)
        b = ev["when"] + timedelta(minutes=post)
        if a <= now < b:
            mins = (ev["when"] - now).total_seconds() / 60
            when = f"in {mins:.0f} min" if mins > 0 else f"{-mins:.0f} min ago"
            _block(rep, f"{ev['name']} ({when})", b)
            if rep.active_event is None or IMPACT_ORDER[ev["impact"]] > \
                    IMPACT_ORDER[rep.active_event["impact"]]:
                rep.active_event = ev

    # expected move exhausted (from L7)
    if em_exhausted and cfg.BLOCK_ON_EM_EXHAUSTED:
        _block(rep, "Today's range has used the options-implied move")

    # cautions
    highs = [e for e in events if e["impact"] in ("high", "fomc") and e["when"] > now]
    if highs:
        rep.next_high = highs[0]
        rep.mins_to_next_high = (highs[0]["when"] - now).total_seconds() / 60
        if rep.mins_to_next_high <= cfg.CAUTION_BEFORE_HIGH_MIN and not rep.reasons:
            rep.cautions.append(f"{highs[0]['name']} in {rep.mins_to_next_high:.0f} min — "
                                "reduce size, no new swing entries")
    hol = cfg.US_HOLIDAYS.get(et.date().isoformat())
    if hol:
        rep.cautions.append(f"US holiday: {hol} — thin liquidity")

    rep.upcoming = [e for e in events if e["when"] >= now][:12]
    today_eat = now.astimezone(DISP).date()
    rep.today = [e for e in events if e["when"].astimezone(DISP).date() == today_eat]
    if et.date() > date.fromisoformat(cfg.CALENDAR_VALID_UNTIL):
        rep.stale_calendar = True
        rep.notes.append(f"Built-in calendar ends {cfg.CALENDAR_VALID_UNTIL} — add new "
                         f"dates to {cfg.EVENTS_CSV} or xau_config.CALENDAR_EVENTS")

    rep.state = "BLOCKED" if rep.reasons else "CAUTION" if rep.cautions else "OPEN"
    rep.ok = True
    return rep


def get_gate_report(now: datetime, em_exhausted: bool = False,
                    csv_path: Optional[str] = None) -> GateReport:
    try:
        return compute(now, em_exhausted, csv_path)
    except Exception as e:  # noqa: BLE001
        return GateReport(error=f"{type(e).__name__}: {e}", state="CAUTION",
                          cautions=["Calendar check failed — trade manually around news"])
