"""Tests for xau_calendar — event times (DST), gate windows, rules, CSV."""
import pathlib
import sys
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import xau_calendar as xk  # noqa: E402

ET = ZoneInfo("America/New_York")
EAT = ZoneInfo("Africa/Nairobi")
NO_CSV = "/nonexistent/xau_events.csv"


def et(y, mo, d, h, mi=0):
    return datetime(y, mo, d, h, mi, tzinfo=ET)


def gate(now, **kw):
    return xk.get_gate_report(now, csv_path=kw.pop("csv_path", NO_CSV), **kw)


def find(events, name):
    return [e for e in events if e["name"] == name]


# ── Event times ──────────────────────────────────────────────────────────────
def test_cpi_eat_time_across_us_dst():
    ev = xk.all_events(date(2026, 10, 1), date(2026, 11, 30), NO_CSV)
    oct_cpi = find(ev, "CPI (Sep)")[0]["when"].astimezone(EAT)
    nov_cpi = find(ev, "CPI (Oct)")[0]["when"].astimezone(EAT)
    assert (oct_cpi.date(), oct_cpi.time()) == (date(2026, 10, 14), time(15, 30))
    assert (nov_cpi.date(), nov_cpi.time()) == (date(2026, 11, 10), time(16, 30))


def test_fomc_is_2100_eat_in_october():
    fomc = find(xk.all_events(date(2026, 10, 27), date(2026, 10, 29), NO_CSV),
                "FOMC decision")[0]
    assert fomc["when"].astimezone(EAT).time() == time(21, 0)
    assert fomc["impact"] == "fomc"


def test_claims_thursdays_and_holiday_moves():
    ev = xk.all_events(date(2026, 11, 1), date(2026, 12, 31), NO_CSV)
    days = {e["date_et"] for e in find(ev, "Jobless claims")}
    assert "2026-11-19" in days and "2026-11-25" in days and "2026-11-26" not in days
    assert "2026-12-23" in days and "2026-12-24" not in days
    # PCE and claims share Nov 25 08:30 and both survive
    assert find(ev, "PCE (Oct) + GDP Q3 2nd")


def test_ism_business_day_rule():
    ev = xk.all_events(date(2026, 10, 1), date(2026, 12, 31), NO_CSV)
    assert {e["date_et"] for e in find(ev, "ISM manufacturing")} == \
        {"2026-10-01", "2026-11-02", "2026-12-01"}
    assert {e["date_et"] for e in find(ev, "ISM services")} == \
        {"2026-10-05", "2026-11-04", "2026-12-03"}


def test_events_sorted():
    ev = xk.all_events(date(2026, 10, 1), date(2026, 12, 31), NO_CSV)
    assert [e["when"] for e in ev] == sorted(e["when"] for e in ev)


# ── Gate windows ─────────────────────────────────────────────────────────────
def test_cpi_window_edges():
    assert gate(et(2026, 10, 14, 7, 59)).state == "CAUTION"      # within 2h, not yet blocked
    r = gate(et(2026, 10, 14, 8, 0))
    assert r.state == "BLOCKED" and "CPI (Sep)" in r.reasons[0]
    assert r.resume_at == et(2026, 10, 14, 8, 45).astimezone(ZoneInfo("UTC"))
    assert gate(et(2026, 10, 14, 8, 44)).state == "BLOCKED"
    assert gate(et(2026, 10, 14, 8, 45)).state == "OPEN"


def test_fomc_window():
    assert gate(et(2026, 10, 28, 12, 59)).state == "CAUTION"
    assert gate(et(2026, 10, 28, 13, 0)).state == "BLOCKED"
    assert gate(et(2026, 10, 28, 14, 44)).state == "BLOCKED"
    assert gate(et(2026, 10, 28, 14, 45)).state != "BLOCKED"


def test_medium_event_window_and_overlap_picks_highest():
    r = gate(et(2026, 11, 25, 8, 25))              # PCE (high) + claims (medium) at 08:30
    assert r.state == "BLOCKED" and r.active_event["impact"] == "high"
    assert len(r.reasons) == 2


def test_rollover_weekly_open_weekend():
    assert "rollover" in gate(et(2026, 9, 29, 16, 50)).reasons[0]
    r = gate(et(2026, 9, 27, 18, 10))
    assert any("Weekly open" in x for x in r.reasons)
    assert gate(et(2026, 9, 27, 18, 31)).state != "BLOCKED" or \
        not any("Weekly open" in x for x in gate(et(2026, 9, 27, 18, 31)).reasons)
    assert gate(et(2026, 9, 26, 12, 0)).state == "BLOCKED"      # Saturday


def test_friday_and_holiday_cautions():
    assert any("Late Friday" in c for c in gate(et(2026, 10, 9, 15, 30)).cautions)
    assert any("Veterans" in c for c in gate(et(2026, 11, 11, 11, 0)).cautions)


def test_em_exhausted_blocks():
    r = gate(et(2026, 10, 20, 11, 0), em_exhausted=True)
    assert r.state == "BLOCKED" and "implied move" in r.reasons[0]


def test_open_state_and_next_high():
    r = gate(et(2026, 10, 20, 11, 0))
    assert r.state == "OPEN" and r.next_high["name"] == "FOMC decision"
    assert r.mins_to_next_high > 0 and r.upcoming


# ── CSV and robustness ───────────────────────────────────────────────────────
def test_csv_adds_and_overrides(tmp_path):
    p = tmp_path / "ev.csv"
    p.write_text("date,time_et,name,impact\n"
                 "2026-11-12,13:00,30y bond auction,medium\n"
                 "2026-10-14,08:30,CPI (Sep),fomc\n"
                 "not-a-date,xx,Broken,high\n")
    ev = xk.all_events(date(2026, 10, 1), date(2026, 11, 30), str(p))
    assert find(ev, "30y bond auction")[-1]["date_et"] == "2026-11-12"
    cpi = find(ev, "CPI (Sep)")
    assert len(cpi) == 1 and cpi[0]["impact"] == "fomc" and cpi[0]["source"] == "csv"
    assert gate(et(2026, 10, 14, 7, 35), csv_path=str(p)).state == "BLOCKED"  # 60-min window


def test_stale_calendar_note():
    r = gate(et(2027, 1, 5, 11, 0))
    assert r.stale_calendar and r.notes


def test_repo_csv_template_loads():
    assert xk.load_csv_events() == []


def test_never_raises():
    r = xk.get_gate_report("not a datetime")
    assert not r.ok and r.state == "CAUTION"
