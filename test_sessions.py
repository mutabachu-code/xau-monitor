"""Synthetic tests for xau_sessions — DST transitions, status, levels."""
import pathlib
import sys
from datetime import date, datetime, time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import xau_sessions as xs  # noqa: E402

UTC = xs.UTC


def _ev(d, key):
    hits = [e for e in xs.schedule_for_display_date(d) if e["key"] == key]
    assert len(hits) == 1, f"{key} on {d}: {hits}"
    return hits[0]


def _u(*a):
    return datetime(*a, tzinfo=UTC)


# ── Schedule across DST ──────────────────────────────────────────────────────
def test_summer_schedule_eat():            # Mon 28 Sep 2026: BST + EDT
    d = date(2026, 9, 28)
    assert _ev(d, "london_open")["start"].time() == time(10, 0)
    assert _ev(d, "lbma_am")["start"].time() == time(12, 30)
    assert _ev(d, "comex_open")["start"].time() == time(15, 20)
    assert _ev(d, "us_data")["start"].time() == time(15, 30)
    assert _ev(d, "ny_cash")["start"].time() == time(16, 30)
    assert _ev(d, "lbma_pm")["start"].time() == time(17, 0)
    assert _ev(d, "london_close")["start"].time() == time(19, 0)
    assert _ev(d, "sge_open")["start"].time() == time(4, 0)


def test_uk_dst_ended_us_not_yet():        # Tue 27 Oct: GMT + EDT
    d = date(2026, 10, 27)
    assert _ev(d, "london_open")["start"].time() == time(11, 0)
    assert _ev(d, "lbma_pm")["start"].time() == time(18, 0)
    assert _ev(d, "us_data")["start"].time() == time(15, 30)


def test_both_on_winter_time():            # Tue 3 Nov: GMT + EST
    d = date(2026, 11, 3)
    assert _ev(d, "london_open")["start"].time() == time(11, 0)
    assert _ev(d, "us_data")["start"].time() == time(16, 30)
    assert _ev(d, "sge_open")["start"].time() == time(4, 0)   # China has no DST


def test_cme_break_shifts_with_us_dst():
    brk = _ev(date(2026, 9, 29), "cme_break")
    assert brk["start"].time() == time(0, 0) and brk["end"].time() == time(1, 0)
    brk = _ev(date(2026, 11, 3), "cme_break")
    assert brk["start"].time() == time(1, 0) and brk["end"].time() == time(2, 0)


def test_weekend_has_no_london_events():
    keys = {e["key"] for e in xs.schedule_for_display_date(date(2026, 9, 26))}  # Saturday
    assert "london_open" not in keys and "us_data" not in keys
    assert "weekly_close" in keys          # Fri 17:00 EDT = Sat 00:00 EAT


def test_schedule_sorted():
    s = xs.schedule_for_display_date(date(2026, 9, 28))
    assert [e["start"] for e in s] == sorted(e["start"] for e in s)


def test_next_event():
    ev = xs.next_event(_u(2026, 9, 28, 6, 30))   # 09:30 EAT
    assert ev["key"] == "london_open"
    ev = xs.next_event(_u(2026, 9, 26, 8, 0))    # Saturday -> Sunday reopen
    assert ev["key"] == "asia_open"
    assert ev["start"].date() == date(2026, 9, 28) and ev["start"].time() == time(1, 0)


# ── Live status ──────────────────────────────────────────────────────────────
def test_market_status():
    assert not xs.market_status(_u(2026, 9, 26, 8, 0))["open"]           # Saturday
    assert not xs.market_status(_u(2026, 9, 27, 21, 0))["open"]          # Sun 17:00 EDT
    assert xs.market_status(_u(2026, 9, 27, 22, 30))["open"]             # Sun 18:30 EDT
    assert "break" in xs.market_status(_u(2026, 9, 28, 21, 30))["state"]  # Mon 17:30 EDT
    assert not xs.market_status(_u(2026, 10, 2, 21, 30))["open"]         # Fri 17:30 EDT


def test_session_labels():
    assert xs.session_label(_u(2026, 9, 28, 2, 0)) == "Asia"
    assert xs.session_label(_u(2026, 9, 28, 10, 0)) == "London"
    assert xs.session_label(_u(2026, 9, 28, 13, 0)) == "London–NY overlap"
    assert xs.session_label(_u(2026, 9, 28, 18, 0)) == "New York"
    assert xs.session_label(_u(2026, 9, 26, 12, 0)) == "Closed"


def test_session_windows_overlap_summer():
    ws = xs.session_windows(_u(2026, 9, 28, 0, 0), _u(2026, 9, 28, 23, 59))
    ov = [w for w in ws if w["name"] == "London–NY overlap"]
    assert len(ov) == 1
    assert ov[0]["start"] == _u(2026, 9, 28, 12, 0) and ov[0]["end"] == _u(2026, 9, 28, 16, 0)


def test_session_windows_overlap_winter_gap_week():
    # Oct 27: London on GMT (08:00-17:00 UTC), NY on EDT (12:00-21:00 UTC)
    ws = xs.session_windows(_u(2026, 10, 27, 0, 0), _u(2026, 10, 27, 23, 59))
    ov = [w for w in ws if w["name"] == "London–NY overlap"]
    assert ov[0]["start"] == _u(2026, 10, 27, 12, 0) and ov[0]["end"] == _u(2026, 10, 27, 17, 0)


# ── Trading day and levels ───────────────────────────────────────────────────
def test_trading_day_rolls_at_18_et():
    assert xs.trading_day(_u(2026, 9, 27, 22, 30)) == date(2026, 9, 28)  # Sun 18:30 EDT
    assert xs.trading_day(_u(2026, 9, 28, 20, 0)) == date(2026, 9, 28)   # Mon 16:00 EDT
    assert xs.trading_day(_u(2026, 9, 28, 22, 15)) == date(2026, 9, 29)  # Mon 18:15 EDT


def _bars(start, end, price=4300.0):
    idx = pd.date_range(start, end, freq="15min", tz="UTC", inclusive="left")
    p = np.full(len(idx), price)
    return pd.DataFrame({"Open": p, "High": p + 1, "Low": p - 1, "Close": p,
                         "Volume": 100.0}, index=idx)


def test_asian_range_excludes_london():
    df = _bars("2026-09-27 22:00", "2026-09-28 14:00")
    df.loc[pd.Timestamp("2026-09-27 23:00", tz="UTC"), "High"] = 4320   # Asia
    df.loc[pd.Timestamp("2026-09-28 09:00", tz="UTC"), "High"] = 4350   # London
    df.loc[pd.Timestamp("2026-09-28 03:00", tz="UTC"), "Low"] = 4280    # Asia
    r = xs.asian_range(df, date(2026, 9, 28))
    assert r["high"] == 4320 and r["low"] == 4280
    assert r["start"] == _u(2026, 9, 27, 22, 0) and r["end"] == _u(2026, 9, 28, 7, 0)


def test_prev_day_levels_uses_friday_for_monday():
    fri = _bars("2026-09-24 22:00", "2026-09-25 21:00", 4250.0)   # Fri trading day
    mon = _bars("2026-09-27 22:00", "2026-09-28 12:00", 4300.0)
    fri.iloc[5, fri.columns.get_loc("High")] = 4275
    df = pd.concat([fri, mon])
    lv = xs.prev_day_levels(df, date(2026, 9, 28))
    assert lv["day"] == date(2026, 9, 25)
    assert lv["high"] == 4275 and lv["low"] == 4249 and lv["close"] == 4250


def test_round_levels():
    assert xs.round_levels(4287.3, 50, 2) == [4200, 4250, 4300, 4350]
