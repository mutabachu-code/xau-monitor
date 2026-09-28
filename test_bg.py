"""Tests for xau_bg (background refresh) and incremental bundle merging."""
import pathlib
import sys
import threading
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import xau_bg as xb  # noqa: E402
import xau_data as xd  # noqa: E402
from synth import bars_from_closes  # noqa: E402


def test_first_call_waits_then_serves_cached_instantly():
    xb.reset()
    calls = []

    def slow():
        calls.append(1)
        time.sleep(0.2)
        return len(calls)

    assert xb.swr("k", slow, ttl=60, wait=2) == 1
    t = time.perf_counter()
    assert xb.swr("k", slow, ttl=60, wait=2) == 1          # fresh → no new call
    assert time.perf_counter() - t < 0.05 and len(calls) == 1


def test_stale_value_returned_while_refreshing_in_background():
    xb.reset()
    gate = threading.Event()
    n = {"v": 0}

    def job():
        n["v"] += 1
        if n["v"] > 1:
            gate.wait(2)
        return n["v"]

    assert xb.swr("s", job, ttl=0.01, wait=2) == 1
    time.sleep(0.02)
    t = time.perf_counter()
    assert xb.swr("s", job, ttl=0.01, wait=2) == 1          # stale value, no waiting
    assert time.perf_counter() - t < 0.05 and xb.status()["s"]["busy"]
    gate.set()
    xb.wait_idle(2)
    assert xb.swr("s", job, ttl=60) == 2


def test_failure_keeps_old_value_and_backs_off():
    xb.reset()
    state = {"fail": False, "calls": 0}

    def job():
        state["calls"] += 1
        if state["fail"]:
            raise ConnectionError("down")
        return "good"

    assert xb.swr("f", job, ttl=0.01, neg_ttl=60, wait=2) == "good"
    state["fail"] = True
    time.sleep(0.02)
    xb.swr("f", job, ttl=0.01, neg_ttl=60)
    xb.wait_idle(2)
    assert xb.swr("f", job, ttl=0.01, neg_ttl=60) == "good"
    assert "ConnectionError" in xb.status()["f"]["error"]
    calls = state["calls"]
    for _ in range(5):
        xb.swr("f", job, ttl=0.01, neg_ttl=60)
    assert state["calls"] == calls                          # no retries during back-off


def test_first_failure_returns_none_quickly_after_backoff():
    xb.reset()

    def boom():
        raise TimeoutError

    assert xb.swr("b", boom, ttl=60, neg_ttl=60, wait=2) is None
    t = time.perf_counter()
    assert xb.swr("b", boom, ttl=60, neg_ttl=60, wait=2) is None
    assert time.perf_counter() - t < 0.05


def _bundle(start, n, price):
    g = bars_from_closes(np.full(n, price), start=start)
    return {"gold": g, "primary": "GC=F", "cross": {"SI=F": g * 0.015},
            "errors": [], "fetched_at": pd.Timestamp.now(tz="UTC").to_pydatetime(),
            "full_at": 1.0}


def test_merge_bundle_overlays_recent_bars_and_trims():
    old = _bundle("2026-07-01 00:00", 96 * 80, 4000.0)        # 80 days
    new = _bundle(str(old["gold"].index[-10]), 20, 4300.0)     # overlaps last 10 bars
    m = xd.merge_bundle(old, new, keep_days=62)
    g = m["gold"]
    assert g.index.is_monotonic_increasing and not g.index.duplicated().any()
    assert g["Close"].iat[-1] == 4300.0 and g.loc[old["gold"].index[-1], "Close"] == 4300.0
    assert (g.index[-1] - g.index[0]).days <= 62
    assert m["cross"]["SI=F"].index[-1] == g.index[-1]
    assert m["mode"] == "incremental" and m["full_at"] == 1.0


def test_bundle_job_full_then_incremental(monkeypatch):
    xb.reset()
    periods = []

    def fake_fetch(interval, period):
        periods.append(period)
        start = "2026-09-20 00:00" if period != "5d" else "2026-09-25 00:00"
        return _bundle(start, 96 * (8 if period != "5d" else 3), 4300.0)

    monkeypatch.setattr(xd, "fetch_bundle", fake_fetch)
    b1 = xd.get_bundle(wait=5)
    assert b1["mode"] == "full" and periods == ["60d"]
    xb._STORE["bundle"]["t"] = 0                                # force stale
    xd.get_bundle(wait=0)
    xb.wait_idle(5)
    b2 = xd.get_bundle(wait=0)
    assert periods[-1] == "5d" and b2["mode"] == "incremental"
    xb.reset()
