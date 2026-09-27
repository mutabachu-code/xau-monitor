"""Tests for xau_spot — parsers, fallback, staleness, live basis, median store."""
import pathlib
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import xau_config as cfg  # noqa: E402
import xau_spot as xsp  # noqa: E402
from synth import bars_from_closes  # noqa: E402

NOW = datetime(2026, 9, 28, 23, 37, tzinfo=timezone.utc)       # Mon 19:37 ET, market open
TS_MS = NOW.timestamp() * 1000

SWISS = [  # shape returned by the live endpoint (26 Sep 2026 check)
    {"topo": {"platform": "SwissquoteLtd", "server": "Live5"}, "ts": TS_MS - 3000,
     "spreadProfilePrices": [
         {"spreadProfile": "premium", "bid": 4260.814, "ask": 4261.476},
         {"spreadProfile": "prime", "bid": 4260.828, "ask": 4261.462},
         {"spreadProfile": "elite", "bid": 4260.893, "ask": 4261.397}]},
    {"topo": {"platform": "AT", "server": "AT1"}, "ts": TS_MS - 60000,
     "spreadProfilePrices": [{"spreadProfile": "prime", "bid": 4259.0, "ask": 4260.0}]},
]
GOLDAPI = {"currency": "USD", "name": "Gold", "price": 4286.200195, "symbol": "XAU",
           "updatedAt": "2026-09-27T11:12:32Z"}


def test_parse_swissquote_prefers_prime_and_freshest():
    q = xsp.parse_swissquote(SWISS)
    assert q["bid"] == 4260.828 and q["ask"] == 4261.462
    assert q["mid"] == pytest.approx(4261.145) and q["source"] == "Swissquote"
    assert q["ts"] == datetime.fromtimestamp((TS_MS - 3000) / 1000, tz=timezone.utc)


def test_parse_swissquote_bad_input():
    assert xsp.parse_swissquote([]) is None
    assert xsp.parse_swissquote([{"spreadProfilePrices": [{"bid": "x", "ask": 1}]}]) is None
    assert xsp.parse_swissquote([{"ts": 1, "spreadProfilePrices":
                                  [{"spreadProfile": "prime", "bid": 5, "ask": 4}]}]) is None


def test_parse_goldapi():
    q = xsp.parse_goldapi(GOLDAPI)
    assert q["mid"] == pytest.approx(4286.2) and q["bid"] is None
    assert q["ts"] == datetime(2026, 9, 27, 11, 12, 32, tzinfo=timezone.utc)
    assert xsp.parse_goldapi({"price": "x"}) is None


def test_fetch_fallback_and_staleness(monkeypatch):
    raw = getattr(xsp._spot_raw, "__wrapped__", xsp._spot_raw)
    monkeypatch.setattr(xsp, "_spot_raw", raw)

    def swiss_down(url):
        if "swissquote" in url:
            raise TimeoutError("down")
        return GOLDAPI

    monkeypatch.setattr(xsp, "_get_json", swiss_down)
    q = xsp.fetch_spot(NOW)
    assert q["source"] == "gold-api" and q["stale"]          # Friday-close quote
    assert "Swissquote: TimeoutError" in q["errors"][0]

    monkeypatch.setattr(xsp, "_get_json", lambda url: SWISS)
    q = xsp.fetch_spot(NOW)
    assert q["source"] == "Swissquote" and not q["stale"] and q["age_sec"] == pytest.approx(3)

    def all_down(url):
        raise TimeoutError("down")

    monkeypatch.setattr(xsp, "_get_json", all_down)
    assert xsp.fetch_spot(NOW) is None


def gold_at(last=4297.6, minutes_old=5):
    end = NOW - timedelta(minutes=minutes_old)
    df = bars_from_closes(np.full(10, last), start=end - timedelta(minutes=15 * 9))
    return df


def spot(mid=4261.15, age=3, stale=False):
    return {"mid": mid, "bid": mid - 0.3, "ask": mid + 0.3, "spread": 0.6,
            "age_sec": age, "stale": stale, "source": "Swissquote"}


def test_live_basis_usable():
    r = xsp.live_basis(gold_at(), spot(), NOW)
    assert r["usable"] and r["basis"] == pytest.approx(36.45)


def test_live_basis_rejections():
    assert xsp.live_basis(gold_at(minutes_old=45), spot(), NOW)["reason"].startswith("GC=F bar")
    assert "old" in xsp.live_basis(gold_at(), spot(stale=True, age=300), NOW)["reason"]
    assert "sanity" in xsp.live_basis(gold_at(last=4600), spot(), NOW)["reason"]
    sat = datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc)
    assert xsp.live_basis(gold_at(), spot(), sat)["reason"] == "market closed"
    assert xsp.live_basis(gold_at(), None, NOW)["reason"] == "no spot quote"
    assert not xsp.live_basis(None, spot(), NOW)["usable"]


def test_record_basis_median_and_cap():
    store = {"hist": []}
    for b in [36.0, 36.5, 50.0, 36.4]:                      # one noisy reading
        m = xsp.record_basis(store, {"usable": True, "basis": b}, NOW)
    assert m == pytest.approx(36.45)
    assert xsp.record_basis(store, {"usable": False, "basis": 99}, NOW) == pytest.approx(36.45)
    for _ in range(20):
        xsp.record_basis(store, {"usable": True, "basis": 30.0}, NOW)
    assert len(store["hist"]) == cfg.BASIS_HISTORY and xsp.record_basis(store, {}, NOW) == 30


def test_effective_basis_modes():
    assert xsp.effective_basis("Auto", 20.0, 36.4) == {"basis": 36.4,
                                                       "source": "live (median of recent readings)"}
    assert xsp.effective_basis("Auto", 20.0, None)["source"].startswith("manual fallback")
    assert xsp.effective_basis("Manual", 20.0, 36.4)["basis"] == 20.0
