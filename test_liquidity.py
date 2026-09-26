"""Synthetic tests for xau_liquidity (Layer 6).

Fixture: Fri 25 Sep (trading day starts Thu 24 Sep 22:00 UTC) ranges
4290–4310, then Mon 28 Sep (starts Sun 27 Sep 22:00 UTC). Monday London
open = 07:00 UTC, overlap 12:00–16:00 UTC.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import xau_config as cfg  # noqa: E402
import xau_liquidity as xl  # noqa: E402
from synth import bars_from_closes  # noqa: E402


def friday():
    n = 92
    c = 4300 + 9 * np.sin(np.linspace(0, 6 * np.pi, n))      # ~4291–4309
    return bars_from_closes(c, start="2026-09-24 22:00", wick=0.5)


def monday(closes, start="2026-09-27 22:00", volume=100.0, lows=None, highs=None):
    df = bars_from_closes(closes, start=start, wick=0.5, volume=volume)
    for k, v in (lows or {}).items():
        df.iloc[k, df.columns.get_loc("Low")] = v
    for k, v in (highs or {}).items():
        df.iloc[k, df.columns.get_loc("High")] = v
    return df


def build(mon):
    return pd.concat([friday(), mon])


# Monday bar k starts at 22:00 UTC + 15m*k → London (07:00 UTC) = k 36, 12:00 UTC = k 56
LONDON_K = 36


def test_rvol_time_of_day_spike():
    days = [bars_from_closes(np.full(96, 4300.0), start=f"2026-09-{d} 00:00")
            for d in range(1, 11)]
    df = pd.concat(days)
    df.iloc[-1, df.columns.get_loc("Volume")] = 400.0
    r = xl.time_of_day_rvol(df)
    assert r.iat[-1] == 4.0 and r.iat[-2] == 1.0


def test_bullish_pdl_sweep_in_london_scores_long():
    n = LONDON_K + 6
    closes = np.r_[np.full(LONDON_K, 4300.0), [4295, 4292.5, 4296, 4299, 4301, 4302]]
    vol = np.full(n, 100.0)
    vol[LONDON_K + 1] = 400.0
    mon = monday(closes, volume=vol, lows={LONDON_K + 1: 4286.0})   # PDL ≈ 4290.5
    df = build(mon)
    rep = xl.get_liq_report(df)
    assert rep.ok, rep.error
    pdl = [s for s in rep.sweeps if s["level"] == "PDL"]
    assert pdl and pdl[0]["dir"] == "bullish" and not pdl[0]["invalid"]
    assert pdl[0]["session"] == "London"
    assert rep.score > cfg.LIQ_BIAS_THRESHOLD and rep.bias == "LONG"
    assert "RVOL" in rep.active["why"]


def test_sweep_invalidated_when_extreme_breaks():
    closes = np.r_[np.full(LONDON_K, 4300.0), [4295, 4292.5, 4296, 4285, 4284, 4283]]
    mon = monday(closes, lows={LONDON_K + 1: 4286.0})
    rep = xl.get_liq_report(build(mon))
    pdl = [s for s in rep.sweeps if s["level"] == "PDL"]
    assert pdl and pdl[0]["invalid"] and pdl[0]["points"] == 0


def test_taken_level_is_not_a_sweep():
    # closes below PDL for longer than the reclaim window (accepted breakdown),
    # later recovers — that is a breakdown, not a bullish sweep
    closes = np.r_[np.full(LONDON_K, 4300.0), [4288, 4287, 4286, 4286, 4295, 4296, 4297]]
    mon = monday(closes, lows={LONDON_K + 3: 4284.0})
    rep = xl.get_liq_report(build(mon))
    assert not [s for s in rep.sweeps if s["level"] == "PDL"]


def test_bearish_asia_high_sweep():
    k = 56                                              # 12:00 UTC overlap
    closes = np.r_[np.full(k, 4300.0), [4303, 4299, 4297, 4295]]
    mon = monday(closes, highs={k: 4304.5})             # Asian high = 4300.5
    rep = xl.get_liq_report(build(mon))
    hits = [s for s in rep.sweeps if s["level"] == "Asia H"]
    assert hits and hits[0]["dir"] == "bearish"
    assert rep.score < 0
    assert "swept Asia H" in rep.asian_state


def test_asia_levels_absent_before_london():
    mon = monday(np.full(20, 4300.0))                   # still in Asia session
    df = build(mon)
    rep = xl.get_liq_report(df)
    assert not [l for l in rep.levels if l.name.startswith("Asia")]
    assert rep.asian_state == "forming"


def test_equal_highs_cluster():
    # three peaks at ~4310 inside tolerance
    base = np.r_[[4300, 4305, 4310, 4305, 4300] * 3, [4298] * 5]
    df = bars_from_closes(np.r_[np.full(80, 4300.0), base], wick=0.3)
    eq = xl.equal_levels(df, atr_val=5.0)
    eqh = [l for l in eq if l.name == "EQH"]
    assert len(eqh) == 1 and abs(eqh[0].price - 4310.3) < 1e-6


def test_old_sweep_decays_out():
    closes = np.r_[np.full(LONDON_K, 4300.0), [4295, 4292.5, 4296], np.full(12, 4297.0)]
    mon = monday(closes, lows={LONDON_K + 1: 4286.0})
    rep = xl.get_liq_report(build(mon))
    pdl = [s for s in rep.sweeps if s["level"] == "PDL"]
    assert pdl and pdl[0]["points"] == 0 and pdl[0]["why"] == "too old"


def test_targets_and_never_raises():
    mon = monday(np.full(LONDON_K + 4, 4300.0))
    rep = xl.get_liq_report(build(mon))
    assert rep.targets["above"] and rep.targets["above"]["price"] > 4300
    assert rep.targets["below"] and rep.targets["below"]["price"] < 4300
    assert not xl.get_liq_report(None).ok
    assert not xl.get_liq_report(build(mon).drop(columns=["Low"])).ok
