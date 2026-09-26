"""Synthetic tests for xau_technicals (Layer 5)."""
import pathlib
import sys
from datetime import date

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import xau_config as cfg  # noqa: E402
import xau_sessions as xs  # noqa: E402
import xau_technicals as xt  # noqa: E402
from synth import bars_from_closes, trading_days, trend, zigzag  # noqa: E402


# ── Primitives ───────────────────────────────────────────────────────────────
def test_rsi_extremes_and_flat():
    up = pd.Series(np.arange(50, dtype=float))
    assert xt.rsi(up).iat[-1] == 100
    assert xt.rsi(-up).iat[-1] < 1
    assert xt.rsi(pd.Series(np.full(30, 5.0))).iat[-1] == 50
    alt = pd.Series(np.tile([1.0, 2.0], 50))
    assert 40 < xt.rsi(alt).iat[-1] < 60


def test_atr_constant_range():
    df = bars_from_closes(np.full(60, 4300.0), wick=2.0)
    assert xt.atr(df).iat[-1] == pytest.approx(4.0)


def test_adx_trend_vs_chop():
    tr_ = bars_from_closes(trend(200, step=2.0))
    ch = bars_from_closes(np.tile([4300.0, 4302.0, 4300.0, 4298.0], 50))
    assert xt.adx(tr_)["adx"].iat[-1] > 40
    assert xt.adx(ch)["adx"].iat[-1] < cfg.ADX_CHOP


def test_vwap_equal_volume_is_mean_and_resets():
    df = trading_days(2, bars_per_day=8,
                      price_fn=lambda k, n: np.full(n, 4300.0 + 100 * k))
    vw = xt.session_vwap(df)
    days = xs.trading_day_index(df.index)
    d1 = df[np.asarray(days) == sorted(set(days))[1]].index
    assert vw.loc[d1[0], "vwap"] == pytest.approx(4400.0)  # reset, no day-1 leakage
    assert vw.loc[d1[-1], "sd"] == pytest.approx(0.0, abs=1e-6)


def test_vwap_volume_weighting_and_zero_volume_fallback():
    df = bars_from_closes([4300.0, 4310.0], start="2026-09-28 08:00", wick=0.0,
                          volume=[100.0, 300.0])
    tp = (df.High + df.Low + df.Close) / 3
    exp = (tp.iat[0] * 100 + tp.iat[1] * 300) / 400
    assert xt.session_vwap(df)["vwap"].iat[-1] == pytest.approx(exp)
    z = bars_from_closes([4300.0, 4310.0], start="2026-09-28 08:00", wick=0.0, volume=0.0)
    assert xt.session_vwap(z)["vwap"].iat[-1] == pytest.approx(tp.mean())


# ── CPR ──────────────────────────────────────────────────────────────────────
def test_cpr_formula():
    c = xt.cpr_from(4320.0, 4280.0, 4310.0)
    assert c["P"] == pytest.approx(4303.3333, rel=1e-6)
    assert c["BC"] == pytest.approx(4300.0)
    assert c["TC"] == pytest.approx(4306.6667, rel=1e-6)
    assert c["TC"] >= c["BC"]
    assert c["R1"] == pytest.approx(2 * c["P"] - 4280)
    assert c["S1"] == pytest.approx(2 * c["P"] - 4320)
    # close below midpoint → TC/BC swap stays ordered
    c2 = xt.cpr_from(4320.0, 4280.0, 4285.0)
    assert c2["TC"] > c2["BC"]


def test_cpr_virgin_detection():
    # day0 range 4290-4310 close 4300 → day1 CPR ≈ 4300; day1 trades 4340+ → virgin
    df = trading_days(2, bars_per_day=8, price_fn=lambda k, n:
                      np.linspace(4292, 4308, n) if k == 0 else np.full(n, 4350.0))
    t = xt.cpr_table(xt.daily_ohlc(df))
    assert bool(t.iloc[-1]["virgin"]) is True
    df2 = trading_days(2, bars_per_day=8, price_fn=lambda k, n: np.linspace(4292, 4308, n))
    assert bool(xt.cpr_table(xt.daily_ohlc(df2)).iloc[-1]["virgin"]) is False


def test_cpr_width_class_uses_history():
    rng = np.random.default_rng(3)
    df = trading_days(12, bars_per_day=20,
                      price_fn=lambda k, n: 4300 + rng.normal(0, 8, n).cumsum())
    t = xt.cpr_table(xt.daily_ohlc(df))
    assert set(t["width_class"]) <= {"narrow", "moderate", "wide"}
    assert len(t) == 11


# ── Structure ────────────────────────────────────────────────────────────────
def test_structure_uptrend_bos():
    df = bars_from_closes(zigzag(120, drift=1.0, amp=5.0))
    st = xt.market_structure(df)
    assert st["trend"] == "up"
    assert any(e["kind"] == "BoS" and e["dir"] == "up" for e in st["events"])


def test_structure_mss_on_reversal():
    up = zigzag(80, drift=1.0, amp=5.0)
    down = up[-1] - 1.5 * np.arange(1, 41)
    st = xt.market_structure(bars_from_closes(np.r_[up, down]))
    assert st["trend"] == "down"
    downs = [e for e in st["events"] if e["dir"] == "down"]
    assert downs[0]["kind"] == "MSS"


def test_swings_confirmed_only():
    df = bars_from_closes(zigzag(40))
    sw = xt.swing_points(df, 2)
    assert all(i <= len(df) - 3 for i in sw["highs"] + sw["lows"])


def test_bearish_divergence():
    # two rising price peaks, the second made on weaker momentum
    a = np.r_[np.linspace(4300, 4330, 10), np.linspace(4330, 4310, 8),
              np.linspace(4310, 4333, 16), np.linspace(4333, 4315, 8)]
    df = bars_from_closes(a, wick=0.2)
    r = xt.rsi(df["Close"], 5)
    assert xt.rsi_divergence(df, r, xt.swing_points(df), lookback=60) == "bearish"


def test_momentum_decay():
    c = pd.Series([1, 2, 3, 4, 5, 6.0])
    r = pd.Series([70, 68, 66, 65, 64, 62.0])
    assert xt.momentum_decay(c, r, bars=5, drop=3) == "bullish fading"
    assert xt.momentum_decay(-c, 100 - r, bars=5, drop=3) == "bearish fading"
    assert xt.momentum_decay(c, r + 2 * np.arange(6), bars=5, drop=3) is None


# ── HTF and report ───────────────────────────────────────────────────────────
def test_htf_trend_up_and_down():
    up = bars_from_closes(trend(400, step=0.5))
    dn = bars_from_closes(trend(400, step=-0.5))
    assert xt.htf_trend(up)["trend"] == "up"
    assert xt.htf_trend(dn)["trend"] == "down"
    assert xt.htf_trend(up.iloc[:40])["trend"] == "unknown"


def test_report_long_in_uptrend():
    df = trading_days(4, price_fn=lambda k, n: zigzag(n, 4300 + 70 * k, drift=0.8, amp=3))
    rep = xt.get_tech_report(df)
    assert rep.ok, rep.error
    assert rep.score > cfg.TECH_BIAS_THRESHOLD and rep.bias == "LONG"
    assert -cfg.TECH_MAX <= rep.score <= cfg.TECH_MAX
    assert set(rep.components) == {"VWAP", "EMA stack", "1h trend", "Structure", "RSI", "CPR"}
    assert rep.frame is not None and len(rep.frame) == len(df)


def test_report_short_in_downtrend():
    df = trading_days(4, price_fn=lambda k, n: zigzag(n, 4600 - 70 * k, drift=-0.8, amp=3))
    rep = xt.get_tech_report(df)
    assert rep.score < -cfg.TECH_BIAS_THRESHOLD and rep.bias == "SHORT"


def test_report_chop_damped():
    df = trading_days(4, price_fn=lambda k, n: np.tile([4300.0, 4303, 4300, 4297], n // 4))
    rep = xt.get_tech_report(df)
    assert rep.chop and any("chop" in n for n in rep.notes)
    assert abs(rep.score) < cfg.TECH_BIAS_THRESHOLD


def test_report_never_raises():
    assert not xt.get_tech_report(None).ok
    assert not xt.get_tech_report(bars_from_closes(trend(20))).ok
    bad = bars_from_closes(trend(100)).drop(columns=["High"])
    r = xt.get_tech_report(bad)
    assert not r.ok and r.error
