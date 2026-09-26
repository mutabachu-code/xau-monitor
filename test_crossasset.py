"""Synthetic tests for xau_crossasset (Layer 3)."""
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import xau_config as cfg  # noqa: E402
import xau_crossasset as xc  # noqa: E402
import xau_sessions as xs  # noqa: E402
from test_macro import frame, session_index  # noqa: E402

B_Y, B_D, B_O = -0.00008, -0.6, 0.05     # gold per bp of yield, per DXY ret, per oil ret


def world(seed=1, today_extra=0.0, ybp_today=0.0, dxy_today=0.0, oil_today=0.0,
          silver=None, copper_today=0.0, noise=0.0002, days=22):
    """Independent driver shocks; gold = B·drivers + noise (+ unexplained
    `today_extra` per bar on the last session). *_today add a per-bar drift
    to that driver on the last session."""
    rng = np.random.default_rng(seed)
    idx = session_index(days)
    n = len(idx)
    last = np.zeros(n)
    last[-92:] = 1.0
    ybp = rng.normal(0, 0.6, n) + ybp_today * last
    dxy = rng.normal(0, 0.0003, n) + dxy_today * last
    oil = rng.normal(0, 0.002, n) + oil_today * last
    g = B_Y * ybp + B_D * dxy + B_O * oil + rng.normal(0, noise, n) + today_extra * last
    zn_ret = -ybp * cfg.FUT_DURATION["ZN=F"] / 1e4
    si_ret = 1.3 * g if silver is None else silver(g, last, rng)
    cu_ret = rng.normal(0, 0.0008, n) + copper_today * last
    return {"gold": frame(idx, g, 4300.0), "primary": "GC=F",
            "cross": {"ZN=F": frame(idx, zn_ret, 110.0),
                      "DX-Y.NYB": frame(idx, dxy, 101.0),
                      "CL=F": frame(idx, oil, 95.0),
                      "SI=F": frame(idx, si_ret, 64.0),
                      "HG=F": frame(idx, cu_ret, 4.6)},
            "errors": []}


def today_of(b):
    return xs.trading_day(b["gold"].index[-1].to_pydatetime())


# ── Model ────────────────────────────────────────────────────────────────────
def test_regression_recovers_betas():
    b = world(noise=0.00005)
    m = xc.fit_implied(xc.driver_frame(b), today_of(b))
    assert m["beta"]["yield_bp"] == pytest.approx(B_Y, rel=0.1)
    assert m["beta"]["dxy"] == pytest.approx(B_D, rel=0.1)
    assert m["beta"]["oil"] == pytest.approx(B_O, rel=0.2)
    assert m["r2"] > 0.8


def test_training_excludes_today():
    b = world(today_extra=0.001)                  # huge unexplained drift today
    m = xc.fit_implied(xc.driver_frame(b), today_of(b))
    days = xs.trading_day_index(xc.driver_frame(b).index)
    assert m["n"] <= int((np.asarray(days) != today_of(b)).sum())
    assert m["beta"]["yield_bp"] == pytest.approx(B_Y, rel=0.25)


def test_hidden_bid_scores_long():
    rep = xc.get_xasset_report(world(today_extra=0.0001))
    assert rep.ok, rep.error
    assert rep.implied["resid_z"] > 1.5
    assert rep.components["Residual"] > 5
    assert any("hidden bid" in n for n in rep.notes)


def test_explained_move_has_small_residual():
    rep = xc.get_xasset_report(world(ybp_today=-0.4))  # yields falling all day
    assert rep.implied["implied_pct"] > 0.1           # drivers imply a rally
    assert abs(rep.implied["resid_z"]) < 1.5
    assert rep.implied["path"].shape[1] == 2


def test_hidden_supply_scores_short():
    rep = xc.get_xasset_report(world(today_extra=-0.0001))
    assert rep.components["Residual"] < -5 and rep.score < 0


# ── Structural bid / offer ───────────────────────────────────────────────────
def test_structural_bid_flag():
    b = world(ybp_today=0.25, dxy_today=0.00004, today_extra=0.0002)
    rep = xc.get_xasset_report(b)
    assert rep.readings["yield_day_bp"] >= cfg.STRUCT_YIELD_BP
    assert rep.readings["dxy_day_pct"] >= cfg.STRUCT_DXY_PCT
    assert rep.flags["structural_bid"] and rep.components["Structural"] == 5
    assert rep.bias == "LONG"


def test_structural_offer_flag():
    b = world(ybp_today=-0.25, dxy_today=-0.00004, today_extra=-0.0002)
    rep = xc.get_xasset_report(b)
    assert rep.flags["structural_offer"] and rep.components["Structural"] == -5


def test_no_structural_when_normal():
    rep = xc.get_xasset_report(world(ybp_today=-0.1, today_extra=0.0))
    assert not rep.flags["structural_bid"] and rep.components["Structural"] == 0


# ── Silver ───────────────────────────────────────────────────────────────────
def test_silver_nonconfirmation_at_gold_high():
    def silver(g, last, rng):
        s = rng.normal(0, 0.0002, len(g))
        k = len(g) - 92
        s[k:k + 20] = 0.002                    # silver pops early in the session …
        s[k + 20:] = -0.0002                   # … then bleeds
        return s
    b = world(today_extra=0.0004, silver=silver)
    rep = xc.get_xasset_report(b)
    assert rep.flags["silver_nonconfirm_high"]
    assert rep.components["Silver"] < 0
    assert any("G3" in n for n in rep.notes)


def test_silver_confirms_rally():
    rep = xc.get_xasset_report(world(today_extra=0.0002))
    assert not rep.flags["silver_nonconfirm_high"]
    assert rep.components["Silver"] > 0


def test_silver_check_mid_range():
    b = world()
    chk = xc.silver_check(b["gold"].iloc[:-40], b["cross"]["SI=F"].iloc[:-40],
                          today_of(b), 5.0)
    assert chk["state"] in {"mid-range", "confirming high", "confirming low",
                            "not confirming high", "not confirming low"}


# ── Metals context and oil ───────────────────────────────────────────────────
def test_broad_metals_bid():
    def silver(g, last, rng):
        return 1.8 * g                         # silver outperforms → ratio falls
    rep = xc.get_xasset_report(world(today_extra=0.0002, silver=silver,
                                     copper_today=0.0002))
    assert rep.gs_ratio_chg < 0
    assert rep.move_type == "broad metals bid" and rep.components["Metals context"] == 3


def test_safe_haven_move():
    def silver(g, last, rng):
        return 0.4 * g                         # silver lags → ratio rises
    rep = xc.get_xasset_report(world(today_extra=0.0002, silver=silver,
                                     copper_today=-0.0002))
    assert rep.move_type == "safe-haven (gold-only)"


def test_oil_spike_flag():
    rep = xc.get_xasset_report(world(oil_today=0.0006))
    assert rep.readings["oil_day_pct"] >= cfg.OIL_SPIKE_PCT
    assert rep.flags["oil_spike_up"] and any("G8" in n for n in rep.notes)


# ── Fallbacks ────────────────────────────────────────────────────────────────
def test_eurusd_fallback_for_dollar():
    b = world()
    dxy = b["cross"].pop("DX-Y.NYB")
    eur = dxy.copy()
    for c in ("Open", "High", "Low", "Close"):
        eur[c] = 1 / dxy[c] * 111
    b["cross"]["EURUSD=X"] = eur
    fr = xc.driver_frame(b)
    assert "dxy" in fr.columns
    rep = xc.get_xasset_report(b)
    assert rep.ok and rep.readings["dxy_day_pct"] is not None


def test_runs_without_drivers():
    b = world()
    b["cross"] = {"SI=F": b["cross"]["SI=F"]}
    rep = xc.get_xasset_report(b)
    assert rep.ok and rep.components["Residual"] is None
    assert rep.components["Structural"] is None


def test_never_raises():
    assert not xc.get_xasset_report({}).ok
    b = world()
    b["gold"] = b["gold"].iloc[:30]
    assert not xc.get_xasset_report(b).ok
    b2 = world()
    b2["cross"]["ZN=F"] = b2["cross"]["ZN=F"].drop(columns=["Close"])
    assert xc.get_xasset_report(b2).error
