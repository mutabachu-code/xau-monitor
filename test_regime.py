"""Tests for xau_regime (Layer 8) using stand-in layer reports."""
import pathlib
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import xau_config as cfg  # noqa: E402
import xau_regime as xg  # noqa: E402
from test_macro import frame, market, session_index  # noqa: E402


def tech(adx=25.0, atr_regime="normal", htf=4, struct=5):
    return NS(ok=True, adx=adx, atr_regime=atr_regime,
              components={"1h trend": htf, "Structure": struct})


def macro(weight=1.0, raw=0.8, score=12.0, eff_max=20.0):
    return NS(ok=True, weight=weight, raw=raw, score=score, eff_max=eff_max)


def xa(z=0.0, bid=False, offer=False):
    return NS(ok=True, implied={"resid_z": z},
              flags={"structural_bid": bid, "structural_offer": offer})


def bundle(es_drift=0.0, vix_drift=0.0, gold_drift=0.0):
    """3 sessions; drifts are per-bar on the last session."""
    idx = session_index(3)
    n = len(idx)
    last = np.zeros(n)
    last[-92:] = 1
    rng = np.random.default_rng(0)
    return {"gold": frame(idx, rng.normal(0, 1e-5, n) + gold_drift * last, 4300.0),
            "cross": {"ES=F": frame(idx, rng.normal(0, 1e-5, n) + es_drift * last, 6500.0),
                      "^VIX": frame(idx, rng.normal(0, 1e-5, n) + vix_drift * last, 18.0)}}


QUIET = bundle()


def test_rates_led_long():
    r = xg.get_regime_report(QUIET, tech(adx=22), macro(1.0, 0.8, 12, 20),
                             macro(0.9, 0.6, 7, 13.5), xa(0.3))
    assert r.ok and r.regime == "rates-led"
    assert r.direction > 0 and r.score > 0 and r.style == "trend-follow"


def test_rates_led_needs_activity():
    # strong 20-day link but yields/dollar flat today → not rates-led
    r = xg.get_regime_report(QUIET, tech(adx=12), macro(1.0, 0.05, 0.5, 20),
                             macro(1.0, 0.05, 0.4, 15), xa(0.2))
    assert r.candidates["rates-led"] < 0.2
    assert r.regime == "chop" and r.score == 0


def test_flow_led_structural_bid():
    r = xg.get_regime_report(QUIET, tech(adx=22), macro(0.2, 0.9, 3, 4),
                             macro(0.1, 0.9, 1, 1.5), xa(1.0, bid=True))
    assert r.regime == "flow-led" and r.direction == 1.0
    assert r.score == pytest.approx(cfg.REGIME_MAX)


def test_flow_led_from_residual_when_decoupled():
    r = xg.get_regime_report(QUIET, tech(adx=21), macro(0.1, 0.2, 0.5, 2),
                             macro(0.1, 0.2, 0.3, 1.5), xa(-2.5))
    assert r.regime == "flow-led" and r.direction < 0 and r.bias == "SHORT"


def test_risk_off_haven_bid():
    b = bundle(es_drift=-0.0002, vix_drift=0.003, gold_drift=0.00005)
    r = xg.get_regime_report(b, tech(adx=22), macro(0.5, 0.2, 1, 10),
                             macro(0.5, 0.2, 1, 7), xa(0.5))
    assert r.readings["es_day_pct"] < cfg.RISKOFF_ES_PCT
    assert r.regime == "risk-off" and r.sub == "haven bid" and r.direction == 1.0


def test_risk_off_liquidation():
    b = bundle(es_drift=-0.0002, vix_drift=0.003, gold_drift=-0.0001)
    r = xg.get_regime_report(b, tech(adx=22), macro(0.5, 0.2, 1, 10),
                             macro(0.5, 0.2, 1, 7), xa(0.5))
    assert r.regime == "risk-off" and r.sub == "liquidation" and r.direction == -1.0
    assert any("liquidation" in n for n in r.notes)


def test_equity_dip_without_vol_is_not_risk_off():
    b = bundle(es_drift=-0.0002, vix_drift=0.0)
    r = xg.get_regime_report(b, tech(adx=22), None, None, None)
    assert r.candidates["risk-off"] < cfg.REGIME_MIN_STRENGTH


def test_technical_trend_direction():
    r = xg.get_regime_report(QUIET, tech(adx=38, htf=-4, struct=-5), None, None, None)
    assert r.regime == "trend" and r.direction == -1.0 and r.score < 0


def test_chop_strength_and_note():
    r = xg.get_regime_report(QUIET, tech(adx=9, atr_regime="compressed"), None, None, None)
    assert r.regime == "chop" and r.candidates["chop"] > 0.8
    assert r.score == 0 and r.style == "mean-revert"
    assert any("G6" in n for n in r.notes)


def test_mixed_when_nothing_strong():
    r = xg.get_regime_report(QUIET, tech(adx=21), macro(0.3, 0.3, 1, 6),
                             macro(0.2, 0.2, 1, 3), xa(0.4))
    assert r.regime == "mixed" and r.score == 0


def test_close_call_note():
    r = xg.get_regime_report(QUIET, tech(adx=36), macro(1.0, 0.45, 8, 20), None, xa(0))
    assert {"trend", "rates-led"} <= set(r.candidates)
    assert any("Close call" in n for n in r.notes)


def test_integration_with_real_layers():
    import xau_crossasset as xc
    import xau_dollar as xdl
    import xau_rates as xr
    import xau_technicals as xt
    b = market(beta=1.5, last_day_drift=0.00012)
    b["cross"]["CL=F"] = b["cross"]["ZN=F"]
    r = xg.get_regime_report(b, xt.get_tech_report(b["gold"]), xr.get_rates_report(b),
                             xdl.get_dollar_report(b), xc.get_xasset_report(b))
    assert r.ok and r.regime in cfg.REGIME_LABELS
    assert -cfg.REGIME_MAX <= r.score <= cfg.REGIME_MAX


def test_never_raises():
    assert not xg.get_regime_report({}).ok
    assert xg.get_regime_report(QUIET, tech=NS(ok=True)).error
