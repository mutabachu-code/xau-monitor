"""Tests for xau_flows (L4) and xau_options (L7)."""
import io
import json
import math
import pathlib
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import xau_config as cfg  # noqa: E402
import xau_flows as xf  # noqa: E402
import xau_options as xo  # noqa: E402
from test_macro import frame, session_index  # noqa: E402

NOW = datetime(2026, 9, 25, 15, 0, tzinfo=timezone.utc)


# ═════════════════════════════ L4 flows ══════════════════════════════════════
def cot_records(nets, oi=400_000.0, start="2023-10-03"):
    """CFTC-shaped records (strings, newest first like the API)."""
    dates = pd.date_range(start, periods=len(nets), freq="7D")
    recs = []
    for d, n in zip(dates, nets):
        long_ = 100_000 + max(n, 0)
        short = long_ - n
        recs.append({"report_date_as_yyyy_mm_dd": f"{d:%Y-%m-%d}T00:00:00.000",
                     "market_and_exchange_names": "GOLD - COMMODITY EXCHANGE INC.",
                     "open_interest_all": str(oi), "m_money_positions_long_all": str(long_),
                     "m_money_positions_short_all": str(short)})
    return list(reversed(recs))


def test_parse_cot_real_shape_and_bad_rows():
    recs = [{"report_date_as_yyyy_mm_dd": "2026-09-22T00:00:00.000",
             "open_interest_all": "412800", "m_money_positions_long_all": "135699",
             "m_money_positions_short_all": "8310"},
            {"report_date_as_yyyy_mm_dd": "2026-09-15T00:00:00.000"},       # malformed
            {"report_date_as_yyyy_mm_dd": "2026-09-15T00:00:00.000",
             "open_interest_all": "420000", "m_money_positions_long_all": "142394",
             "m_money_positions_short_all": "8000"}]
    df = xf.parse_cot(recs)
    assert list(df.index) == [pd.Timestamp("2026-09-15"), pd.Timestamp("2026-09-22")]
    assert df["net"].iat[-1] == 135699 - 8310


def test_cot_crowded_long_and_trend():
    nets = list(np.linspace(20_000, 120_000, 150)) + [140_000.0]
    rep = xf.get_flow_report(cot=xf.parse_cot(cot_records(nets)), fetch=False)
    assert rep.ok and rep.flags["cot_crowded_long"]
    assert rep.components["COT crowding"] == -2
    assert rep.components["COT trend"] > 0
    assert any("G4" in n for n in rep.notes)


def test_cot_washed_out():
    nets = list(np.linspace(150_000, 60_000, 150)) + [40_000.0]
    rep = xf.get_flow_report(cot=xf.parse_cot(cot_records(nets)), fetch=False)
    assert rep.flags["cot_washed_out"] and rep.components["COT crowding"] == 2
    assert rep.components["COT trend"] < 0


def test_cot_short_history_skips_percentile():
    rep = xf.get_flow_report(cot=xf.parse_cot(cot_records([50_000, 60_000, 58_000])),
                             fetch=False)
    assert rep.components["COT crowding"] is None and rep.components["COT trend"] is not None


def shares_series(pct_5d):
    idx = pd.date_range("2026-08-01", periods=30, freq="B")
    s = np.full(30, 100e6)
    s[-5:] = 100e6 * (1 + pct_5d / 100)
    return pd.Series(s, index=idx)


def test_shares_flow_creations_and_redemptions():
    rep = xf.get_flow_report(shares=shares_series(0.8), fetch=False)
    assert rep.components["ETF flow"] == 5 and rep.etf_source == "GLD shares outstanding"
    rep = xf.get_flow_report(shares=shares_series(-0.25), fetch=False)
    assert rep.components["ETF flow"] == pytest.approx(-2.5)


def etf_daily(up_days_volume_mult=3.0, n=60):
    idx = pd.date_range("2026-06-01", periods=n, freq="B", tz="UTC")
    close = 400 + np.arange(n) * 0.1
    vol = np.full(n, 1e6)
    vol[-5:] *= up_days_volume_mult
    return pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close,
                         "Volume": vol}, index=idx)


def test_volume_proxy_fallback():
    rep = xf.get_flow_report(etfs={"GLD": etf_daily(), "IAU": etf_daily()}, fetch=False)
    assert rep.etf_source.startswith("GLD+IAU")
    assert rep.components["ETF flow"] == 5          # all up days on 3× volume


def test_flows_nothing_reachable_and_fetch_failure(monkeypatch):
    assert not xf.get_flow_report(fetch=False).ok

    def boom(*a, **k):
        raise TimeoutError("blocked")

    monkeypatch.setattr(xf.urllib.request, "urlopen", boom)
    monkeypatch.setattr(xf, "_cot_raw", getattr(xf._cot_raw, "__wrapped__", xf._cot_raw))
    assert xf.fetch_cot() is None


def test_fetch_cot_parses_response(monkeypatch):
    payload = json.dumps(cot_records([10_000, 20_000])).encode()

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(xf.urllib.request, "urlopen", lambda *a, **k: Resp(payload))
    monkeypatch.setattr(xf, "_cot_raw", getattr(xf._cot_raw, "__wrapped__", xf._cot_raw))
    df = xf.fetch_cot()
    assert len(df) == 2 and df["net"].iat[-1] == 20_000


# ═════════════════════════════ L7 options ════════════════════════════════════
def side(strikes, oi, iv, spot, is_call):
    mid = np.maximum((strikes - spot) if not is_call else (spot - strikes), 0) + 2.0
    return pd.DataFrame({"contractSymbol": [f"GLD{k}" for k in strikes], "strike": strikes,
                         "lastPrice": mid, "bid": mid - 0.1, "ask": mid + 0.1,
                         "volume": 100.0, "openInterest": oi, "impliedVolatility": iv})


def chain(spot=400.0, call_oi=None, put_oi=None, call_iv=0.20, put_iv=0.20,
          expiries=("2026-09-26", "2026-10-02", "2026-10-17")):
    strikes = np.arange(360.0, 441.0, 1.0)      # GLD: $1 strikes near the money
    n = len(strikes)
    call_oi = np.full(n, 1000.0) if call_oi is None else call_oi(strikes)
    put_oi = np.full(n, 1000.0) if put_oi is None else put_oi(strikes)
    civ = np.full(n, call_iv) if np.isscalar(call_iv) else call_iv(strikes)
    piv = np.full(n, put_iv) if np.isscalar(put_iv) else put_iv(strikes)
    exps = [{"expiry": e, "calls": side(strikes, call_oi, civ, spot, True),
             "puts": side(strikes, put_oi, piv, spot, False)} for e in expiries]
    return {"spot": spot, "expiries": exps}


def gold_today(last=4300.0, day_range=20.0):
    """Two sessions; last session spans `day_range` dollars."""
    idx = session_index(2)
    n = len(idx)
    df = frame(idx, np.zeros(n), last)
    df.iloc[-92:, df.columns.get_loc("High")] = last + day_range / 2
    df.iloc[-92:, df.columns.get_loc("Low")] = last - day_range / 2
    return df


def gvz(day_pct):
    idx = session_index(2)
    n = len(idx)
    r = np.zeros(n)
    r[-1] = math.log(1 + day_pct / 100)
    return frame(idx, r, 22.0)


def opt(ch, g=None, gz=None, gd=None):
    return xo.get_options_report(g if g is not None else gold_today(), gz, chain=ch,
                                 gvz_daily=gd, fetch=False, now=NOW)


def test_bs_gamma_peaks_atm():
    atm = xo.bs_gamma(400, 400, 10 / 365, 0.2)
    assert atm > xo.bs_gamma(400, 440, 10 / 365, 0.2) > 0
    assert xo.bs_gamma(400, 400, 0, 0.2) == 0


def test_expected_move_and_ratio():
    r = opt(chain())
    assert r.ok and r.ratio == pytest.approx(4300 / 400)
    assert r.atm_iv == pytest.approx(0.20)
    assert r.em_daily == pytest.approx(4300 * 0.2 / math.sqrt(252))
    assert not r.flags["em_exhausted"]


def test_em_exhausted_flag():
    r = opt(chain(), g=gold_today(day_range=60.0))       # EM ≈ $54
    assert r.em_used >= cfg.OPT_EM_USED_FLAG and r.flags["em_exhausted"]


def test_call_skew_is_bullish():
    r = opt(chain(call_iv=lambda k: np.where(k > 400, 0.24, 0.20),
                  put_iv=lambda k: np.where(k < 400, 0.19, 0.20)))
    assert r.skew_vol == pytest.approx(5.0) and r.components["Skew"] == 3


def test_put_skew_is_bearish():
    r = opt(chain(put_iv=lambda k: np.where(k < 400, 0.23, 0.20)))
    assert r.components["Skew"] == -3


def test_near_call_wall_bearish():
    r = opt(chain(call_oi=lambda k: np.where(k == 401, 20000.0, 500.0)))
    assert r.call_wall == pytest.approx(401 * 4300 / 400)
    assert r.flags["near_call_wall"] and r.components["Walls"] == -3


def test_near_put_wall_bullish():
    r = opt(chain(put_oi=lambda k: np.where(k == 399, 20000.0, 500.0)))
    assert r.flags["near_put_wall"] and r.components["Walls"] == 3


def test_flat_oi_is_not_a_wall_and_pin():
    r = opt(chain())                                   # uniform OI everywhere
    assert r.call_wall is None and r.put_wall is None and r.components["Walls"] is None
    r = opt(chain(call_oi=lambda k: np.where(k == 401, 20000.0, 500.0),
                  put_oi=lambda k: np.where(k == 399, 20000.0, 500.0)))
    assert r.components["Walls"] == 0 and any("Pinned" in n for n in r.notes)


def test_pcr_and_gex_sign():
    r = opt(chain(put_oi=lambda k: np.full(len(k), 3000.0)))
    assert r.pcr == pytest.approx(3.0) and r.components["Put/call"] == -2
    assert r.gex < 0
    r2 = opt(chain(call_oi=lambda k: np.full(len(k), 3000.0)))
    assert r2.gex > 0 and r2.components["Put/call"] == 2


def test_gvz_spike_with_gold_direction():
    up = gold_today()
    up.iloc[-1, up.columns.get_loc("Close")] = 4340.0          # gold +0.9% on day
    r = opt(None, g=up, gz=gvz(8.0))
    assert r.components["Vol"] == 2 and "upside chase" in r.details["Vol"]
    dn = gold_today()
    dn.iloc[-1, dn.columns.get_loc("Close")] = 4260.0
    r = opt(None, g=dn, gz=gvz(8.0))
    assert r.components["Vol"] == -2


def test_gvz_percentile():
    idx = pd.date_range("2025-09-01", periods=250, freq="B", tz="UTC")
    d = pd.DataFrame({"Open": 20.0, "High": 20.0, "Low": 20.0,
                      "Close": np.linspace(15, 30, 250), "Volume": 0.0}, index=idx)
    r = opt(None, gz=gvz(0.0), gd=d)
    assert 0 <= r.gvz_pctile <= 100


def test_no_chain_still_scores_vol():
    r = opt(None, gz=gvz(0.0))
    assert r.ok and r.components["Skew"] is None and r.components["Walls"] is None


def test_options_never_raises():
    assert not xo.get_options_report(None, fetch=False).ok
    assert not opt(None).ok                               # nothing at all
    bad = chain()
    bad["expiries"][0]["calls"] = "garbage"
    assert opt(bad).error or opt(bad).ok is False
