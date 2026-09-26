"""Synthetic tests for xau_macro, xau_rates (L1) and xau_dollar (L2)."""
import io
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import xau_config as cfg  # noqa: E402
import xau_data as xd  # noqa: E402
import xau_dollar as xdl  # noqa: E402
import xau_macro as xm  # noqa: E402
import xau_rates as xr  # noqa: E402


# ── Synthetic market ─────────────────────────────────────────────────────────
def session_index(days=22):
    """Weekday CME sessions of 92 × 15m bars starting 22:00 UTC (EDT)."""
    out, d = [], pd.Timestamp("2026-08-23 22:00", tz="UTC")
    while len(out) < days:
        label = d.tz_convert("America/New_York") + pd.Timedelta(hours=6)
        if label.weekday() < 5:
            out.append(pd.date_range(d, periods=92, freq="15min"))
        d += pd.Timedelta(days=1)
    return out[0].append(out[1:])


def frame(idx, rets, base):
    p = base * np.exp(np.cumsum(rets))
    o = np.r_[p[0], p[:-1]]
    return pd.DataFrame({"Open": o, "High": np.maximum(o, p) * 1.0002,
                         "Low": np.minimum(o, p) * 0.9998, "Close": p,
                         "Volume": 100.0}, index=idx)


def market(beta=1.5, noise=0.0004, seed=0, last_day_drift=0.0, tail=None):
    """ZN, ZT, DXY, EURUSD share one shock z. gold = beta·z + noise, so
    beta>0 means gold rises when yields fall and the dollar falls."""
    rng = np.random.default_rng(seed)
    idx = session_index()
    n = len(idx)
    z = rng.normal(0, 0.0004, n)
    z[-92:] += last_day_drift                     # directional final session
    if tail is not None:
        z[-len(tail):] += tail
    g = beta * z + rng.normal(0, noise, n)
    return {
        "gold": frame(idx, g, 4300.0),
        "primary": "GC=F",
        "cross": {
            "ZN=F": frame(idx, z * 0.5, 110.0),
            "ZT=F": frame(idx, z * 0.15, 103.0),
            "DX-Y.NYB": frame(idx, -z * 0.8, 101.0),
            "EURUSD=X": frame(idx, z * 0.8, 1.10),
            "JPY=X": frame(idx, -z * 0.6, 150.0),
            "^TNX": frame(idx, -z * 2, 4.9),
            "^FVX": frame(idx, -z * 2, 4.6),
        },
        "errors": [],
    }


# ── Helpers ──────────────────────────────────────────────────────────────────
def test_dynamic_weight():
    assert xm.dynamic_weight(0.4, 0.4) == 1.0
    assert xm.dynamic_weight(0.2, 0.2) == pytest.approx(0.5)
    assert xm.dynamic_weight(-0.3, -0.3) == 0.0
    assert xm.dynamic_weight(None, None) is None
    assert xm.dynamic_weight(None, 0.2) == pytest.approx(0.5)
    assert xm.dynamic_weight(0.8, 0.0) == pytest.approx(0.8)  # 0.4·0.8 / 0.4


def test_aligned_returns_drops_gaps():
    idx = session_index(3)
    df = frame(idx, np.full(len(idx), 0.001), 100.0)
    r = xm.aligned_returns(df, df, lambda s: s.pct_change())
    assert len(r) == len(idx) - 3      # first bar + 2 session-open gaps removed


def test_normalize_yield_and_fut_bp():
    df = pd.DataFrame({"Open": [49.5], "High": [49.5], "Low": [49.5], "Close": [49.5]})
    assert xm.normalize_yield(df)["Close"].iat[0] == pytest.approx(4.95)
    assert xm.fut_pct_to_bp(-0.13, 6.5) == pytest.approx(2.0)


def test_usable_staleness():
    b = market()
    stale = b["cross"]["DX-Y.NYB"].iloc[:-10]      # 150 min behind gold
    assert xm.usable(b["cross"]["DX-Y.NYB"], b["gold"])
    assert not xm.usable(stale, b["gold"])


# ── FRED ─────────────────────────────────────────────────────────────────────
def test_fred_parse_and_failure(monkeypatch):
    csv = b"observation_date,DFII10\n2026-09-21,2.40\n2026-09-22,.\n2026-09-23,2.46\n"

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(xm.urllib.request, "urlopen", lambda *a, **k: Resp(csv))
    raw = getattr(xm._fred_raw, "__wrapped__", xm._fred_raw)
    monkeypatch.setattr(xm, "_fred_raw", raw)
    s = xm.fetch_fred("DFII10")
    assert list(s.values) == [2.40, 2.46]

    def boom(*a, **k):
        raise TimeoutError("blocked")

    monkeypatch.setattr(xm.urllib.request, "urlopen", boom)
    assert xm.fetch_fred("DFII10") is None


# ── L1 rates ─────────────────────────────────────────────────────────────────
def test_rates_long_when_yields_fall_and_correlated():
    b = market(beta=1.5, last_day_drift=0.00012)      # futures up → yields down
    rep = xr.get_rates_report(b)
    assert rep.ok, rep.error
    assert rep.corr_long < -0.5 and rep.weight == 1.0
    assert rep.readings["10y_day_bp"] < 0 and rep.readings["2y_day_bp"] < 0
    assert rep.score >= cfg.MACRO_BIAS_FRAC * cfg.RATES_MAX and rep.bias == "LONG"
    assert rep.components["Real yield"] is None           # no FRED passed
    assert len(rep.corr_series) > 0


def test_rates_short_when_yields_rise():
    rep = xr.get_rates_report(market(beta=1.5, last_day_drift=-0.00012))
    assert rep.score < 0 and rep.bias == "SHORT"


def test_rates_muted_when_decoupled():
    b = market(beta=0.0, noise=0.0006, last_day_drift=0.00012)
    rep = xr.get_rates_report(b)
    assert rep.weight is not None and rep.weight < 0.25
    assert abs(rep.score) < 0.25 * cfg.RATES_MAX + 1e-9
    assert any("barely trading off yields" in n for n in rep.notes)
    assert abs(rep.raw) > 0.2                # the move is there, just not trusted


def test_rates_spike_flag():
    tail = np.r_[np.zeros(4), [-0.0045, -0.0045]]      # ZT ≈−0.07%/bar ⇒ ≈+7bp in 30m
    rep = xr.get_rates_report(market(tail=tail))
    assert rep.flags["yield_spike_up"] and rep.readings["spike_bp"] >= cfg.SPIKE_FLAG_BP
    assert rep.components["30-min spike"] == -1.0
    assert any("G1" in n for n in rep.notes)


def test_rates_tnx_fallback_and_scaling():
    b = market(last_day_drift=0.00012)
    b["cross"].pop("ZN=F")
    b["cross"].pop("ZT=F")
    tnx = b["cross"]["^TNX"].copy()
    for c in ("Open", "High", "Low", "Close"):
        tnx[c] *= 10                                   # quoted as 49.x
    b["cross"]["^TNX"] = tnx
    rep = xr.get_rates_report(b)
    assert rep.ok and "^TNX" in rep.details["10y day"]
    assert "^FVX" in rep.details["2y day"]
    assert rep.readings["tnx_level"] < 20
    assert "US hours" in rep.corr_source


def test_rates_real_yield_component_and_note():
    days = pd.date_range("2026-09-14", periods=8, freq="B")
    fred = {cfg.FRED_REAL_10Y: pd.Series([2.40, 2.42, 2.45, 2.47, 2.50, 2.52, 2.55, 2.58],
                                         index=days),
            cfg.FRED_BREAKEVEN_10Y: pd.Series([2.3] * 8, index=days)}
    rep = xr.get_rates_report(market(), fred)
    assert rep.readings["real_10y"] == pytest.approx(2.58)
    assert rep.readings["real_5d_bp"] == pytest.approx(13.0)
    assert rep.components["Real yield"] == -1.0
    assert any("opportunity cost" in n for n in rep.notes)


def test_rates_never_raises():
    assert not xr.get_rates_report({}).ok
    assert not xr.get_rates_report({"gold": xd._empty(), "cross": {}}).ok
    bad = market()
    bad["cross"]["ZN=F"] = bad["cross"]["ZN=F"].drop(columns=["Close"])
    assert xr.get_rates_report(bad).error


# ── L2 dollar ────────────────────────────────────────────────────────────────
def test_dollar_long_when_dollar_falls():
    rep = xdl.get_dollar_report(market(beta=1.5, last_day_drift=0.00012))
    assert rep.ok, rep.error
    assert rep.readings["DXY day"] < 0 and rep.readings["EUR/USD day"] > 0
    assert rep.corr_long < -0.5 and rep.weight == 1.0
    assert rep.bias == "LONG"


def test_dollar_stale_dxy_uses_eurusd():
    b = market(beta=1.5, last_day_drift=-0.00012)
    b["cross"]["DX-Y.NYB"] = b["cross"]["DX-Y.NYB"].iloc[:-12]
    rep = xdl.get_dollar_report(b)
    assert rep.components["DXY day"] is None
    assert "EUR/USD inverted" in rep.details["DXY 1h"]
    assert any("stale" in n for n in rep.notes)
    assert rep.score < 0


def test_dollar_eurusd_correlation_fallback():
    b = market(beta=1.5)
    b["cross"].pop("DX-Y.NYB")
    rep = xdl.get_dollar_report(b)
    assert rep.corr_source == "gold vs EUR/USD" and rep.corr_expected == "positive"
    assert rep.weight == 1.0


def test_dollar_never_raises():
    assert not xdl.get_dollar_report({}).ok
    assert not xdl.get_dollar_report({"gold": market()["gold"], "cross": {}}).ok


# ── Data layer includes the extra tickers ────────────────────────────────────
def test_fetch_bundle_includes_extra_tickers(monkeypatch):
    seen = {}

    def fake(tickers, interval, period):
        seen.setdefault("t", []).extend(tickers)
        return pd.DataFrame()

    monkeypatch.setattr(xd, "_download", fake)
    fn = getattr(xd.fetch_bundle, "__wrapped__", xd.fetch_bundle)
    b = fn()
    assert set(cfg.EXTRA_TICKERS) <= set(seen["t"])
    assert set(cfg.EXTRA_TICKERS) <= set(b["cross"])
