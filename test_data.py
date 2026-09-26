"""Synthetic tests for xau_data — column layouts, fallbacks, helpers."""
import pathlib
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import xau_config as cfg  # noqa: E402
import xau_data as xd  # noqa: E402

FIELDS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def _ohlc(n=8, start="2026-09-24 22:00", price=4300.0, tz=None):
    idx = pd.date_range(start, periods=n, freq="15min", tz=tz)
    p = price + np.arange(n, dtype=float)
    return pd.DataFrame({"Open": p, "High": p + 1, "Low": p - 1, "Close": p,
                         "Adj Close": p, "Volume": 10.0}, index=idx)


def _multi(frames, ticker_first=True):
    parts = {}
    for t, df in frames.items():
        for c in df.columns:
            parts[(t, c) if ticker_first else (c, t)] = df[c]
    return pd.DataFrame(parts)


def test_extract_ticker_first_layout():
    raw = _multi({"GC=F": _ohlc(), "SI=F": _ohlc(price=50)})
    df = xd.extract_ticker(raw, "SI=F")
    assert list(df.columns) == xd.OHLC and df["Close"].iloc[0] == 50
    assert str(df.index.tz) == "UTC"


def test_extract_field_first_layout():
    raw = _multi({"GC=F": _ohlc(), "^TNX": _ohlc(price=5)}, ticker_first=False)
    assert xd.extract_ticker(raw, "^TNX")["Close"].iloc[0] == 5


def test_extract_flat_and_missing_volume():
    flat = _ohlc(tz="America/New_York").drop(columns=["Volume"])
    df = xd.extract_ticker(flat, "GC=F")
    assert (df["Volume"] == 0).all()
    assert str(df.index.tz) == "UTC"


def test_extract_missing_ticker_and_nans():
    raw = _multi({"GC=F": _ohlc()})
    assert xd.extract_ticker(raw, "CL=F").empty
    assert xd.extract_ticker(None, "GC=F").empty
    g = _ohlc()
    g.iloc[2, g.columns.get_loc("Close")] = np.nan
    assert len(xd.extract_ticker(g, "GC=F")) == 7


def test_fetch_bundle_fallback(monkeypatch):
    calls = []

    def fake(tickers, interval, period):
        calls.append(tuple(tickers))
        if tickers == ["MGC=F"]:
            return _multi({"MGC=F": _ohlc()})
        return _multi({"SI=F": _ohlc(price=50)})     # GC=F missing in batch

    monkeypatch.setattr(xd, "_download", fake)
    fn = getattr(xd.fetch_bundle, "__wrapped__", xd.fetch_bundle)
    b = fn()
    assert b["primary"] == "MGC=F" and not b["gold"].empty
    assert any("using MGC=F" in e for e in b["errors"])
    assert set(b["cross"]) == set(cfg.CROSS_ASSETS)


def test_fetch_bundle_total_failure(monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("rate limited")

    monkeypatch.setattr(xd, "_download", boom)
    fn = getattr(xd.fetch_bundle, "__wrapped__", xd.fetch_bundle)
    b = fn()
    assert b["gold"].empty
    assert any("batch download failed" in e for e in b["errors"])
    assert any("no gold data" in e for e in b["errors"])


def test_change_vs_prev_close():
    fri = _ohlc(n=4, start="2026-09-25 18:00", price=4250, tz="UTC")   # Fri trading day
    mon = _ohlc(n=4, start="2026-09-28 08:00", price=4300, tz="UTC")
    ch = xd.change_vs_prev_close(xd.extract_ticker(pd.concat([fri, mon]), "GC=F"))
    assert ch["prev_close"] == 4253 and ch["last"] == 4303
    assert abs(ch["change"] - 50) < 1e-9 and abs(ch["pct"] - 50 / 4253 * 100) < 1e-9


def test_change_single_day_has_no_prev():
    ch = xd.change_vs_prev_close(xd.extract_ticker(_ohlc(tz="UTC"), "GC=F"))
    assert ch["change"] is None


def test_to_spot_and_age():
    assert xd.to_spot(4300.0, 20.0) == 4280.0 and xd.to_spot(None, 20) is None
    df = xd.extract_ticker(_ohlc(n=1, start="2026-09-28 10:00", tz="UTC"), "GC=F")
    now = datetime(2026, 9, 28, 10, 30, tzinfo=timezone.utc)
    assert xd.bar_age_minutes(df, now) == 30
