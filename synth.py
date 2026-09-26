"""Synthetic 15m gold bars for tests."""
import numpy as np
import pandas as pd

DAY_START = {  # trading day -> first bar (18:00 ET prior evening, EDT = 22:00 UTC)
}


def bars_from_closes(closes, start="2026-09-24 22:00", wick=0.5, volume=100.0,
                     freq="15min"):
    closes = np.asarray(closes, dtype=float)
    idx = pd.date_range(start, periods=len(closes), freq=freq, tz="UTC")
    opens = np.r_[closes[0], closes[:-1]]
    hi = np.maximum(opens, closes) + wick
    lo = np.minimum(opens, closes) - wick
    vol = np.full(len(closes), volume) if np.isscalar(volume) else np.asarray(volume, float)
    return pd.DataFrame({"Open": opens, "High": hi, "Low": lo, "Close": closes,
                         "Volume": vol}, index=idx)


def trend(n, start=4300.0, step=0.8, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    return start + step * np.arange(n) + rng.normal(0, noise, n)


def zigzag(n, start=4300.0, drift=0.5, amp=6.0, period=12):
    t = np.arange(n)
    return start + drift * t + amp * np.sin(2 * np.pi * t / period)


def trading_days(days, first="2026-09-20 22:00", bars_per_day=92, price_fn=None):
    """Consecutive weekday trading days (skips weekends), 23h each with a
    1h break. price_fn(day_k, n) -> closes for that day."""
    frames, start = [], pd.Timestamp(first, tz="UTC")
    k = 0
    d = start
    while k < days:
        # trading day label = calendar date of d + 1 day (ET 18:00 start)
        label = (d + pd.Timedelta(hours=2)).tz_convert("America/New_York") + pd.Timedelta(hours=6)
        if label.weekday() < 5:
            closes = price_fn(k, bars_per_day) if price_fn else trend(bars_per_day, 4300 + k)
            frames.append(bars_from_closes(closes, start=d))
            k += 1
        d += pd.Timedelta(days=1)
    return pd.concat(frames)
