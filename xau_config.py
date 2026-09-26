"""
xau_config.py — central configuration for xau-monitor.

Session times are defined in each venue's HOME timezone so that daylight
saving is handled by zoneinfo, then converted to the display timezone
(Africa/Nairobi, EAT). Never hard-code EAT hours anywhere else.

Decisions (2026-09-26):
  - Signal timeframe: 15m intraday swing (1h used as higher timeframe)
  - Basis (GC=F minus broker XAUUSD): manual sidebar input until MT5 (Phase 8)
  - Forward-test target: >75% win rate over 60 days
"""
from dataclasses import dataclass
from datetime import time
from typing import Optional, Tuple

# ── Timezones ────────────────────────────────────────────────────────────────
DISPLAY_TZ = "Africa/Nairobi"
ET = "America/New_York"
LONDON = "Europe/London"
SHANGHAI = "Asia/Shanghai"

# ── Instruments ──────────────────────────────────────────────────────────────
PRIMARY = "GC=F"                  # COMEX gold front month
PRIMARY_FALLBACKS = ["MGC=F"]     # micro gold if GC=F fails

CROSS_ASSETS = {
    "DX-Y.NYB": "DXY",
    "^TNX": "US 10y",
    "JPY=X": "USD/JPY",
    "SI=F": "Silver",
    "CL=F": "WTI",
    "HG=F": "Copper",
    "^GVZ": "GVZ",
}
YIELD_TICKERS = {"^TNX"}          # changes shown in basis points

# ── Data ─────────────────────────────────────────────────────────────────────
SIGNAL_INTERVAL = "15m"
SIGNAL_PERIOD = "60d"             # yfinance max for sub-hourly bars
HTF_INTERVAL = "1h"
HTF_PERIOD = "180d"
CACHE_TTL_SEC = 120
STALE_AFTER_MIN = 45              # warn if last bar older than this while market open

# ── Pricing ──────────────────────────────────────────────────────────────────
DEFAULT_BASIS = 20.0              # $ ; GC=F minus broker XAUUSD spot
ROUND_NUMBER_STEP = 50.0
ROUND_NUMBER_COUNT = 3            # levels shown above and below price

# ── Forward test ─────────────────────────────────────────────────────────────
WIN_RATE_TARGET = 0.75
FORWARD_TEST_DAYS = 60


# ── Session calendar ─────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SessionEvent:
    key: str
    label: str
    tz: str
    start: time
    end: Optional[time]           # None = point-in-time event
    days: Tuple[int, ...]         # weekdays in HOME tz (0=Mon ... 6=Sun)
    kind: str                     # session | fix | data | break
    note: str = ""


MON_FRI = (0, 1, 2, 3, 4)

SESSION_EVENTS = (
    SessionEvent("cme_break", "CME daily break", ET, time(17, 0), time(18, 0),
                 (0, 1, 2, 3), "break", "Spreads widen, no entries"),
    SessionEvent("weekly_close", "CME weekly close", ET, time(17, 0), None,
                 (4,), "break", "Market shut until Sunday 18:00 ET"),
    SessionEvent("asia_open", "CME reopen / Asia", ET, time(18, 0), None,
                 (6, 0, 1, 2, 3), "session", "Thin liquidity, Asian range builds"),
    SessionEvent("sge_open", "Shanghai Gold Exchange open", SHANGHAI, time(9, 0), None,
                 MON_FRI, "session", "Chinese physical tone"),
    SessionEvent("london_open", "London open", LONDON, time(8, 0), None,
                 MON_FRI, "session", "Asian range often swept"),
    SessionEvent("lbma_am", "LBMA AM auction", LONDON, time(10, 30), None,
                 MON_FRI, "fix", "Benchmark pivot"),
    SessionEvent("comex_open", "COMEX open", ET, time(8, 20), None,
                 MON_FRI, "session", "Futures volume arrives"),
    SessionEvent("us_data", "US data window", ET, time(8, 30), None,
                 MON_FRI, "data", "CPI / NFP / PCE / claims on release days"),
    SessionEvent("ny_cash", "NY cash equity open", ET, time(9, 30), None,
                 MON_FRI, "session", "Risk sentiment feeds in"),
    SessionEvent("lbma_pm", "LBMA PM auction", LONDON, time(15, 0), None,
                 MON_FRI, "fix", "Benchmark pivot, London winds down"),
    SessionEvent("london_close", "London close", LONDON, time(17, 0), None,
                 MON_FRI, "session", "Frequent reversal point"),
)

# Trading windows used for chart bands and the live session label
LONDON_WINDOW = (time(8, 0), time(17, 0))   # London time
NY_WINDOW = (time(8, 0), time(17, 0))       # ET

BAND_COLORS = {
    "London": "rgba(66, 135, 245, 0.07)",
    "New York": "rgba(245, 166, 35, 0.07)",
    "London–NY overlap": "rgba(46, 204, 113, 0.12)",
}
