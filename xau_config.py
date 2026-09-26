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


# ═════════════════════════════════════════════════════════════════════════════
# Phase 1 — Layer 5 technicals (±25) and Layer 6 liquidity (±15)
# ═════════════════════════════════════════════════════════════════════════════
TECH_MAX = 25
LIQ_MAX = 15

EMA_FAST, EMA_MID, EMA_SLOW = 9, 21, 50       # 15m
HTF_EMA_FAST, HTF_EMA_SLOW = 50, 200          # 1h, resampled from 15m
RSI_N = 14
ATR_N = 14
ADX_N = 14
ADX_CHOP = 18                                 # below = chop, technical score damped
CHOP_DAMPING = 0.6
SWING_N = 2                                   # fractal half-width (bars each side)
VWAP_SLOPE_BARS = 4
VWAP_STRETCH_SIGMA = 2.0                      # beyond this, trend points halved
RSI_DECAY_BARS = 5
RSI_DECAY_DROP = 3.0                          # RSI points lost while price rises
DIVERGENCE_LOOKBACK = 40                      # bars searched for swing pairs
STRUCTURE_RECENT_BARS = 12                    # BoS/MSS counts as fresh within this
ATR_REGIME_BARS = 480                         # ~5 trading days of 15m bars
ATR_REGIMES = ((0.7, "compressed"), (1.3, "normal"), (2.0, "expanding"),
               (float("inf"), "extreme"))

CPR_LOOKBACK_DAYS = 20
CPR_NARROW_PCT = 0.10                         # fallback thresholds (% of pivot)
CPR_WIDE_PCT = 0.30

TECH_BIAS_THRESHOLD = 8                       # |score| >= this gives LONG/SHORT

# Liquidity
SWEEP_SCAN_BARS = 16                          # bars searched for sweeps (4h)
SWEEP_VALID_BARS = 8                          # sweep scores only if this recent (2h)
SWEEP_EPS_ATR = 0.05                          # min pierce beyond level, in ATR
SWEEP_RECLAIM_BARS = 2                        # bars allowed to close back inside
SWEEP_BASE_PTS = 8
SWEEP_RVOL_PTS = 3
SWEEP_FOLLOW_PTS = 2
SWEEP_FOLLOW_ATR = 0.25
RVOL_CONFIRM = 1.5
RVOL_DAYS = 20
EQ_TOL_ATR = 0.10                             # equal highs/lows tolerance
EQ_MIN_TOL = 1.5                              # $ floor for tolerance
EQ_LOOKBACK_BARS = 96                         # one trading day

LEVEL_WEIGHTS = {"PDH": 1.0, "PDL": 1.0, "PWH": 1.0, "PWL": 1.0,
                 "Asia H": 0.9, "Asia L": 0.9, "EQH": 0.8, "EQL": 0.8,
                 "Round": 0.5}
SESSION_MULT = {"Asia": 0.5, "London": 1.0, "London–NY overlap": 1.0,
                "New York": 0.8, "Closed": 0.0}
LIQ_BIAS_THRESHOLD = 5
