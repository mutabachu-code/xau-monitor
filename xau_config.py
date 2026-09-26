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

# Bumped whenever a phase adds settings; app.py refuses to run on an older copy.
CONFIG_VERSION = 4
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


# ═════════════════════════════════════════════════════════════════════════════
# Phase 2 — Layer 1 rates (±20, dynamic) and Layer 2 dollar (±15, dynamic)
# ═════════════════════════════════════════════════════════════════════════════
RATES_MAX = 20
DOLLAR_MAX = 15

# Extra intraday tickers pulled in the same batch download (not in the strip).
# Treasury futures trade ~23h on CME, unlike ^TNX which only updates in US hours.
EXTRA_TICKERS = {
    "ZN=F": "10y T-Note fut",
    "ZT=F": "2y T-Note fut",
    "^FVX": "US 5y",
    "EURUSD=X": "EUR/USD",
    "ES=F": "S&P 500 fut",        # Phase 4: risk-off detection
    "^VIX": "VIX",
}
# Approximate modified duration used to turn futures % moves into yield bp
FUT_DURATION = {"ZN=F": 6.5, "ZT=F": 1.9}

# FRED daily series (no API key needed for the CSV endpoint)
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
FRED_REAL_10Y = "DFII10"          # 10y TIPS real yield, %
FRED_BREAKEVEN_10Y = "T10YIE"     # 10y breakeven inflation, %
FRED_TTL_SEC = 6 * 3600
FRED_TIMEOUT = 8
REAL_YIELD_HIGH = 2.5             # WGC: level associated with high opportunity cost

# Reference moves that map to a full-strength (±1) component
REF_10Y_DAY_BP = 8.0
REF_2Y_DAY_BP = 8.0
REF_SPIKE_BP = 4.0                # 30-min move
SPIKE_BARS = 2                    # 2 × 15m = 30 min
SPIKE_FLAG_BP = 5.0               # G1: short-end up ≥5bp in 30 min
REF_REAL_5D_BP = 10.0
RATES_WEIGHTS = {"10y day": 0.30, "2y day": 0.30, "30-min spike": 0.20, "Real yield": 0.20}

REF_DXY_DAY_PCT = 0.40
REF_EUR_DAY_PCT = 0.40
REF_JPY_DAY_PCT = 0.60
REF_DXY_1H_PCT = 0.15
DOLLAR_WEIGHTS = {"DXY day": 0.35, "EUR/USD day": 0.25, "USD/JPY day": 0.20,
                  "DXY 1h": 0.20}

# Correlation-driven weighting
CORR_SHORT_DAYS = 5
CORR_LONG_DAYS = 20
BARS_PER_DAY = 92
CORR_MIN_POINTS = 150
CORR_FULL = 0.40                  # |inverse corr| at which the layer gets full weight
CORR_BLEND_LONG = 0.6             # weight of the 20-day corr vs 5-day
MACRO_STALE_MIN = 60              # drop an input whose last bar is this much older than gold
MACRO_BIAS_FRAC = 0.30            # |score| ≥ this × max → LONG/SHORT


# ═════════════════════════════════════════════════════════════════════════════
# Phase 3 — Layer 3 cross-asset agreement (±20)
# ═════════════════════════════════════════════════════════════════════════════
XASSET_MAX = 20
IMPLIED_TRAIN_DAYS = 20            # regression window (prior trading days)
IMPLIED_MIN_TRAIN = 300            # bars needed to fit
IMPLIED_MIN_WINDOW = 16            # bars scored: today's bars, at least this many
RESID_Z_FULL = 2.0                 # residual z that earns the full residual points
XASSET_PTS = {"Residual": 8, "Structural": 5, "Silver": 4, "Metals context": 3}

STRUCT_YIELD_BP = 2.0              # yields up at least this much on the day …
STRUCT_DXY_PCT = 0.10              # … and dollar up at least this much …
STRUCT_GOLD_PCT = 0.15             # … while gold is up at least this much = structural bid

SILVER_HIGH_TOL_ATR = 0.50         # gold within this of its session high = "at the high"
SILVER_LAG_PCT = 0.30              # silver this far below its own session high = not confirming
NEW_EXTREME_BARS = 4               # gold's high must be set within the last N bars

OIL_SPIKE_PCT = 3.0                # G8: oil ±3% on the day


# ═════════════════════════════════════════════════════════════════════════════
# Phase 4 — Layer 8 regime classifier (±15)
# ═════════════════════════════════════════════════════════════════════════════
REGIME_MAX = 15
REGIME_MIN_STRENGTH = 0.35          # below this for every candidate → "mixed"
RATES_LED_ACTIVE_RAW = 0.5          # |raw| L1/L2 signal that counts as fully active macro

# risk-off: equities down AND volatility up
RISKOFF_ES_PCT = -1.0               # ES day % that counts as a full-strength sell-off
RISKOFF_VIX_CHG_PCT = 15.0          # VIX day % rise for full strength
RISKOFF_VIX_LEVEL = 25.0            # or VIX at/above this level
RISKOFF_HAVEN_GOLD_PCT = 0.10       # gold up this much during risk-off = haven bid

# technical trend
TREND_ADX_START = 20                # ADX where trend strength starts
TREND_ADX_FULL = 40                 # ADX for full trend strength

REGIME_LABELS = {
    "rates-led": "Rates/dollar-led",
    "flow-led": "Flow-led (decoupled)",
    "risk-off": "Risk-off",
    "trend": "Technical trend",
    "chop": "Chop",
    "mixed": "Mixed / no clear driver",
}
REGIME_STYLE = {           # what the master signal should favour
    "rates-led": "trend-follow",
    "flow-led": "buy-dips / sell-rips with the flow",
    "risk-off": "event-driven, reduced size",
    "trend": "trend-follow",
    "chop": "mean-revert",
    "mixed": "selective, reduced size",
}
REGIME_PLAYBOOK = {
    "rates-led": "Trade with the yields/dollar read. Watch 2y spikes and US data; "
                 "fade gold moves that fight the macro tape.",
    "flow-led": "Macro headwinds matter less — the bid/offer is gold's own flow. "
                "Lean on L3 residual, VWAP pullbacks and structure.",
    "risk-off": "Two sub-cases: haven bid (gold up with equities down) favours "
                "longs; liquidation (gold sold with equities) favours patience — "
                "wait for the flush to end.",
    "trend": "No macro driver dominates but price is trending. Follow structure, "
             "buy/sell pullbacks to EMA21 or VWAP.",
    "chop": "Range day. Mean-revert between VWAP bands, CPR and liquidity levels; "
            "no breakout entries.",
    "mixed": "Drivers disagree. Only take A-grade setups with multi-layer agreement.",
}
