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
CONFIG_VERSION = 10
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
CACHE_TTL_SEC = 60                # yfinance bundle (NAS100 uses 60s too)
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


# ═════════════════════════════════════════════════════════════════════════════
# Phase 5 — Layer 4 flows/positioning (±10) and Layer 7 options/vol (±10)
# ═════════════════════════════════════════════════════════════════════════════
FLOWS_MAX = 10
OPTIONS_MAX = 10

# CFTC disaggregated futures-only, COMEX gold (Socrata JSON, no key needed)
COT_URL = ("https://publicreporting.cftc.gov/resource/72hh-3qpy.json"
           "?cftc_contract_market_code=088691"
           "&$order=report_date_as_yyyy_mm_dd%20DESC&$limit=170")
COT_TTL_SEC = 6 * 3600
COT_TIMEOUT = 10
COT_PCT_WEEKS = 156                   # 3-year percentile window
COT_CROWDED_PCT = 90                  # G4: net long above this percentile = crowded
COT_WASHED_PCT = 10
COT_REF_WEEK_CHG_OI = 2.0             # net change (% of OI) for full trend points
FLOW_PTS = {"ETF flow": 5, "COT trend": 3, "COT crowding": 2}

ETF_TICKERS = ["GLD", "IAU"]
ETF_DAILY_PERIOD = "1y"
ETF_TTL_SEC = 3600
ETF_FLOW_DAYS = 5
ETF_REF_SHARES_PCT = 0.5              # 5-day shares-outstanding change for full points
ETF_REF_PROXY = 0.30                  # signed $-volume / avg $-volume (5d) for full points

# Options (GLD chain, scaled to GC=F by live price ratio)
OPT_TICKER = "GLD"
OPT_TTL_SEC = 15 * 60
OPT_EXPIRIES = 2                      # nearest expiries used for OI / PCR / walls
OPT_SKEW_MIN_DAYS = 5                 # expiry used for skew must be at least this far
OPT_WALL_RANGE_PCT = 10.0             # walls searched within ±10% of spot
OPT_SKEW_OTM_PCT = 5.0                # put ~95% / call ~105% strikes
OPT_REF_SKEW_VOL = 2.0                # call-minus-put IV (vol pts) for full skew points
OPT_WALL_NEAR_EM = 0.25               # within 0.25 expected move of a wall = "at" it
OPT_WALL_MIN_MULT = 1.5               # wall OI must be ≥ this × median strike OI on its side
OPT_EM_USED_FLAG = 0.95               # today's range ≥ 95% of daily expected move
OPT_RISK_FREE = 0.04
OPT_PTS = {"Skew": 3, "Put/call": 2, "Walls": 3, "Vol": 2}

GVZ_DAILY_PERIOD = "1y"
GVZ_SPIKE_PCT = 5.0                   # GVZ day change that counts as a move


# ═════════════════════════════════════════════════════════════════════════════
# Phase 6 — calendar and trading gates
# ═════════════════════════════════════════════════════════════════════════════
# Official dates verified 2026-09-26 against:
#   BLS  bls.gov/schedule/news_release/{cpi,empsit,ppi}.htm
#   BEA  bea.gov/news/schedule            (Personal Income & Outlays = PCE, GDP)
#   Fed  federalreserve.gov/monetarypolicy/fomccalendars.htm
#   UST  home.treasury.gov tentative auction schedule (Q4 2026)
# Times are US Eastern (ET); DST is applied per date. Add or override events
# without touching code via xau_events.csv (date,time_et,name,impact).
# (date, time ET, name, impact)
CALENDAR_EVENTS = [
    # FOMC — statement 14:00 ET, press conference 14:30 ET (SEP = dot plot)
    ("2026-10-28", "14:00", "FOMC decision", "fomc"),
    ("2026-12-09", "14:00", "FOMC decision + SEP", "fomc"),
    ("2027-01-27", "14:00", "FOMC decision", "fomc"),
    ("2027-03-17", "14:00", "FOMC decision + SEP", "fomc"),
    # FOMC minutes — 3 weeks after each decision
    ("2026-10-07", "14:00", "FOMC minutes (Sep)", "medium"),
    ("2026-11-18", "14:00", "FOMC minutes (Oct)", "medium"),
    ("2026-12-30", "14:00", "FOMC minutes (Dec)", "medium"),
    # CPI
    ("2026-10-14", "08:30", "CPI (Sep)", "high"),
    ("2026-11-10", "08:30", "CPI (Oct)", "high"),
    ("2026-12-10", "08:30", "CPI (Nov)", "high"),
    # Employment Situation (NFP)
    ("2026-10-02", "08:30", "NFP (Sep)", "high"),
    ("2026-11-06", "08:30", "NFP (Oct)", "high"),
    ("2026-12-04", "08:30", "NFP (Nov)", "high"),
    # PPI
    ("2026-10-15", "08:30", "PPI (Sep)", "high"),
    ("2026-11-13", "08:30", "PPI (Oct)", "high"),
    ("2026-12-15", "08:30", "PPI (Nov)", "high"),
    # PCE (Personal Income & Outlays); GDP same time on the last three
    ("2026-09-30", "08:30", "PCE (Aug)", "high"),
    ("2026-10-29", "08:30", "PCE (Sep) + GDP Q3 adv", "high"),
    ("2026-11-25", "08:30", "PCE (Oct) + GDP Q3 2nd", "high"),
    ("2026-12-23", "08:30", "PCE (Nov) + GDP Q3 3rd", "high"),
    # Treasury auctions — results 13:00 ET (Nov refunding not yet published)
    ("2026-10-07", "13:00", "10y note auction", "medium"),
    ("2026-10-08", "13:00", "30y bond auction", "medium"),
    ("2026-12-08", "13:00", "10y note auction", "medium"),
    ("2026-12-10", "13:00", "30y bond auction", "medium"),
]
CALENDAR_VALID_UNTIL = "2026-12-31"      # warn after this: refresh the event list

# Rule-based recurring events
CLAIMS_TIME = "08:30"                    # weekly jobless claims, Thursdays
CLAIMS_MOVED = {"2026-11-26": "2026-11-25",   # Thanksgiving → Wednesday
                "2026-12-24": "2026-12-23"}   # Christmas Eve → Wednesday
ISM_MFG_TIME = "10:00"                   # 1st business day
ISM_SVC_TIME = "10:00"                   # 3rd business day
US_HOLIDAYS = {"2026-10-12": "Columbus Day (bond market shut)",
               "2026-11-11": "Veterans Day (bond market shut)",
               "2026-11-26": "Thanksgiving (CME closed)",
               "2026-11-27": "Day after Thanksgiving (early close)",
               "2026-12-24": "Christmas Eve (early close)",
               "2026-12-25": "Christmas (CME closed)",
               "2027-01-01": "New Year (CME closed)"}
EVENTS_CSV = "xau_events.csv"

# Windows (minutes before, minutes after): entries blocked inside
GATE_WINDOWS = {"fomc": (60, 45), "high": (30, 15), "medium": (15, 10), "low": (5, 5)}
CAUTION_BEFORE_HIGH_MIN = 120           # high/FOMC event within 2h → caution
ROLLOVER_BLOCK_ET = ("16:45", "18:30")  # around the CME daily break (Mon–Thu)
WEEKLY_OPEN_BLOCK_MIN = 30              # first 30 min after Sunday 18:00 ET reopen
FRIDAY_CAUTION_ET = "15:00"             # late Friday: weekend-gap risk
BLOCK_ON_EM_EXHAUSTED = True            # L7 flag: today's range ≥95% of expected move


# ═════════════════════════════════════════════════════════════════════════════
# Phase 7 — master signal, conflict rules G1–G10, trade plan, journal
# ═════════════════════════════════════════════════════════════════════════════
LAYER_MAX = {"L1": 20, "L2": 15, "L3": 20, "L4": 10, "L5": 25, "L6": 15, "L7": 10, "L8": 15}
LAYER_NAMES = {"L1": "Rates", "L2": "Dollar", "L3": "Cross-asset", "L4": "Flows",
               "L5": "Technicals", "L6": "Liquidity", "L7": "Options/vol", "L8": "Regime"}
MASTER_SCALE = 100 / sum(LAYER_MAX.values())      # ±130 → ±100

TIERS = [(60, "A"), (40, "B"), (25, "C")]          # |score| ≥ → tier; below 25 = none
TIER_SIZE = {"A": 1.0, "B": 0.75, "C": 0.5}
AGREE_FRAC = 0.2                                   # layer "agrees" at ≥20% of its max
AGREE_UPGRADE = 6                                  # G10: ≥6 of 8 agreeing
G4_EVENT_DAYS = 7                                  # crowded COT into FOMC/CPI within 7 days
G6_MR_SIGMA = 1.0                                  # chop: only fade from beyond ±1σ VWAP
G9_GVZ_SPIKE_PCT = 5.0
CAUTION_SIZE = 0.5
ATR_SIZE = {"expanding": 0.75, "extreme": 0.5}

# Plan geometry (multiples of 15m ATR)
ENTRY_ZONE_ATR = 0.3
STOP_BUFFER_ATR = 0.2
STOP_MIN_ATR = 1.0
STOP_MAX_ATR = 3.0
STOP_DEFAULT_ATR = 1.5
SWING_LOOKBACK_BARS = 60
TP1_R, TP2_R = 1.5, 3.0
TP1_TARGET_R = (1.0, 2.5)                          # use a liquidity level for TP1 in this band
TARGET_BUFFER_ATR = 0.1

# Forward-test journal
JOURNAL_CSV = "xau_journal.csv"
JOURNAL_EXPIRY_BARS = 48                           # 12h of 15m bars, then close at market
JOURNAL_TP1_PART = 0.5                             # take half at TP1, stop to breakeven


# ═════════════════════════════════════════════════════════════════════════════
# Live spot price, automatic basis, refresh
# ═════════════════════════════════════════════════════════════════════════════
SPOT_SOURCES = [
    ("Swissquote", "https://forex-data-feed.swissquote.com/public-quotes/bboquotes/instrument/XAU/USD"),
    ("gold-api", "https://api.gold-api.com/price/XAU"),
]
SPOT_TTL_SEC = 10                 # spot quote cache
SPOT_TIMEOUT = 3
SPOT_STALE_SEC = 120              # quote older than this is flagged stale
SPOT_TICK_SEC = 10                # live ticker refresh
BASIS_GC_MAX_AGE_MIN = 20         # GC=F bar must be this fresh to measure the basis
BASIS_BOUNDS = (-10.0, 150.0)     # sanity range for GC=F − spot ($)
BASIS_HISTORY = 10                # readings kept; the median is used
BASIS_DRIFT_WARN = 5.0            # warn if manual basis is this far from live
REFRESH_OPTIONS = {"Off": 0, "30 s": 30, "1 min": 60, "2 min": 120, "5 min": 300}
REFRESH_DEFAULT = "1 min"


# ═════════════════════════════════════════════════════════════════════════════
# Speed: background refresh, incremental downloads, failure back-off
# ═════════════════════════════════════════════════════════════════════════════
BUNDLE_INCR_PERIOD = "5d"         # after the first load only this much is re-downloaded
BUNDLE_FULL_REFRESH_SEC = 6 * 3600
BUNDLE_KEEP_DAYS = 62             # trim merged history to this many calendar days
BUNDLE_WAIT_FIRST = 45            # first page load waits at most this long for data
BG_NEG_TTL = 300                  # failed source: don't retry for 5 min
BG_WAIT_FIRST = 8                 # other sources: first load waits at most this long
SPOT_NEG_TTL = 60                 # spot feed unreachable: back off 60 s


# ═════════════════════════════════════════════════════════════════════════════
# Entry engine — MTF zones, FVG/IFVG, order blocks, gamma levels, confluence
# ═════════════════════════════════════════════════════════════════════════════
ZONE_TFS = {"15m": {"rule": None, "bars": 400, "weight": 1.0},
            "1h": {"rule": "1h", "bars": 300, "weight": 2.0},
            "4h": {"rule": "4h", "bars": 180, "weight": 3.0}}
ZONE_4H_OFFSET = "2h"               # 4h bins at 22/02/06/10/14/18 UTC ≈ CME day start
DISP_ATR = 1.5                      # displacement: move ≥ 1.5 ATR within DISP_BARS
DISP_BARS = 3
BASE_MAX_ATR = 1.0                  # base candle range ≤ this × ATR
FVG_MIN_ATR = 0.10                  # minimum gap size
ZONE_MAX_HEIGHT_ATR = 2.0           # taller zones are trimmed to the proximal part
KIND_WEIGHT = {"ob": 1.2, "sd": 1.0, "fvg": 0.8, "ifvg": 0.9, "liq": 1.0,
               "gamma": 1.0, "vwap": 0.5, "cpr": 0.5, "wall": 1.0}
FRESH_BONUS = 1.2
TESTED_PENALTY = 0.7                # 2+ tests

# Gamma levels (GLD chain → GC=F)
GAMMA_EXPIRIES = 3
GAMMA_RANGE_PCT = 8.0
GAMMA_TOP_N = 6
GAMMA_FLIP_GRID = 81                # price points scanned for the zero-gamma level
GAMMA_SHARE_FULL = {"gex": 0.10, "oi": 0.08, "theta": 0.08}   # share of chain total = full marks
GAMMA_STRONG = 0.60
GAMMA_MODERATE = 0.35

# Confluence & entry
ENTRY_MAX_DIST_ATR = 3.0            # look for entry zones within 3 × 15m ATR
CLUSTER_GAP_ATR = 0.25              # zones closer than this merge into one cluster
CLUSTER_MIN_SCORE = 2.0
ENTRY_DEPTH = 0.25                  # enter 25% into the cluster from its proximal edge
STOP_BEYOND_ATR = 0.25              # stop buffer beyond the cluster's distal edge
LIQ_STOP_GUARD_ATR = 0.5            # pool within this beyond the stop → stop goes past it
ENTRY_RISK_MIN_ATR = 0.6
ENTRY_RISK_MAX_ATR = 3.0
TP1_MIN_R, TP2_MIN_R, TP2_DEFAULT_R = 1.0, 2.0, 2.5
OBSTACLE_MIN_R = 1.0                # opposing zone closer than this = poor RR

# Momentum / chase
MOM_DISP_ATR = 1.2                  # last 3 bars moved ≥ 1.2 ATR in the direction
MOM_RVOL = 1.3
MOM_ADX = 25
MOM_STRONG, MOM_MODERATE = 4, 3     # of 6 checks
CHASE_FVG_BARS = 6                  # fresh FVG created within this many bars
CHASE_MAX_DIST_ATR = 1.0            # chase only if the best zone is further than this
CHASE_FVG_MAX_ATR = 2.0             # fresh FVG retest used if within this of price
CHASE_SIZE = 0.5

# Journal pending orders
PENDING_BARS = 16                   # limit order lives 4h, then cancelled
