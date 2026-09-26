"""Tests for xau_master_signal and xau_journal (Phase 7)."""
import pathlib
import sys
from datetime import datetime, timezone
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import xau_config as cfg  # noqa: E402
import xau_journal as xj  # noqa: E402
import xau_master_signal as xms  # noqa: E402
from synth import bars_from_closes, zigzag  # noqa: E402

LONDON_NOW = datetime(2026, 9, 29, 10, 0, tzinfo=timezone.utc)     # Tue, London session
ASIA_NOW = datetime(2026, 9, 29, 2, 0, tzinfo=timezone.utc)


# ── Fakes ────────────────────────────────────────────────────────────────────
def gold(up=True):
    c = zigzag(160, 4300, drift=0.6 if up else -0.6, amp=4, period=10)
    return bars_from_closes(c, start="2026-09-28 22:00", wick=1.0)


def layers(frac=0.5, **over):
    """Every layer at `frac` of its max (sign = direction of frac)."""
    m = cfg.LAYER_MAX
    L = {
        "L1": NS(ok=True, score=frac * m["L1"], bias="", flags={}, readings={}, weight=1.0),
        "L2": NS(ok=True, score=frac * m["L2"], bias="", readings={}),
        "L3": NS(ok=True, score=frac * m["L3"], bias="", flags={},
                 readings={"gold_day_pct": 0.5 if frac > 0 else -0.5}),
        "L4": NS(ok=True, score=frac * m["L4"], bias="", flags={}),
        "L5": NS(ok=True, score=frac * m["L5"], bias="", atr=4.0, atr_regime="normal",
                 vwap_z=0.3),
        "L6": NS(ok=True, score=frac * m["L6"], bias="", active=None, targets={},
                 asian_state="above"),
        "L7": NS(ok=True, score=frac * m["L7"], bias="", flags={}, call_wall=None,
                 put_wall=None, gvz_day_pct=None, em_daily=None),
        "L8": NS(ok=True, score=frac * m["L8"], bias="", regime="trend"),
    }
    for k, v in over.items():
        for a, b in v.items():
            setattr(L[k], a, b)
    return L


def open_gate():
    return NS(ok=True, state="OPEN", reasons=[], cautions=[], mins_to_next_high=9999,
              next_high={"name": "CPI"})


def sig(L=None, g=None, gate=None, now=LONDON_NOW, basis=20.0):
    return xms.get_master_signal(L or layers(), g if g is not None else gold(),
                                 gate or open_gate(), basis, now)


def codes(s):
    return {r["code"] for r in s.rules}


# ── Scoring and tiers ────────────────────────────────────────────────────────
def test_full_agreement_is_a_tier_long():
    s = sig(layers(1.0))
    assert s.score == pytest.approx(100) and s.tier == "A" and s.action == "LONG"
    assert s.agree == 8 and "G10" in codes(s) and s.size_mult == 1.0


def test_tiers_and_neutral():
    assert sig(layers(0.5)).tier == "A"          # 50 → B, +G10 upgrade → A
    assert sig(layers(0.3)).tier == "B"          # 30 → C, +G10 → B
    s = sig(layers(0.1))
    assert s.direction == "NEUTRAL" and s.action == "WAIT" and s.size_mult == 0


def test_short_mirror():
    s = sig(layers(-0.5), g=gold(up=False))
    assert s.action == "SHORT" and s.score < 0
    p = s.plan
    assert p["tp2"] < p["tp1"] < p["entry"] < p["sl"]


# ── Rules ────────────────────────────────────────────────────────────────────
def test_g1_yield_spike_downgrades():
    s = sig(layers(0.5, L1={"flags": {"yield_spike_up": True}, "readings": {"spike_bp": 6}}))
    assert "G1" in codes(s) and s.tier == "B"


def test_g2_blocks_long_when_yields_and_dollar_against():
    L = layers(0.5, L1={"readings": {"10y_day_bp": 4.0}}, L2={"readings": {"DXY day": 0.3}},
               L3={"readings": {"gold_day_pct": 0.0}})
    s = sig(L)
    assert "G2" in s.blocked_by and s.action == "WAIT"


def test_g3_silver_nonconfirm():
    s = sig(layers(0.5, L3={"flags": {"silver_nonconfirm_high": True},
                            "readings": {"gold_day_pct": 0.5}}))
    assert "G3" in codes(s)


def test_g4_crowded_cot_into_event_halves_size():
    g = open_gate()
    g.mins_to_next_high = 3 * 1440
    s = sig(layers(0.5, L4={"flags": {"cot_crowded_long": True}}), gate=g)
    assert "G4" in codes(s) and s.size_mult == pytest.approx(0.5)


def test_g5_failed_asia_breakout_and_asia_session():
    s = sig(layers(0.5, L6={"asian_state": "inside (swept Asia H)"}))
    assert "G5" in s.blocked_by
    s2 = sig(layers(0.5), now=ASIA_NOW)
    assert "G5" in codes(s2) and s2.tier == "B"


def test_g6_chop_only_fades():
    L = layers(0.5, L8={"regime": "chop"})
    assert "G6" in sig(L).blocked_by
    L["L5"].vwap_z = -1.4
    assert "G6" not in sig(L).blocked_by


def test_g7_structural_bid_upgrades():
    s = sig(layers(0.3, L3={"flags": {"structural_bid": True},
                            "readings": {"gold_day_pct": 0.5}}))
    assert "G7" in codes(s) and s.tier == "A"      # C → +G7 → +G10 → A


def test_g8_oil_spike_with_live_yield_link():
    s = sig(layers(0.5, L3={"flags": {"oil_spike_up": True},
                            "readings": {"gold_day_pct": 0.5, "oil_day_pct": 3.4}}))
    assert "G8" in codes(s)


def test_g9_vol_spike_into_call_wall():
    s = sig(layers(0.5, L7={"gvz_day_pct": 7.0, "flags": {"near_call_wall": True}}))
    assert "G9" in s.blocked_by


def test_m1_mixed_regime_skips_c_tier():
    L = layers(0.28)
    for k in ("L1", "L2", "L3"):
        L[k].score = 0.0                                  # break G10 agreement
    L["L5"].score, L["L6"].score, L["L8"].score = 25.0, 15.0, 0.0
    L["L8"].regime = "mixed"
    s = sig(L)
    assert "M1" in codes(s) and s.action == "WAIT"


def test_gate_block_and_caution_and_atr():
    g = open_gate()
    g.state, g.reasons = "BLOCKED", ["CPI (Sep) (in 10 min)"]
    s = sig(layers(1.0), gate=g)
    assert "GATE" in s.blocked_by and s.action == "WAIT" and s.plan is not None
    g2 = open_gate()
    g2.state, g2.cautions = "CAUTION", ["CPI in 90 min"]
    assert sig(layers(1.0), gate=g2).size_mult == pytest.approx(0.5)
    assert sig(layers(1.0, L5={"atr_regime": "extreme"})).size_mult == pytest.approx(0.5)


# ── Plan geometry ────────────────────────────────────────────────────────────
def test_long_plan_geometry_and_spot():
    s = sig(layers(1.0))
    p = s.plan
    assert p["sl"] < p["entry"] < p["tp1"] < p["tp2"]
    assert cfg.STOP_MIN_ATR * 4 - 1e-6 <= p["risk"] <= cfg.STOP_MAX_ATR * 4 + 1e-6
    assert p["spot"]["entry"] == pytest.approx(p["entry"] - 20.0)
    assert p["rr2"] > p["rr1"] >= 1.0


def test_tp1_uses_liquidity_and_tp2_front_runs_wall():
    g = gold()
    price = float(g["Close"].iat[-1])
    base = sig(layers(1.0), g=g).plan
    R = base["risk"]
    L = layers(1.0, L6={"targets": {"above": {"name": "PDH", "price": price + 2.0 * R}}},
               L7={"call_wall": price + 2.6 * R})
    p = sig(L, g=g).plan
    assert p["tp1_src"] == "PDH liquidity" and p["tp1"] == pytest.approx(price + 2.0 * R - 0.4, abs=0.05)
    assert p["tp2_src"] == "option wall" and p["tp2"] < price + 3 * R


def test_sweep_extreme_sets_stop():
    g = gold()
    price = float(g["Close"].iat[-1])
    L = layers(1.0, L6={"active": {"points": 8.0, "extreme": price - 9.0}})
    p = sig(L, g=g).plan
    assert p["stop_src"] == "sweep extreme" and p["sl"] == pytest.approx(price - 9.8, abs=0.01)


def test_master_never_raises():
    assert not xms.get_master_signal({}, None).ok
    assert xms.get_master_signal({"L5": NS(ok=True)}, gold()).error


# ═════════════════════════════ Journal ═══════════════════════════════════════
def trade_sig(direction="LONG", entry=4300.0, risk=10.0, rr1=1.5, rr2=3.0):
    s = 1 if direction == "LONG" else -1
    return NS(ok=True, action=direction, tier="A", score=70.0 * s, plan={
        "entry": entry, "sl": entry - s * risk, "tp1": entry + s * rr1 * risk,
        "tp2": entry + s * rr2 * risk, "risk": risk, "rr1": rr1, "rr2": rr2})


def path(prices, start="2026-09-29 08:00", wick=0.0):
    """Bars whose closes follow `prices`; first bar is the entry bar."""
    return bars_from_closes(prices, start=start, wick=wick)


def run(sig_, bars_entry, bars_after):
    df = xj.maybe_open(xj.empty(), sig_, bars_entry, "trend")
    return xj.update(df, pd.concat([bars_entry, bars_after]))


def test_journal_tp1_then_tp2():
    e = path([4300.0])
    after = path([4305, 4316, 4322, 4331], start="2026-09-29 08:15")
    df = run(trade_sig(), e, after)
    t = df.iloc[0]
    assert t["status"] == "tp2" and bool(t["tp1_hit"])
    assert t["r_mult"] == pytest.approx(0.5 * 1.5 + 0.5 * 3.0)


def test_journal_stop_loss():
    df = run(trade_sig(), path([4300.0]), path([4295, 4289], start="2026-09-29 08:15"))
    assert df.iloc[0]["status"] == "sl" and df.iloc[0]["r_mult"] == -1


def test_journal_same_bar_takes_stop_first():
    after = path([4300.0], start="2026-09-29 08:15")
    after.iloc[0, after.columns.get_loc("High")] = 4320
    after.iloc[0, after.columns.get_loc("Low")] = 4285
    df = run(trade_sig(), path([4300.0]), after)
    assert df.iloc[0]["status"] == "sl"


def test_journal_tp1_then_breakeven_and_replay_is_deterministic():
    after = path([4310, 4316, 4308, 4299], start="2026-09-29 08:15")
    df = run(trade_sig(), path([4300.0]), after)
    t = df.iloc[0]
    assert t["status"] == "tp1+be" and t["r_mult"] == pytest.approx(0.75)
    # an open trade replayed twice gives the same state
    open_df = xj.maybe_open(xj.empty(), trade_sig(), path([4300.0]))
    bars = pd.concat([path([4300.0]), path([4304, 4316], start="2026-09-29 08:15")])
    a = xj.update(xj.update(open_df, bars), bars)
    assert a.iloc[0]["status"] == "open" and bool(a.iloc[0]["tp1_hit"])


def test_journal_expiry():
    after = path(list(np.full(cfg.JOURNAL_EXPIRY_BARS, 4305.0)), start="2026-09-29 08:15")
    df = run(trade_sig(), path([4300.0]), after)
    assert df.iloc[0]["status"] == "expired" and df.iloc[0]["r_mult"] == pytest.approx(0.5)


def test_journal_dedupe_and_reversal():
    e = path([4300.0])
    df = xj.maybe_open(xj.empty(), trade_sig(), e)
    df = xj.maybe_open(df, trade_sig(), e)                  # same bar, same direction
    assert len(df) == 1
    later = path([4304.0], start="2026-09-29 09:00")
    df = xj.maybe_open(df, trade_sig("SHORT", entry=4304.0), later)
    assert list(df["status"]) == ["reversed", "open"]
    assert df.iloc[0]["r_mult"] == pytest.approx(0.4)


def test_journal_ignores_wait_and_csv_roundtrip(tmp_path):
    w = NS(ok=True, action="WAIT", plan=None)
    assert xj.maybe_open(xj.empty(), w, path([4300.0])).empty
    df = run(trade_sig(), path([4300.0]), path([4295, 4289], start="2026-09-29 08:15"))
    p = tmp_path / "j.csv"
    assert xj.save(df, str(p))
    back = xj.load(str(p))
    assert back.iloc[0]["status"] == "sl" and back.iloc[0]["r_mult"] == -1
    assert xj.from_csv_bytes(xj.to_csv_bytes(df)).iloc[0]["id"] == df.iloc[0]["id"]


def test_journal_stats():
    rows = []
    for i, (st_, tp1, r) in enumerate([("tp2", True, 2.25), ("sl", False, -1.0),
                                       ("tp1+be", True, 0.75), ("open", False, np.nan)]):
        rows.append({"id": str(i), "opened_utc": pd.Timestamp("2026-09-01", tz="UTC"),
                     "status": st_, "tp1_hit": tp1, "r_mult": r, "tier": "A"})
    s = xj.stats(xj.normalize(pd.DataFrame(rows)),
                 now=datetime(2026, 9, 26, tzinfo=timezone.utc))
    assert s["closed"] == 3 and s["open"] == 1 and s["wins"] == 2
    assert s["win_rate"] == pytest.approx(2 / 3) and s["net_r"] == pytest.approx(2.0)
    assert s["days"] == 25 and not s["on_track"]


def test_step_never_raises(tmp_path):
    df, saved, err = xj.step(trade_sig(), path([4300.0]), path=str(tmp_path / "x.csv"))
    assert saved and not err and len(df) == 1
    df2, saved2, err2 = xj.step(trade_sig(), None, path=str(tmp_path / "y.csv"))
    assert err2 == "" or isinstance(err2, str)
