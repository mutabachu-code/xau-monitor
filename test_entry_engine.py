"""Tests for xau_zones, xau_gamma, xau_entry and pending orders in xau_journal."""
import math
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
import xau_entry as xe  # noqa: E402
import xau_gamma as xgm  # noqa: E402
import xau_journal as xj  # noqa: E402
import xau_master_signal as xms  # noqa: E402
import xau_zones as xz  # noqa: E402
from synth import bars_from_closes, zigzag  # noqa: E402
from test_flows_options import chain  # noqa: E402


def ohlc(rows, start="2026-09-29 00:00"):
    """rows = [(o, h, l, c), ...] as 15m bars."""
    idx = pd.date_range(start, periods=len(rows), freq="15min", tz="UTC")
    a = np.array(rows, float)
    return pd.DataFrame({"Open": a[:, 0], "High": a[:, 1], "Low": a[:, 2], "Close": a[:, 3],
                         "Volume": 100.0}, index=idx)


FLAT = [(4300, 4300.5, 4299.5, 4300)] * 30


# ═════════════════════════════ zones ═════════════════════════════════════════
def test_demand_zone_from_displacement():
    rows = FLAT + [(4300, 4304, 4300, 4303.5), (4303.5, 4311, 4303, 4310),
                   (4310, 4319, 4309, 4318)] + [(4318, 4318.5, 4317.5, 4318)] * 5
    z = xz.detect(ohlc(rows), "15m", 1.0)
    dem = [x for x in z if x["side"] == "support" and x["kind"] in ("sd", "ob")]
    assert dem and dem[-1]["lo"] == 4299.5 and dem[-1]["hi"] == 4300.5
    assert dem[-1]["fresh"] and dem[-1]["weight"] == pytest.approx(1.2)


def test_order_block_needs_opposite_candle_and_bos():
    rows = FLAT[:-1] + [(4300.4, 4300.5, 4299.5, 4299.8),          # bearish base
                        (4300, 4304, 4300, 4303.5), (4303.5, 4311, 4303, 4310),
                        (4310, 4319, 4309, 4318)] + [(4318, 4318.5, 4317.5, 4318)] * 5
    z = xz.detect(ohlc(rows), "15m", 1.0)
    assert any(x["kind"] == "ob" and x["side"] == "support" for x in z)


def test_fvg_and_touch_and_break_to_ifvg():
    base = FLAT + [(4300, 4301, 4299.5, 4300.8), (4301, 4309, 4300.8, 4308),
                   (4308, 4312, 4305, 4311)]                       # gap 4301 → 4305
    z = xz.detect(ohlc(base + [(4311, 4312, 4310, 4311)] * 3), "15m", 1.0)
    f = [x for x in z if x["kind"] == "fvg" and x["side"] == "support" and x["hi"] == 4305]
    assert f and 4301 <= f[0]["lo"] < 4305 and f[0]["fresh"]        # tall gaps trimmed
    touched = base + [(4311, 4311, 4304, 4306), (4306, 4310, 4305.5, 4309)]
    f2 = [x for x in xz.detect(ohlc(touched), "15m", 1.0) if x["kind"] == "fvg"
          and x["hi"] == 4305]
    assert f2 and f2[-1]["touches"] == 1 and not f2[-1]["fresh"]
    broken = base + [(4311, 4311, 4299, 4299.5), (4299.5, 4300, 4298, 4299)]
    zs = xz.detect(ohlc(broken), "15m", 1.0)
    assert not [x for x in zs if x["kind"] == "fvg" and x["side"] == "support"
                and x["hi"] == 4305]
    inv = [x for x in zs if x["kind"] == "ifvg" and x["hi"] >= 4304]
    assert inv and inv[-1]["side"] == "resistance" and inv[-1]["lo"] >= 4301


def test_mtf_report_and_resample():
    g = bars_from_closes(zigzag(1500, 4200, drift=0.2, amp=12, period=40), wick=1.0,
                         start="2026-08-20 22:00")
    rep = xz.get_zone_report(g)
    assert rep.ok and set(rep.counts) == {"15m", "1h", "4h"}
    assert sum(rep.counts.values()) > 0
    h4 = xz.resample(g, "4h")
    assert (h4.index.hour % 4 == 2).all()                          # 22/02/06… UTC bins
    assert not xz.get_zone_report(None).ok


# ═════════════════════════════ gamma ═════════════════════════════════════════
NOW = datetime(2026, 9, 25, 15, 0, tzinfo=timezone.utc)


def test_greeks_sanity():
    g = xgm.greeks(400, 400, 10 / 365, 0.2)
    assert 0.5 < g["call_delta"] < 0.55 and -0.5 < g["put_delta"] < -0.45
    assert g["gamma"] > 0 and g["call_theta"] < 0 and g["put_theta"] < 0
    far = xgm.greeks(400, 440, 10 / 365, 0.2)
    assert far["call_delta"] < 0.05 and far["gamma"] < g["gamma"]


def test_gamma_levels_flip_and_strength():
    ch = chain(call_oi=lambda k: np.where(k == 410, 30000.0, np.where(k > 400, 800.0, 100.0)),
               put_oi=lambda k: np.where(k == 390, 30000.0, np.where(k < 400, 800.0, 100.0)))
    rep = xgm.get_gamma_report(ch, 4300.0, NOW)
    assert rep.ok and rep.ratio == pytest.approx(10.75)
    by = {round(l["strike"]): l for l in rep.levels}
    assert by[410]["gex"] > 0 and by[390]["gex"] < 0
    assert by[410]["grade"] == "STRONG" and by[410]["price"] == pytest.approx(410 * 10.75)
    assert by[410]["theta_day"] > 0 and 0 < by[410]["delta"] < 0.5
    assert rep.flip is not None and 390 * 10.75 < rep.flip < 410 * 10.75
    assert rep.regime in ("positive", "negative")
    assert isinstance(rep.total_gex, float)


def test_gamma_unavailable():
    assert not xgm.get_gamma_report(None, 4300.0, NOW).ok


# ═════════════════════════════ entry engine ══════════════════════════════════
def tech(price=4300.0, atr=4.0, adx=20.0, vwap=None, z=0.3, cpr=None):
    return NS(ok=True, atr=atr, adx=adx, vwap=vwap, vwap_z=z, cpr=cpr or {}, price=price)


def liq(levels=(), rvol=1.0):
    return NS(ok=True, levels=list(levels), rvol_now=rvol)


def lvl(name, price, side, weight=1.0):
    return NS(name=name, price=price, side=side, weight=weight)


def zones(*zs):
    return NS(ok=True, zones=list(zs))


def zone(lo, hi, side="support", kind="ob", tf="1h", fresh=True, weight=None,
         created=pd.Timestamp("2026-09-28 10:00", tz="UTC")):
    w = weight if weight is not None else cfg.ZONE_TFS[tf]["weight"] * cfg.KIND_WEIGHT[kind]
    return {"lo": lo, "hi": hi, "side": side, "kind": kind, "tf": tf, "fresh": fresh,
            "weight": w, "label": f"{tf} {kind}", "created": created, "touches": 0}


def flat_gold(price=4300.0, n=40):
    return bars_from_closes(np.full(n, price), wick=1.0, start="2026-09-29 00:00")


def test_limit_entry_in_zone_with_stop_past_pool():
    z = zones(zone(4290, 4294, tf="1h", kind="ob"), zone(4291, 4293, tf="4h", kind="fvg"))
    L = liq([lvl("PDL", 4288.5, "sell"), lvl("PDH", 4320, "buy")])
    ep = xe.get_entry_plan("LONG", flat_gold(), tech(), L, z, None, None, basis=36.0)
    p = ep.plan
    assert ep.ok and p["entry_type"] == "limit"
    assert 4290 < p["entry"] <= 4294 and p["sl"] < 4288.5          # past the PDL pool
    assert p["stop_src"] == "beyond liquidity pool"
    assert p["tp1"] > p["entry"] and p["spot"]["entry"] == pytest.approx(p["entry"] - 36)
    assert "1h ob" in p["confluence"] or "4h fvg" in p["confluence"]


def test_no_zone_and_weak_momentum_waits():
    ep = xe.get_entry_plan("LONG", flat_gold(), tech(), liq(), zones(), None, None)
    assert ep.plan["entry_type"] == "none" and ep.plan["size_factor"] == 0


def test_strong_momentum_chases_fresh_fvg():
    closes = np.r_[np.full(36, 4290.0), [4292, 4296, 4300, 4306]]
    g = bars_from_closes(closes, wick=0.5, start="2026-09-29 00:00")
    f = zone(4297, 4301, kind="fvg", tf="15m", created=g.index[-2])
    ep = xe.get_entry_plan("LONG", g, tech(price=4306, adx=30, z=1.6), liq(rvol=1.8),
                           zones(f), None, None)
    p = ep.plan
    assert ep.momentum["grade"] == "STRONG" and p["entry_type"] == "chase"
    assert p["entry"] == pytest.approx(4299) and p["size_factor"] <= cfg.CHASE_SIZE


def test_obstacle_lowers_grade_and_targets_use_zones():
    base = [zone(4290, 4294, tf="4h", kind="ob"), zone(4291, 4293, tf="1h", kind="sd")]
    good = xe.get_entry_plan("LONG", flat_gold(), tech(), liq(), zones(*base), None, None).plan
    blocked = xe.get_entry_plan("LONG", flat_gold(), tech(), liq(),
                                zones(*base, zone(4296, 4298, side="resistance", kind="sd",
                                                  tf="4h")), None, None).plan
    assert blocked["obstacle"] is not None
    assert "ABC".index(blocked["grade"]) >= "ABC".index(good["grade"])


def test_short_mirror_and_gamma_confluence():
    z = zones(zone(4306, 4310, side="resistance", tf="1h", kind="ob"))
    gm = NS(ok=True, regime="positive", levels=[
        {"price": 4308.0, "strike": 401, "grade": "STRONG", "gex": 5e6}])
    ep = xe.get_entry_plan("SHORT", flat_gold(), tech(), liq([lvl("PDH", 4311.5, "buy")]),
                           z, gm, None)
    p = ep.plan
    assert p["entry_type"] == "limit" and 4306 <= p["entry"] < 4310 and p["sl"] > 4311.5
    assert any(l.startswith("γ 401") for l in p["confluence"])
    assert p["tp1"] < p["entry"]


def test_master_uses_engine_and_blocks_without_zone():
    from test_master import layers, open_gate, gold
    g = gold()
    price = float(g["Close"].iat[-1])
    none_ctx = {"zones": zones(), "gamma": None}
    s = xms.get_master_signal(layers(1.0), g, open_gate(), 20.0,
                              datetime(2026, 9, 29, 10, tzinfo=timezone.utc), none_ctx)
    assert "ZONE" in s.blocked_by and s.action == "WAIT"
    ctx = {"zones": zones(zone(price - 6, price - 3, tf="4h", kind="ob"),
                          zone(price - 5, price - 3.5, tf="1h", kind="fvg")), "gamma": None}
    s2 = xms.get_master_signal(layers(1.0), g, open_gate(), 20.0,
                               datetime(2026, 9, 29, 10, tzinfo=timezone.utc), ctx)
    assert s2.action == "LONG" and s2.plan["entry_type"] in ("limit", "market")
    assert s2.plan["entry"] < price


# ═════════════════════════════ pending orders ════════════════════════════════
def limit_sig(entry=4295.0, sl=4290.0, tp1=4302.5, tp2=4310.0, direction="LONG"):
    return NS(ok=True, action=direction, tier="A", score=70.0, plan={
        "entry_type": "limit", "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2,
        "risk": abs(entry - sl), "rr1": abs(tp1 - entry) / abs(entry - sl),
        "rr2": abs(tp2 - entry) / abs(entry - sl), "grade": "A", "entry_src": "1h ob"})


def run(rows_after, sig=None):
    e = ohlc([(4300, 4300.5, 4299.5, 4300)], start="2026-09-29 08:00")
    after = ohlc(rows_after, start="2026-09-29 08:15")
    df = xj.maybe_open(xj.empty(), sig or limit_sig(), e)
    return xj.update(df, pd.concat([e, after]))


def test_pending_then_fill_then_tp2():
    df = run([(4300, 4300, 4296, 4297), (4297, 4297, 4294.5, 4296),     # fills at 4295
              (4296, 4303, 4296, 4302), (4302, 4311, 4301, 4310)])
    t = df.iloc[0]
    assert t["order"] == "limit" and t["status"] == "tp2" and not pd.isna(t["filled_utc"])
    assert t["r_mult"] == pytest.approx(0.5 * 1.5 + 0.5 * 3.0)


def test_pending_missed_when_tp1_prints_first():
    df = run([(4300, 4303, 4299, 4302)])
    assert df.iloc[0]["status"] == "missed" and pd.isna(df.iloc[0]["r_mult"])


def test_pending_cancelled_after_timeout():
    df = run([(4300, 4301, 4298, 4300)] * (cfg.PENDING_BARS + 1))
    assert df.iloc[0]["status"] == "cancelled"


def test_fill_bar_stop_is_conservative():
    df = run([(4300, 4300, 4289, 4291)])
    assert df.iloc[0]["status"] == "sl" and df.iloc[0]["r_mult"] == -1


def test_replace_only_when_entry_moves_and_stats_exclude_unfilled():
    e = ohlc([(4300, 4300.5, 4299.5, 4300)], start="2026-09-29 08:00")
    e2 = ohlc([(4300, 4300.5, 4299.5, 4300)], start="2026-09-29 09:00")
    df = xj.maybe_open(xj.empty(), limit_sig(), e)
    df = xj.maybe_open(df, limit_sig(entry=4294.0, sl=4289.0), e2)     # 0.2R move → keep
    assert len(df) == 1
    df = xj.maybe_open(df, limit_sig(entry=4291.0, sl=4286.0), e2)     # 0.8R → replace
    assert list(df["status"]) == ["replaced", "pending"]
    st = xj.stats(df, now=datetime(2026, 9, 30, tzinfo=timezone.utc))
    assert st["closed"] == 0 and st["pending"] == 1 and st["unfilled"] == 1


def test_opposite_signal_cancels_pending():
    e = ohlc([(4300, 4300.5, 4299.5, 4300)], start="2026-09-29 08:00")
    e2 = ohlc([(4300, 4300.5, 4299.5, 4300)], start="2026-09-29 09:00")
    df = xj.maybe_open(xj.empty(), limit_sig(), e)
    df = xj.maybe_open(df, limit_sig(entry=4305, sl=4310, tp1=4297.5, tp2=4290,
                                     direction="SHORT"), e2)
    assert list(df["status"]) == ["cancelled", "pending"]


def test_diffuse_chain_has_no_strong_levels():
    import xau_gamma as _g
    _g._CACHE["key"] = None
    rep = xgm.get_gamma_report(chain(), 4300.0, NOW)            # uniform OI on every strike
    assert rep.ok and rep.levels
    assert all(l["grade"] != "STRONG" for l in rep.levels)


# ── Yahoo sometimes blanks GLD open interest ────────────────────────────────
def _zero_oi_chain(volume):
    ch = chain(call_oi=lambda k: np.zeros(len(k)), put_oi=lambda k: np.zeros(len(k)))
    for e in ch["expiries"]:
        for s in ("calls", "puts"):
            e[s] = e[s].copy()
            e[s]["volume"] = volume(e[s]["strike"].to_numpy()) if callable(volume) else volume
    return ch


def test_gamma_suppressed_when_oi_and_volume_missing():
    xgm._CACHE.clear()
    rep = xgm.get_gamma_report(_zero_oi_chain(0.0), 4300.0, NOW)
    assert not rep.ok and "suppressed" in rep.error and rep.levels == []
    assert rep.regime in ("", None) or not rep.ok


def test_gamma_volume_proxy_when_oi_missing():
    xgm._CACHE.clear()
    vol = lambda k: np.where(k == 410, 5000.0, np.where((k >= 395) & (k <= 405), 50.0, 0.0))
    rep = xgm.get_gamma_report(_zero_oi_chain(vol), 4300.0, NOW)
    assert rep.ok and "volume proxy" in rep.weight_mode
    assert rep.oi_coverage == 0
    assert any(round(l["strike"]) == 410 for l in rep.levels)
    assert all(l["call_oi"] + l["put_oi"] > 0 for l in rep.levels)
    assert any("volume" in n for n in rep.notes)


def test_gamma_skips_empty_strikes_with_oi():
    xgm._CACHE.clear()
    ch = chain(call_oi=lambda k: np.where((k >= 395) & (k <= 410), 2000.0, 0.0),
               put_oi=lambda k: np.where((k >= 390) & (k <= 405), 2000.0, 0.0))
    rep = xgm.get_gamma_report(ch, 4300.0, NOW)
    assert rep.ok and rep.weight_mode == "open interest"
    assert all(390 <= l["strike"] <= 410 for l in rep.levels)
    assert all(l["call_oi"] + l["put_oi"] > 0 for l in rep.levels)
