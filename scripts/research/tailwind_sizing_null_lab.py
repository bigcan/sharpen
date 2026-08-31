"""TAILWIND sizing-overlay falsification lab -- is the risk machinery de-levering in costume?

Applies the S553-cont-170 method (``scripts/research/risk_overlay_lab.py``) to the TAILWIND
book (TSMOM engine + BAB crash hedge, 18 free-data ETFs, monthly rebalance).

The TSMOM *signal* is not under test. What is under test is the SIZING stack:
  * per-asset vol targeting     ``target_vol_asset / realized_vol``  (``vol_scaled_weights``)
  * per-asset leverage cap      ``lev_cap``
  * sleeve risk-parity combine  ``portfolio_frontier.risk_parity``   (full-sample = LEAK-2)
  * the BAB sleeve itself       (crash hedge, inverse-vol weighted)
  * the gross-exposure cap      ``max_gross_exposure``               (executor only)

RULE 1 -- de-levering is the null.  Every arm is scored against what a constant leverage
multiplier ``lev_scale=k`` would have produced AT THE SAME median gross exposure
(``PF_excess``), never against the raw baseline.

RULE 2 -- flip the edge sign.  Every arm is re-run on the sign-inverted conviction stream.
An arm whose advantage is real survives; one exploiting the sign reverses.

LEAK-2 discipline: the sleeve risk-parity scalars are offered in four flavours -- FULL
(the leaky full-sample constant every certifying number uses), TRAILING (the config's own
causal 252d/63min monthly-meta spec), FROZEN (fit strictly pre-cutoff) and EQUAL (none).
The harness never uses a full-sample statistic except in the arm explicitly labelled FULL.

Usage
-----
    python scripts/research/tailwind_sizing_null_lab.py --test all --out <dir>
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import xsec_momentum_falsification as mom  # noqa: E402

from sharpen.features import defensive_signals as dfs  # noqa: E402

logger = logging.getLogger("tailwind_sizing_null_lab")

ANN = mom.ANN
CFG = ROOT / "configs" / "tailwind_v1.yaml"
STAGE4_GATES = ROOT / "configs" / "tailwind_v1_stage4.gates.yaml"

# Controls: the plain leverage knob that every sizing arm must beat.
CONTROL_K = (0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0, 1.25, 1.5, 2.0, 3.0)


# --------------------------------------------------------------------- config
def load_cfg() -> dict:
    """Sizing constants read from configs/tailwind_v1.yaml -- never hardcoded here."""
    c = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    env, rp = c["env"], c["risk_parity"]
    gates = yaml.safe_load(STAGE4_GATES.read_text(encoding="utf-8"))["stage4_recent_oos"]
    return {
        "assets": c["universe"]["assets"],
        "asset_class": c["universe"]["asset_class"],
        "target_vol_asset": float(env["target_vol_asset"]),
        "lev_cap": float(env["lev_cap"]),
        "max_gross_exposure": float(env["max_gross_exposure"]),
        "min_trade_pct": float(env["min_trade_pct"]),
        "taker_fee": float(env["taker_fee"]),
        "slippage_base_bps": float(env["slippage_base_bps"]),
        "beta_window": int(c["sleeves"]["defensive"]["beta_window"]),
        "beta_min_periods": int(c["sleeves"]["defensive"]["min_periods"]),
        "rp_trailing_window": int(rp["trailing_window"]),
        "rp_min_periods": int(rp["min_periods"]),
        "rp_monthly_meta": bool(rp["monthly_meta"]),
        "research_cutoff": gates["research_cutoff"],
        "min_rebalances_for_power": int(gates["min_rebalances_for_power"]),
        "sharpe_ci_block_days": int(gates["sharpe_ci_block_days"]),
    }


# ----------------------------------------------------------------------- data
def build_panel(cfg: dict):
    """Close panel, returns, monthly rebalance dates and the two raw conviction streams."""
    close = mom.get_prices()
    close = close[[t for t in cfg["assets"] if t in close.columns]]
    rets = close.pct_change()
    rebal = mom.last_trading_of_period(close.index, "monthly")
    rebal = rebal[rebal >= close.index[max(mom.LOOKBACKS) + mom.SKIP + mom.VOL_WIN]]
    sig_mom = mom.tsmom_signal(close, rebal)
    conv_bab = dfs.defensive_conviction(
        close, cfg["asset_class"],
        beta_window=cfg["beta_window"], min_periods=cfg["beta_min_periods"],
    ).reindex(rebal)
    return close, rets, rebal, sig_mom, conv_bab


# -------------------------------------------------------------------- sizing
def size(conv: pd.DataFrame, rets: pd.DataFrame, rebal: pd.DatetimeIndex, *,
         mode: str = "volscale", target_vol: float = 0.10, lev_cap: float = 2.0,
         vol_win: int = 63) -> pd.DataFrame:
    """Conviction -> rebalance-date target weights.

    ``volscale``   : w = clip(conv * clip(target/vol_causal, <= lev_cap), +-lev_cap)
                     -- byte-identical to ``mom.vol_scaled_weights`` /
                     ``MultiAssetAllocatorEnv._action_to_weights``.
    ``flat``       : w = conv * lev_cap  -- conviction only, NO per-asset risk weighting.
    ``sign_flat``  : w = sign(conv) * lev_cap -- neither vol weighting nor conviction size.
    ``invvol_only``: w = sign(conv) * clip(target/vol, <= lev_cap) -- vol weighting WITHOUT
                     conviction magnitude (isolates the two multiplicative parts).

    Causality: ``rets.rolling(vol_win).std().shift(1)`` -- the vol used at ``dt`` reads
    returns <= dt-1 only (LEAK-2).
    """
    realized = rets.rolling(vol_win).std().shift(1) * np.sqrt(ANN)
    w = pd.DataFrame(0.0, index=rebal, columns=conv.columns, dtype=float)
    for dt in rebal:
        rv = realized.loc[:dt]
        if rv.empty:
            continue
        c = conv.loc[dt]
        if mode == "flat":
            w.loc[dt] = (c * lev_cap).clip(-lev_cap, lev_cap)
            continue
        if mode == "sign_flat":
            w.loc[dt] = np.sign(c) * lev_cap
            continue
        scale = (target_vol / rv.iloc[-1]).clip(upper=lev_cap)
        base = np.sign(c) if mode == "invvol_only" else c
        w.loc[dt] = (base * scale).clip(-lev_cap, lev_cap)
    return w.fillna(0.0)


def gross_cap(w: pd.DataFrame, cap: float | None) -> pd.DataFrame:
    """Proportional gross-exposure cap -- verbatim ``_enforce_gross_exposure`` semantics."""
    if cap is None or cap <= 0:
        return w
    g = w.abs().sum(axis=1)
    s = np.minimum(1.0, cap / g.replace(0.0, np.nan)).fillna(1.0)
    return w.mul(s, axis=0)


def apply_dust(w: pd.DataFrame, min_trade_pct: float) -> pd.DataFrame:
    """Stateful ``min_trade_pct`` dust filter -- deltas below the floor are NOT traded, so
    the held book drifts from target. This is the one production mechanism that makes a
    plain leverage knob non-neutral, which is why the control frontier is not flat under it.
    """
    if min_trade_pct <= 0:
        return w
    tgt = w.to_numpy()
    held = np.zeros(w.shape[1])
    out = np.empty_like(tgt)
    for i in range(len(w)):
        d = tgt[i] - held
        d[np.abs(d) < min_trade_pct] = 0.0
        held = held + d
        out[i] = held
    return pd.DataFrame(out, index=w.index, columns=w.columns)


# ------------------------------------------------------------------- sleeves
def sleeve_paths(w_rebal: pd.DataFrame, rets: pd.DataFrame, cost_rate: float):
    """(gross_ret, cost, exposure) daily series for one sleeve.

    Weights are ffilled to daily and shifted 1 bar (the certifying basis' causal lag);
    turnover cost is charged on the SAME weight series that generates the P&L.
    """
    w_daily = w_rebal.reindex(rets.index).ffill().fillna(0.0)
    w_eff = w_daily.shift(1).fillna(0.0)
    gross_ret = (w_eff * rets).sum(axis=1)
    dw = w_rebal.fillna(0.0).diff().abs().sum(axis=1)
    if len(dw):
        dw.iloc[0] = w_rebal.iloc[0].abs().sum()
    cost = dw.reindex(rets.index).fillna(0.0) * cost_rate
    return gross_ret, cost, w_eff.abs().sum(axis=1)


def rp_scalars(series: pd.Series, mode: str, cfg: dict, cutoff: pd.Timestamp) -> pd.Series:
    """10%-vol sleeve scalar under four provenance regimes.

    FULL     -- ``0.10 / ann_vol(full sample)``. This is ``portfolio_frontier.risk_parity``
                and it is a LEAK-2 defect: the scalar is a function of the whole sample,
                so on any holdout the sleeve scales itself with future information.
    TRAILING -- causal 252d rolling vol, shifted 1, held monthly (the config's own
                ``risk_parity`` block: trailing_window 252, min_periods 63, monthly_meta).
    FROZEN   -- fit on bars strictly < research_cutoff, held constant forward.
    EQUAL    -- 1.0 (no vol equalisation at all: the null for risk parity).
    """
    idx = series.index
    if mode == "EQUAL":
        return pd.Series(1.0, index=idx)
    if mode == "FULL":
        v = float(series.std() * np.sqrt(ANN))
        return pd.Series(0.10 / v if v > 0 else 1.0, index=idx)
    if mode == "FROZEN":
        pre = series[idx < cutoff]
        v = float(pre.std() * np.sqrt(ANN))
        return pd.Series(0.10 / v if v > 0 else 1.0, index=idx)
    if mode == "TRAILING":
        rv = series.rolling(cfg["rp_trailing_window"],
                            min_periods=cfg["rp_min_periods"]).std().shift(1) * np.sqrt(ANN)
        k = (0.10 / rv).replace([np.inf, -np.inf], np.nan)
        if cfg["rp_monthly_meta"]:
            month_end = mom.last_trading_of_period(idx, "monthly")
            k = k.where(idx.isin(month_end)).ffill()
        return k.clip(upper=20.0).bfill().fillna(1.0)
    raise ValueError(f"unknown rp mode {mode}")


# ----------------------------------------------------------------------- book
def build_book(panel, cfg, *, size_mode="volscale", target_vol=None, lev_cap=None,
               vol_win=63, sleeve_cap=None, combined_cap=None, rp_mode="FULL",
               bab_weight=0.5, cost_rate=0.0002, dust=0.0, lev_scale=1.0, pre_lev=1.0,
               dd_throttle=None, invert=False, want_parts=False):
    """Assemble one sizing variant; return (daily net series, daily gross exposure)."""
    _close, rets, rebal, sig_mom, conv_bab = panel
    cutoff = pd.Timestamp(cfg["research_cutoff"])
    tv = cfg["target_vol_asset"] if target_vol is None else target_vol
    lc = cfg["lev_cap"] if lev_cap is None else lev_cap
    sm, sb = (-sig_mom, -conv_bab) if invert else (sig_mom, conv_bab)

    weights = {}
    parts = []
    for tag, conv, wt in (("mom", sm, 1.0 - bab_weight), ("bab", sb, bab_weight)):
        if wt <= 0:
            parts.append(None)
            continue
        w = size(conv.fillna(0.0), rets, rebal, mode=size_mode,
                 target_vol=tv, lev_cap=lc, vol_win=vol_win)
        # `pre_lev` is a leverage knob applied UPSTREAM of the gross cap and the dust
        # filter, so it passes through the two nonlinear production mechanics. `lev_scale`
        # (below, post-combine) is the pure book-level knob. Both are needed: a
        # risk-parity combine renormalises each sleeve to 10% vol and therefore DIVIDES
        # OUT any constant pre-combine leverage, so `pre_lev` is only meaningful with
        # rp_mode="EQUAL".
        w = w * pre_lev
        weights[tag + "_precap"] = w
        w = gross_cap(w, sleeve_cap)
        w = apply_dust(w, dust)
        weights[tag] = w
        parts.append(sleeve_paths(w, rets, cost_rate))

    idx = rets.index
    tot_g = pd.Series(0.0, index=idx)
    tot_c = pd.Series(0.0, index=idx)
    tot_e = pd.Series(0.0, index=idx)
    scalars = {}
    for tag, part, wt in zip(("mom", "bab"), parts, (1.0 - bab_weight, bab_weight)):
        if part is None:
            continue
        g, c, e = part
        net = (g - c).dropna()
        k = rp_scalars(net, rp_mode, cfg, cutoff).reindex(idx).ffill().fillna(1.0)
        a = 2.0 * wt * k * lev_scale
        scalars[tag] = a
        tot_g = tot_g + a * g.fillna(0.0)
        tot_c = tot_c + a * c.fillna(0.0)
        tot_e = tot_e + a * e.fillna(0.0)

    if combined_cap is not None and combined_cap > 0:
        s = np.minimum(1.0, combined_cap / tot_e.replace(0.0, np.nan)).fillna(1.0)
        tot_g, tot_c, tot_e = tot_g * s, tot_c * s, tot_e * s

    net = tot_g - tot_c
    if dd_throttle is not None:
        net, tot_e = apply_dd_throttle(net, tot_e, *dd_throttle)

    warm_bars = max(mom.LOOKBACKS) + mom.SKIP + mom.VOL_WIN + cfg["beta_window"]
    warm = rets.index[min(warm_bars, len(rets) - 1)]
    m = net.index >= warm
    out = (net[m].dropna(), tot_e[m])
    if want_parts:
        return out + (weights, scalars)
    return out


def apply_dd_throttle(net: pd.Series, expo: pd.Series, start: float, full: float,
                      floor: float):
    """Equity-drawdown throttle -- the ONE sizing overlay in this lab that reads the book's
    own P&L, hence the only one the edge-sign flip has any power against."""
    r = net.to_numpy()
    e = expo.reindex(net.index).fillna(0.0).to_numpy()
    eq, peak = 1.0, 1.0
    out_r, out_e = np.empty_like(r), np.empty_like(e)
    for t in range(len(r)):
        dd = 1.0 - eq / peak if peak > 0 else 0.0
        s = 1.0
        if dd > start:
            s = max(floor, 1.0 - (dd - start) / max(1e-9, full - start))
        out_r[t], out_e[t] = s * r[t], s * e[t]
        eq *= 1.0 + out_r[t]
        peak = max(peak, eq)
    return pd.Series(out_r, index=net.index), pd.Series(out_e, index=net.index)


def stats(net: pd.Series, expo: pd.Series) -> dict:
    d = net.dropna()
    return {
        "PF": round(mom.pf(d), 5),
        "SR": round(mom.sharpe(d), 4),
        "MDD_pct": round(mom.max_dd(d) * 100, 2),
        "annvol_pct": round(float(d.std() * np.sqrt(ANN)) * 100, 2),
        "annret_pct": round(float(d.mean() * ANN) * 100, 3),
        "Expo": round(float(expo.reindex(d.index).median()), 4),
        "n_days": int(len(d)),
    }


# --------------------------------------------------------------------- grid
def grid(cfg: dict) -> list[tuple[str, dict]]:
    """Sizing arms. CONTROL_* is the plain leverage knob -- the null Rule 1 scores against."""
    cap = cfg["max_gross_exposure"]
    g: list[tuple[str, dict]] = [("baseline_research_basis", {})]
    g += [(f"CONTROL_lev_x{k}", dict(lev_scale=k)) for k in CONTROL_K]
    # Second control family: the same knob applied UPSTREAM of the cap + dust filter,
    # with rp normalisation off so it is not divided out. Tests whether the nonlinear
    # production mechanics break PF leverage-invariance.
    g += [(f"CONTROLP_predust_lev_x{k}", dict(pre_lev=k, rp_mode="EQUAL"))
          for k in (0.25, 0.5, 1.0, 2.0, 4.0)]
    # --- per-asset vol targeting ---
    g += [("A_no_volscale_flat", dict(size_mode="flat")),
          ("A_no_volscale_signflat", dict(size_mode="sign_flat")),
          ("A_invvol_no_conviction", dict(size_mode="invvol_only"))]
    g += [(f"A_volwin_{w}", dict(vol_win=w)) for w in (21, 42, 126, 252)]
    g += [(f"A_targetvol_{v:.2f}", dict(target_vol=v)) for v in (0.05, 0.20, 0.40)]
    g += [(f"A_levcap_{c}", dict(lev_cap=c)) for c in (0.5, 1.0, 3.0, 10.0)]
    # --- gross-exposure cap (executor-only mechanism) ---
    g += [(f"B_sleevecap_{c}", dict(sleeve_cap=c)) for c in (1.5, cap, 6.0, 12.0)]
    g += [("B_sleevecap3_combcap3", dict(sleeve_cap=cap, combined_cap=cap)),
          ("B_combcap3_only", dict(combined_cap=cap)),
          ("B_cap3_flat_sizing", dict(size_mode="flat", sleeve_cap=cap)),
          ("B_cap3_signflat_sizing", dict(size_mode="sign_flat", sleeve_cap=cap))]
    # --- sleeve combine / BAB ---
    g += [(f"C_rp_{m}", dict(rp_mode=m)) for m in ("FULL", "TRAILING", "FROZEN", "EQUAL")]
    g += [(f"C_bab_w{w}", dict(bab_weight=w)) for w in (0.25, 0.75)]
    g += [("C_momentum_only", dict(bab_weight=0.0)),
          ("C_bab_only", dict(bab_weight=1.0))]
    # --- state-dependent overlays (flip-test positive controls) ---
    g += [(f"D_ddthrottle_{s}-{f}_fl{fl}", dict(dd_throttle=(s, f, fl)))
          for s, f, fl in ((0.05, 0.15, 0.0), (0.03, 0.10, 0.25), (0.10, 0.25, 0.5))]
    return g


def run_grid(panel, cfg, arms, **base) -> tuple[pd.DataFrame, dict]:
    rows, series = [], {}
    for name, ov in arms:
        p = dict(base)
        p.update(ov)
        net, expo = build_book(panel, cfg, **p)
        series[name] = net
        r = stats(net, expo)
        r["arm"] = name
        rows.append(r)
    return pd.DataFrame(rows).set_index("arm"), series


def frontier_excess(df: pd.DataFrame) -> pd.DataFrame:
    """PF excess over what a plain leverage knob delivers AT THE SAME median exposure."""
    # ONLY the CONTROL_lev family defines the null frontier. CONTROLP_ is a separate book
    # (rp_mode=EQUAL) and is a leverage-INVARIANCE diagnostic, not a matched control.
    ctrl = df[df.index.str.startswith("CONTROL_lev")].sort_values("Expo")
    a = df.copy()
    a["PF_ctrl"] = np.interp(a["Expo"], ctrl["Expo"], ctrl["PF"]).round(5)
    a["PF_excess"] = (a["PF"] - a["PF_ctrl"]).round(5)
    a["SR_excess"] = (a["SR"] - np.interp(a["Expo"], ctrl["Expo"], ctrl["SR"])).round(4)
    a["MDD_exc_pp"] = (a["MDD_pct"] - np.interp(a["Expo"], ctrl["Expo"],
                                                ctrl["MDD_pct"])).round(2)
    return a


def block_bootstrap_dpf(a: pd.Series, b: pd.Series, block: int = 21,
                        n: int = 2000, seed: int = 20260826) -> tuple:
    """CI95 on PF(a) - PF(b) by circular block bootstrap on the PAIRED daily series."""
    idx = a.dropna().index.intersection(b.dropna().index)
    x, y = a.reindex(idx).to_numpy(), b.reindex(idx).to_numpy()
    m = len(x)
    if m < 3 * block:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(m / block))
    out = np.empty(n)
    for i in range(n):
        starts = rng.integers(0, m, nb)
        sel = (starts[:, None] + np.arange(block)[None, :]).ravel()[:m] % m
        xs, ys = x[sel], y[sel]
        px = xs[xs > 0].sum() / max(1e-12, -xs[xs < 0].sum())
        py = ys[ys > 0].sum() / max(1e-12, -ys[ys < 0].sum())
        out[i] = px - py
    return (round(float(np.percentile(out, 2.5)), 4),
            round(float(np.percentile(out, 97.5)), 4))


# ---------------------------------------------------------------------- H1
def cmd_h1(panel, cfg) -> dict:
    """Hypothesis 1: with the gross cap binding ~always, is vol targeting operative?

    Splits the question the 2026-08-18 Tier-2 did not: a proportional cap destroys the
    TIME-SERIES (de-levering) channel but leaves the CROSS-SECTIONAL (relative risk
    weighting) channel intact. Both are measured.
    """
    _close, rets, _rebal, _sm, _sb = panel
    cap = cfg["max_gross_exposure"]
    out: dict = {"max_gross_exposure": cap}

    _n, _e, w, _s = build_book(panel, cfg, sleeve_cap=None, want_parts=True)
    for tag in ("mom", "bab"):
        pre = w[tag + "_precap"].abs().sum(axis=1)
        out[f"{tag}_precap_gross"] = {
            "mean": round(float(pre.mean()), 3), "median": round(float(pre.median()), 3),
            "min": round(float(pre.min()), 3), "max": round(float(pre.max()), 3),
            "frac_above_cap": round(float((pre > cap).mean()), 4),
            "n_rebalances": int(len(pre)),
        }

    # TIME-SERIES channel: does book gross exposure respond to market vol, capped vs not?
    panel_vol = rets.mean(axis=1).rolling(63).std().shift(1) * np.sqrt(ANN)
    for label, sc in (("uncapped", None), (f"capped_{cap}", cap)):
        _net, expo = build_book(panel, cfg, sleeve_cap=sc)
        v = panel_vol.reindex(expo.index)
        m = v.notna() & expo.notna()
        out[f"ts_channel_{label}"] = {
            "corr_expo_vs_panel_vol": round(float(np.corrcoef(expo[m], v[m])[0, 1]), 4),
            "expo_cv": round(float(expo[m].std() / expo[m].mean()), 4),
            "expo_p05": round(float(expo[m].quantile(0.05)), 4),
            "expo_p95": round(float(expo[m].quantile(0.95)), 4),
        }

    # CROSS-SECTIONAL channel: at the SAME capped gross, does vol weighting still matter?
    for label, mode in (("volscale", "volscale"), ("flat", "flat"), ("signflat", "sign_flat")):
        net, expo = build_book(panel, cfg, size_mode=mode, sleeve_cap=cap)
        out[f"xs_channel_cap{cap}_{label}"] = stats(net, expo)
    return out


# ---------------------------------------------------------------------- CLI
def cmd_flip(panel, cfg, arms, cost_rate: float) -> pd.DataFrame:
    real, s_real = run_grid(panel, cfg, arms, cost_rate=cost_rate, invert=False)
    inv, s_inv = run_grid(panel, cfg, arms, cost_rate=cost_rate, invert=True)
    fr, fi = frontier_excess(real), frontier_excess(inv)
    rows = []
    for k in fr.index:
        if k.startswith("CONTROL") or k == "baseline_research_basis":
            continue
        pf_r, pf_i = fr.loc[k, "PF_excess"], fi.loc[k, "PF_excess"]
        rows.append(dict(
            arm=k, PF_exc_REAL=pf_r, PF_exc_INVERTED=pf_i,
            PF_real=fr.loc[k, "PF"], PF_inv=fi.loc[k, "PF"],
            recip_resid=round(float(fr.loc[k, "PF"] * fi.loc[k, "PF"] - 1.0), 6),
            verdict=("HELPS BOTH" if pf_r > 0 and pf_i > 0 else
                     "REVERSES (de-risking artifact)" if pf_r > 0 else
                     "helps loser only" if pf_i > 0 else "harmful in both"),
        ))
    _ = (s_real, s_inv)
    return pd.DataFrame(rows).sort_values("PF_exc_REAL", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", choices=["grid", "flip", "h1", "all"], default="all")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--price-cache", type=Path, default=None,
                    help="override xsec_momentum_falsification.CACHE (worktrees have no "
                         "results/ tree; point this at the main checkout's frozen cache)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    if a.price_cache is not None:
        if not a.price_cache.exists():
            raise SystemExit(f"price cache not found: {a.price_cache}")
        mom.CACHE = a.price_cache

    cfg = load_cfg()
    panel = build_panel(cfg)
    arms = grid(cfg)
    art: dict = {"config": {k: v for k, v in cfg.items() if k != "asset_class"}}

    if a.test in ("h1", "all"):
        h1 = cmd_h1(panel, cfg)
        art["h1_gross_cap"] = h1
        print("\n=== H1: is vol targeting operative under the gross cap? ===")
        print(json.dumps(h1, indent=2))

    if a.test in ("grid", "all"):
        for label, rate, dust in (("research_2bps", 0.0002, 0.0),
                                  ("executor_3bps_dust", 0.0003, cfg["min_trade_pct"])):
            df, series = run_grid(panel, cfg, arms, cost_rate=rate, dust=dust)
            f = frontier_excess(df)
            art[f"grid_{label}"] = json.loads(f.reset_index().to_json(orient="records"))
            print(f"\n=== RULE 1 -- PF excess over matched-exposure de-levering "
                  f"[{label}] ===")
            print(f[["PF", "SR", "Expo", "PF_ctrl", "PF_excess", "SR_excess",
                     "MDD_pct", "MDD_exc_pp"]].to_string())
            base = series["baseline_research_basis"]
            ci = {k: block_bootstrap_dpf(series[k], base,
                                         block=cfg["sharpe_ci_block_days"])
                  for k in df.index if not k.startswith("CONTROL")}
            art[f"ci95_dPF_vs_baseline_{label}"] = ci
            print(f"\n--- CI95 on PF(arm) - PF(baseline), {cfg['sharpe_ci_block_days']}d "
                  f"block bootstrap [{label}] ---")
            for k, v in ci.items():
                print(f"  {k:34s} [{v[0]:+.4f}, {v[1]:+.4f}]")

    if a.test in ("flip", "all"):
        fl = cmd_flip(panel, cfg, arms, 0.0002)
        art["flip_research_2bps"] = json.loads(fl.to_json(orient="records"))
        print("\n=== RULE 2 -- EDGE-SIGN FLIP (2 bps both arms) ===")
        print(fl.to_string(index=False))
        flf = cmd_flip(panel, cfg, arms, 0.0)
        art["flip_frictionless"] = json.loads(flf.to_json(orient="records"))
        print("\n=== RULE 2 -- EDGE-SIGN FLIP (frictionless; recip_resid ~ 0 proves the "
              "arm is sign-INVARIANT and the flip test has NO power against it) ===")
        print(flf.to_string(index=False))

    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "tailwind_sizing_null_lab.json").write_text(
            json.dumps(art, indent=2, default=str), encoding="utf-8")
        logger.warning("wrote %s", a.out / "tailwind_sizing_null_lab.json")


if __name__ == "__main__":
    main()
