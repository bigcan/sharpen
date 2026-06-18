"""PRISM regime-detector eval — how GOOD / how USEFUL are the price & vol regimes.

Turns the PRISM falsification lens from the Chronos forecast head (cont-49/50/51,
all no-edge) onto the GAHMM regime detectors. cont-49 already showed regime
DIRECTION is inside the shuffle-null; R0-regime (cont-30) killed gold directional
GATING. The untested gap, and where a regime detector is most likely to pay off:

  * Stage 1 (how GOOD): does `vol_regime` forecast forward REALIZED VOL (not
    return)? Vol clusters, so this is the detector's natural home. Head-to-head
    against a 25-bar rolling-RV baseline built from the HMM's own input — the HMM
    must BEAT the rolling std, not tie it. Price regime re-confirmed directionally.
  * Stage 2 (how USEFUL): use `vol_regime` to SIZE a constant-long position
    (de-risk in high vol) and ask whether net-of-cost risk-adjusted return
    improves vs unscaled AND vs the rolling-RV scaler.

Causality: regime feature at bar t uses only data <= t (Path-A guarantee, asserted
by tests/prism_research/test_pathA_walk_forward.py). Forward realized vol / return
use only data > t. Trailing-RV baseline uses only data <= t. No look-ahead; the new
target/signal builders are guarded by tests/prism_research/test_regime_eval_causality.py.

ALL thresholds come from configs/prism_regime_eval.gates.yaml (never hardcoded).
Spec + pre-registration: docs/research/prism_regime_eval_spec_2026-06-18.md

Usage:
    python scripts/prism_research/prism_regime_eval.py
    python scripts/prism_research/prism_regime_eval.py --assets gold --cost-bps 3
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
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the EXACT metric/sim functions the Chronos eval used — identical bar.
from scripts.prism_research.prism_predictive_eval import (  # noqa: E402
    _rank_ic, _shuffle_null_ic, _trade_sim, RESULTS_DIR,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("prism_research.regime_eval")

GATES_PATH = PROJECT_ROOT / "configs" / "prism_regime_eval.gates.yaml"
OUT_DIR = RESULTS_DIR / "regime_eval"
_SIZE_KEY = {"low": 0, "normal": 1, "high": 2}   # hard vol_regime label codes


# --------------------------------------------------------------------------- #
# Causal feature builders (guarded by test_regime_eval_causality.py)
# --------------------------------------------------------------------------- #
def log_returns(close: np.ndarray) -> np.ndarray:
    """r_t = log(close_t / close_{t-1}); r_0 = nan."""
    close = np.asarray(close, dtype=float)
    r = np.full_like(close, np.nan)
    r[1:] = np.log(close[1:] / close[:-1])
    return r


def fwd_realized_vol(r: np.ndarray, k: int) -> np.ndarray:
    """Forward realized vol at t = sqrt(sum_{i=1..k} r_{t+i}^2). Strictly future.

    k=1 reduces to |r_{t+1}|. Last k bars have nan target (no future yet).
    """
    r = np.asarray(r, dtype=float)
    n = len(r)
    out = np.full(n, np.nan)
    sq = r ** 2
    for t in range(n - k):
        win = sq[t + 1: t + 1 + k]
        if np.all(np.isfinite(win)):
            out[t] = np.sqrt(win.sum())
    return out


def trailing_realized_vol(r: np.ndarray, w: int) -> np.ndarray:
    """Trailing realized vol at t = sqrt(mean(r_{t-w+1..t}^2)). Causal (uses <= t)."""
    s = pd.Series(r, dtype=float)
    return np.sqrt((s ** 2).rolling(w, min_periods=w).mean()).to_numpy()


def fwd_log_return(close: np.ndarray) -> np.ndarray:
    """Realized t -> t+1 log return, aligned at t (position chosen at close t)."""
    c = pd.Series(np.asarray(close, dtype=float))
    return np.log(c.shift(-1) / c).to_numpy()


def momentum(close: np.ndarray, w: int) -> np.ndarray:
    """Trailing w-bar log return at t = log(close_t / close_{t-w}). Causal."""
    c = pd.Series(np.asarray(close, dtype=float))
    return np.log(c / c.shift(w)).to_numpy()


def _terciles(x_train: np.ndarray) -> tuple[float, float]:
    v = x_train[np.isfinite(x_train)]
    if v.size < 6:
        return float("nan"), float("nan")
    return float(np.quantile(v, 1 / 3)), float(np.quantile(v, 2 / 3))


def _bucket(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Bucket into {0,1,2} by frozen tercile breakpoints (nan -> nan)."""
    out = np.full(len(x), np.nan)
    fin = np.isfinite(x)
    out[fin & (x <= lo)] = 0
    out[fin & (x > lo) & (x <= hi)] = 1
    out[fin & (x > hi)] = 2
    return out


def _dwell_stats(labels: np.ndarray) -> dict:
    """State fractions + mean run-length (dwell) of a hard label sequence."""
    lab = labels[np.isfinite(labels)].astype(int)
    n = len(lab)
    if n == 0:
        return {"state_frac": {}, "mean_dwell": float("nan"), "min_state_frac": float("nan")}
    frac = {int(s): float((lab == s).mean()) for s in np.unique(lab)}
    # run-length encode
    runs = 1 + np.sum(lab[1:] != lab[:-1])
    mean_dwell = n / runs
    return {"state_frac": frac, "mean_dwell": float(mean_dwell),
            "min_state_frac": float(min(frac.values()))}


# --------------------------------------------------------------------------- #
# Per-asset evaluation
# --------------------------------------------------------------------------- #
def _split_masks(idx: pd.DatetimeIndex, train_end: str) -> tuple[np.ndarray, np.ndarray]:
    te = pd.Timestamp(train_end)
    train = np.asarray(idx <= te)
    test = np.asarray(idx > te)
    return train, test


def _vol_stage1(vol_score, trail_rv, fwd_rv, mask, k: int) -> dict:
    """IC(vol_score, fwd_rv) vs null, vs rolling-RV baseline, on a window mask."""
    vs, tr, fv = vol_score[mask], trail_rv[mask], fwd_rv[mask]
    ic_g, p_g = _rank_ic(vs, fv)
    null_g = _shuffle_null_ic(vs, fv)
    ic_b, p_b = _rank_ic(tr, fv)
    return {
        f"vol_ic_k{k}": ic_g, f"vol_ic_p_k{k}": p_g, f"vol_ic_null95_k{k}": null_g,
        f"vol_base_ic_k{k}": ic_b, f"vol_base_ic_p_k{k}": p_b,
        f"vol_delta_ic_k{k}": (ic_g - ic_b) if np.isfinite(ic_g) and np.isfinite(ic_b) else float("nan"),
        f"vol_ic_beats_null_k{k}": bool(np.isfinite(ic_g) and np.isfinite(null_g) and abs(ic_g) > null_g),
    }


def _vol_kruskal(fwd_rv, vol_regime, mask, k: int) -> dict:
    fv, vr = fwd_rv[mask], vol_regime[mask]
    df = pd.DataFrame({"rv": fv, "state": vr}).dropna()
    groups = [g["rv"].to_numpy() for _, g in df.groupby("state") if len(g) >= 5]
    means = df.groupby("state")["rv"].mean().to_dict()
    monotone = (len(means) == 3
                and means.get(0, np.inf) <= means.get(1, np.inf) <= means.get(2, -np.inf))
    out = {f"vol_kruskal_monotone_k{k}": bool(monotone),
           f"vol_state_mean_rv_k{k}": {int(s): float(m) for s, m in means.items()}}
    if len(groups) >= 2:
        try:
            h, p = stats.kruskal(*groups)
            out[f"vol_kruskal_h_k{k}"] = float(h)
            out[f"vol_kruskal_p_k{k}"] = float(p)
        except ValueError:
            pass
    return out


def _riskscale(vol_regime, trail_rv, fwd_ret, train_mask, mask, size_map, cost) -> dict:
    """Compare unscaled vs GAHMM-vol-scaled vs rolling-RV-scaled constant-long."""
    # frozen rolling-RV tercile breakpoints from TRAIN
    lo, hi = _terciles(trail_rv[train_mask])
    base_state = _bucket(trail_rv, lo, hi)
    sizes = np.array([size_map["low"], size_map["normal"], size_map["high"]])

    def _pos_from_state(state: np.ndarray) -> np.ndarray:
        p = np.full(len(state), np.nan)
        fin = np.isfinite(state)
        p[fin] = sizes[state[fin].astype(int)]
        return p

    pos_unscaled = np.ones(len(vol_regime))
    pos_gahmm = _pos_from_state(vol_regime)
    pos_base = _pos_from_state(base_state)

    fr = fwd_ret[mask]
    sim_u = _trade_sim(pos_unscaled[mask], fr, cost)
    sim_g = _trade_sim(pos_gahmm[mask], fr, cost)
    sim_b = _trade_sim(pos_base[mask], fr, cost)

    dd_u, dd_g = abs(sim_u["max_dd"]), abs(sim_g["max_dd"])
    return {
        "rs_sharpe_unscaled": sim_u["net_sharpe"], "rs_sharpe_gahmm": sim_g["net_sharpe"],
        "rs_sharpe_base": sim_b["net_sharpe"],
        "rs_dd_unscaled": sim_u["max_dd"], "rs_dd_gahmm": sim_g["max_dd"],
        "rs_dd_base": sim_b["max_dd"],
        "rs_ret_unscaled": sim_u["total_return"], "rs_ret_gahmm": sim_g["total_return"],
        "rs_sharpe_uplift": sim_g["net_sharpe"] - sim_u["net_sharpe"],
        "rs_marginal_vs_base": sim_g["net_sharpe"] - sim_b["net_sharpe"],
        "rs_dd_reduction_frac": (dd_u - dd_g) / dd_u if dd_u > 1e-9 else float("nan"),
    }


def _price_stage1(price_score, mom20, price_regime, fwd_ret, mask, cost) -> dict:
    ps, mm, pr, fr = price_score[mask], mom20[mask], price_regime[mask], fwd_ret[mask]
    ic_g, p_g = _rank_ic(ps, fr)
    null_g = _shuffle_null_ic(ps, fr)
    ic_b, _ = _rank_ic(mm, fr)
    pos = (pr == 2).astype(float)   # bull-gated long
    sim = _trade_sim(pos, fr, cost)
    return {
        "price_ic": ic_g, "price_ic_p": p_g, "price_ic_null95": null_g,
        "price_base_ic_mom20": ic_b,
        "price_ic_beats_null": bool(np.isfinite(ic_g) and np.isfinite(null_g) and abs(ic_g) > null_g),
        "price_beats_mom20": bool(np.isfinite(ic_g) and np.isfinite(ic_b) and abs(ic_g) > abs(ic_b)),
        "price_gated_net_pf": sim["net_pf"], "price_gated_net_sharpe": sim["net_sharpe"],
        "price_gated_mdd": sim["max_dd"],
    }


def eval_asset(label: str, df: pd.DataFrame, cfg: dict) -> dict:
    df = df.sort_index().copy()
    if not isinstance(df.index, pd.DatetimeIndex) and "date" in df.columns:
        df = df.set_index(pd.DatetimeIndex(pd.to_datetime(df["date"])))
    run = cfg["run"]
    cost = run["cost_bps_oneway"] / 1e4
    w = run["trail_vol_window"]
    train_mask, test_mask = _split_masks(df.index, run["train_end"])
    full_mask = np.ones(len(df), dtype=bool)

    close = df["close"].to_numpy(dtype=float)
    r = log_returns(close)
    trail_rv = trailing_realized_vol(r, w)
    fwd_ret = fwd_log_return(close)
    vol_score = (df["vol_high_prob"] - df["vol_low_prob"]).to_numpy()
    price_score = (df["price_bull_prob"] - df["price_bear_prob"]).to_numpy()
    vol_regime = df["vol_regime"].to_numpy(dtype=float)
    price_regime = df["price_regime"].to_numpy(dtype=float)
    mom20 = momentum(close, 20)

    n_tr, n_te = int(train_mask.sum()), int(test_mask.sum())
    logger.info("\n%s  [%d bars %s..%s | train %d test %d (split %s)]",
                label.upper(), len(df), df.index[0].date(), df.index[-1].date(),
                n_tr, n_te, run["train_end"])

    # --- label sanity (degeneracy guard) ---
    vol_dwell = _dwell_stats(vol_regime)
    price_dwell = _dwell_stats(price_regime)
    vol_ok = (vol_dwell["min_state_frac"] >= cfg["sanity"]["min_state_frac"]
              and vol_dwell["mean_dwell"] >= cfg["sanity"]["min_mean_dwell_bars"])
    price_ok = (price_dwell["min_state_frac"] >= cfg["sanity"]["min_state_frac"]
                and price_dwell["mean_dwell"] >= cfg["sanity"]["min_mean_dwell_bars"])
    logger.info("  vol_regime  states=%s dwell=%.1f  -> %s",
                {k: round(v, 2) for k, v in vol_dwell["state_frac"].items()},
                vol_dwell["mean_dwell"], "OK" if vol_ok else "DEGENERATE")
    logger.info("  price_regime states=%s dwell=%.1f -> %s",
                {k: round(v, 2) for k, v in price_dwell["state_frac"].items()},
                price_dwell["mean_dwell"], "OK" if price_ok else "DEGENERATE")

    row: dict = {"asset": label, "n_bars": len(df), "n_train": n_tr, "n_test": n_te,
                 "vol_label_ok": vol_ok, "price_label_ok": price_ok,
                 "vol_state_frac": vol_dwell["state_frac"], "vol_mean_dwell": vol_dwell["mean_dwell"],
                 "price_state_frac": price_dwell["state_frac"], "price_mean_dwell": price_dwell["mean_dwell"]}

    for tag, mask in (("full", full_mask), ("oos", test_mask)):
        sub: dict = {}
        # Stage 1A: vol -> forward realized vol, each horizon
        for k in run["horizons_days"]:
            fwd_rv = fwd_realized_vol(r, k)
            sub.update(_vol_stage1(vol_score, trail_rv, fwd_rv, mask, k))
            sub.update(_vol_kruskal(fwd_rv, vol_regime, mask, k))
        # Stage 1B: price -> forward return
        sub.update(_price_stage1(price_score, mom20, price_regime, fwd_ret, mask, cost))
        # Stage 2: vol risk-scaling
        sub.update(_riskscale(vol_regime, trail_rv, fwd_ret, train_mask, mask,
                              run["vol_size_map"], cost))
        row.update({f"{tag}__{kk}": vv for kk, vv in sub.items()})

        kp = run["primary_horizon"]
        logger.info("  [%s] vol IC(k%d)=%+.3f null=%.3f baseIC=%+.3f dIC=%+.3f | "
                    "risk-scale Sharpe unscaled=%+.2f gahmm=%+.2f base=%+.2f (uplift %+.2f, vs-base %+.2f) "
                    "DD %.0f%%->%.0f%%",
                    tag, kp, sub[f"vol_ic_k{kp}"], sub[f"vol_ic_null95_k{kp}"],
                    sub[f"vol_base_ic_k{kp}"], sub[f"vol_delta_ic_k{kp}"],
                    sub["rs_sharpe_unscaled"], sub["rs_sharpe_gahmm"], sub["rs_sharpe_base"],
                    sub["rs_sharpe_uplift"], sub["rs_marginal_vs_base"],
                    100 * sub["rs_dd_unscaled"], 100 * sub["rs_dd_gahmm"])
    return row


# --------------------------------------------------------------------------- #
# Verdict
# --------------------------------------------------------------------------- #
def _vol_useful_for_asset(row: dict, cfg: dict) -> bool:
    """OOS vol-regime usefulness, primary horizon. Degenerate label -> excluded."""
    if not row.get("vol_label_ok", False):
        return False
    g = cfg["vol_regime_useful"]
    kp = cfg["run"]["primary_horizon"]
    p = "oos__"
    ic_ok = (not g["ic_must_exceed_null95"]) or row.get(f"{p}vol_ic_beats_null_k{kp}", False)
    dic = row.get(f"{p}vol_delta_ic_k{kp}", float("nan"))
    dic_ok = np.isfinite(dic) and dic > g["min_delta_ic_vs_baseline"]
    uplift = row.get(f"{p}rs_sharpe_uplift", float("nan"))
    ddred = row.get(f"{p}rs_dd_reduction_frac", float("nan"))
    risk_ok = (np.isfinite(uplift) and uplift >= g["min_sharpe_uplift_vs_unscaled"]) or \
              (np.isfinite(ddred) and ddred >= g["min_dd_reduction_frac"])
    marg = row.get(f"{p}rs_marginal_vs_base", float("nan"))
    marg_ok = np.isfinite(marg) and marg > g["min_marginal_sharpe_vs_baseline"]
    return bool(ic_ok and dic_ok and risk_ok and marg_ok)


def _price_useful_for_asset(row: dict, cfg: dict) -> bool:
    if not row.get("price_label_ok", False):
        return False
    g = cfg["price_regime_useful"]
    p = "oos__"
    ic_ok = (not g["ic_must_exceed_null95"]) or row.get(f"{p}price_ic_beats_null", False)
    pf_ok = row.get(f"{p}price_gated_net_pf", 0.0) > g["min_gated_net_pf"]
    mom_ok = (not g["must_beat_mom20_baseline"]) or row.get(f"{p}price_beats_mom20", False)
    return bool(ic_ok and pf_ok and mom_ok)


def build_verdict(rows: list[dict], cfg: dict) -> dict:
    vol_pass = {r["asset"]: _vol_useful_for_asset(r, cfg) for r in rows}
    price_pass = {r["asset"]: _price_useful_for_asset(r, cfg) for r in rows}
    vol_useful = sum(vol_pass.values()) >= cfg["vol_regime_useful"]["min_assets_pass"]
    price_useful = sum(price_pass.values()) >= cfg["price_regime_useful"]["min_assets_pass"]
    if vol_useful or price_useful:
        headline = []
        if vol_useful:
            headline.append("VOL-REGIME USEFUL (risk-scaling)")
        if price_useful:
            headline.append("PRICE-REGIME USEFUL (directional)")
        verdict = " + ".join(headline)
    else:
        verdict = "FALSIFIED — PRISM regime detectors not useful at daily granularity; document, do not deploy"
    return {"verdict": verdict, "vol_regime_useful": vol_useful,
            "price_regime_useful": price_useful, "vol_pass_by_asset": vol_pass,
            "price_pass_by_asset": price_pass}


def main() -> None:
    ap = argparse.ArgumentParser(description="PRISM regime-detector eval (price + vol)")
    ap.add_argument("--assets", nargs="+", default=None,
                    help="override the asset list from the gates file")
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--cost-bps", type=float, default=None,
                    help="override one-way cost (bps) from the gates file")
    ap.add_argument("--gates", default=str(GATES_PATH))
    args = ap.parse_args()

    with open(args.gates, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.assets:
        cfg["run"]["assets"] = args.assets
    if args.cost_bps is not None:
        cfg["run"]["cost_bps_oneway"] = args.cost_bps

    results_dir = Path(args.results_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for label in cfg["run"]["assets"]:
        path = results_dir / f"prism_features_{label}_daily.parquet"
        if not path.exists():
            logger.error("Missing %s — run precompute_prism_pathA.py first.", path)
            continue
        rows.append(eval_asset(label, pd.read_parquet(path), cfg))

    if not rows:
        logger.error("No assets evaluated.")
        sys.exit(1)

    pd.DataFrame(rows).to_csv(OUT_DIR / "regime_eval_summary.csv", index=False)
    verdict = build_verdict(rows, cfg)
    out = {"spec": "docs/research/prism_regime_eval_spec_2026-06-18.md",
           "gates": str(Path(args.gates).name), "run_params": cfg["run"],
           "verdict": verdict, "per_asset": rows}
    with open(OUT_DIR / "verdict.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)

    kp = cfg["run"]["primary_horizon"]
    logger.info("\n%s\nHEADLINE — PRISM regime usefulness (OOS, net %.0fbps, primary horizon k=%d)\n%s",
                "=" * 72, cfg["run"]["cost_bps_oneway"], kp, "=" * 72)
    for r in rows:
        logger.info("  %-7s: vol %-9s (dIC=%+.3f, Sharpe-uplift %+.2f, vs-base %+.2f) | price %-9s (netPF=%.2f)",
                    r["asset"].upper(),
                    "USEFUL" if verdict["vol_pass_by_asset"][r["asset"]] else "no-edge",
                    r.get(f"oos__vol_delta_ic_k{kp}", float("nan")),
                    r.get("oos__rs_sharpe_uplift", float("nan")),
                    r.get("oos__rs_marginal_vs_base", float("nan")),
                    "USEFUL" if verdict["price_pass_by_asset"][r["asset"]] else "no-edge",
                    r.get("oos__price_gated_net_pf", float("nan")))
    logger.info("\n  VERDICT: %s", verdict["verdict"])
    logger.info("  Saved -> %s", OUT_DIR / "verdict.json")


if __name__ == "__main__":
    main()
