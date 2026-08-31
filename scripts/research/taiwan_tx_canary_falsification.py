"""Taiwan TX (TAIEX futures) gmgp1/sg-1 falsification canary — frictionless directional edge?

The CHEAP gate before any GPU (S553-cont-85 less-efficient-market thesis): does a LINEAR core
on the gmgp1 env's OWN multiscale features (the exact 8-dim/scale set the SAC agent would see)
have any frictionless OOS directional edge on TX, beyond a block-shuffle null? Project doctrine
= "RL must beat the linear core"; the gmgp1-btc CLEAN-CANARY and gold both FALSIFIED here (best
frictionless edge inside the shuffle-null). If TX also shows nothing, RL is unjustified — stop
before HPO. If linear shows structure, GO to the RL canary (minimal SAC WF), then HPO.

Method (no RL training; CPU, minutes):
  1. DATA-PREP: load TX_1min via the real `MultiScaleOHLCVHandler` (scales 15/60/240, the X2
     causal coarse-bar map), so the features are bit-faithful to what the env serves.
  2. For each base (15-min) bar, the agent's current multiscale state = the last-CLOSED feature
     row per scale (via `_scale_index_map`) → 3×8 = 24-dim X_t. Target y_t = next-bar log return.
  3. Rolling walk-forward Ridge (re-fit per fold, train-only standardization, embargo purge) →
     OOS predictions only.
  4. OOS metrics: rank-IC(pred, realized) + frictionless directional PF (pos=sign(pred)).
  5. Block-shuffle NULL: circular block-bootstrap the realized series (preserves intraday
     autocorrelation), recompute IC/PF vs the FIXED predictions → null distribution + p-values.
  6. VERDICT from `configs/taiwan_tx_canary.gates.yaml` (no hardcoded thresholds): GO only if OOS
     IC and PF both beat the null AND clear absolute floors; else FALSIFIED (BTC/gold class).

Usage:
    python scripts/research/taiwan_tx_canary_falsification.py \
        --parquet data/taiwan_intraday/TX_1min.parquet \
        --gates configs/taiwan_tx_canary.gates.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from sharpen.data.multiscale_handler import MultiScaleOHLCVHandler

ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger("tx_canary")
RNG = np.random.RandomState(20260628)


def pf(pnl: np.ndarray) -> float:
    up = pnl[pnl > 0].sum()
    dn = -pnl[pnl < 0].sum()
    return float(up / dn) if dn > 0 else float("inf")


def build_state_matrix(h: MultiScaleOHLCVHandler) -> tuple[np.ndarray, np.ndarray]:
    """Per base bar t: stack the last-CLOSED feature row of each scale → (n, 24); target =
    next-bar base log return. Uses the handler's own causal `_scale_index_map` (X2-safe)."""
    base = h._base_scale
    cols = []
    for scale in h.scales:
        feats = h._scale_features[scale]                      # (n_scale, 8)
        idx = h._scale_index_map[scale]                       # base-bar → last-closed coarse idx
        cols.append(feats[idx])                               # (n_base, 8), causal-aligned
    X = np.concatenate(cols, axis=1).astype(np.float64)       # (n_base, 24)
    close = h._base_close.astype(np.float64)
    y = np.full(len(close), np.nan)
    y[:-1] = np.log(np.where(close[1:] > 0, close[1:], np.nan)
                    / np.where(close[:-1] > 0, close[:-1], np.nan))  # next-bar return
    # Drop the warmup window + final NaN-target bar.
    lo = h.window_size
    valid = np.isfinite(y) & np.isfinite(X).all(axis=1)
    valid[:lo] = False
    return X[valid], y[valid], base


def walk_forward_oos(X: np.ndarray, y: np.ndarray, folds: int, alpha: float,
                     embargo: int) -> tuple[np.ndarray, np.ndarray]:
    """Rolling WF Ridge → concatenated OOS (pred, realized). Train-only standardization;
    an embargo gap purges train→test boundary leakage."""
    n = len(X)
    bounds = np.linspace(0, n, folds + 1).astype(int)
    preds, reals = [], []
    for k in range(1, folds):                                 # fold 0 is the first train block
        tr_end = bounds[k]
        te_lo, te_hi = bounds[k], bounds[k + 1]
        tr_hi = max(0, tr_end - embargo)
        if tr_hi < 200 or te_hi - te_lo < 50:
            continue
        Xtr, ytr = X[:tr_hi], y[:tr_hi]
        Xte, yte = X[te_lo:te_hi], y[te_lo:te_hi]
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd = np.where(sd < 1e-9, 1.0, sd)
        m = Ridge(alpha=alpha).fit((Xtr - mu) / sd, ytr)
        preds.append(m.predict((Xte - mu) / sd))
        reals.append(yte)
    return np.concatenate(preds), np.concatenate(reals)


def block_shuffle_null(pred: np.ndarray, real: np.ndarray, n_boot: int, block: int) -> dict:
    """Circular block-bootstrap the realized series vs FIXED preds → null IC/PF dist + p-values."""
    n = len(real)
    n_blocks = int(np.ceil(n / block))
    obs_ic = spearmanr(pred, real).correlation
    obs_pf = pf(np.sign(pred) * real)
    null_ic = np.empty(n_boot)
    null_pf = np.empty(n_boot)
    for i in range(n_boot):
        starts = RNG.randint(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).reshape(-1)[:n] % n
        r = real[idx]
        null_ic[i] = spearmanr(pred, r).correlation
        null_pf[i] = pf(np.sign(pred) * r)
    return {
        "obs_ic": float(obs_ic), "obs_pf": float(obs_pf),
        "ic_null_p": float((null_ic >= obs_ic).mean()),
        "pf_null_p": float((null_pf >= obs_pf).mean()),
        "ic_null_mean": float(np.nanmean(null_ic)), "pf_null_mean": float(np.nanmean(null_pf)),
        "ic_null_p95": float(np.nanpercentile(null_ic, 95)),
        "pf_null_p95": float(np.nanpercentile(null_pf, 95)),
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="TX gmgp1/sg-1 frictionless falsification canary")
    ap.add_argument("--parquet", default="data/taiwan_intraday/TX_1min.parquet")
    ap.add_argument("--gates", default="configs/taiwan_tx_canary.gates.yaml")
    ap.add_argument("--out", default="results/taiwan_tx_canary")
    ap.add_argument("--scales", default=None,
                    help="override gates scales, e.g. 3,15,60 for the sg-1 (3-min) base")
    args = ap.parse_args()

    g = yaml.safe_load(Path((ROOT / args.gates) if not Path(args.gates).is_absolute()
                            else Path(args.gates)).read_text())
    c, gate, nm = g["canary"], g["gate"], g["null_model"]
    if args.scales:
        c["scales"] = [int(s) for s in args.scales.split(",")]
    parquet = (ROOT / args.parquet) if not Path(args.parquet).is_absolute() else Path(args.parquet)
    out_dir = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- DATA-PREP: faithful env features via the real handler ---
    log.info("Loading TX via MultiScaleOHLCVHandler (scales=%s)...", c["scales"])
    h = MultiScaleOHLCVHandler(
        file_path=str(parquet), ticker="TX",
        feature_config={"scales": c["scales"], "window_size": c["window_size"]})
    X, y, base = build_state_matrix(h)
    ppy = 252 * (1440 / base) * (19 / 24)      # ~19h TX session → bars/yr at the base scale
    log.info("State matrix: %d base(%dmin) bars × %d features", len(X), base, X.shape[1])

    # --- LINEAR CORE walk-forward OOS ---
    pred, real = walk_forward_oos(X, y, c["wf_folds"], c["ridge_alpha"], c["embargo_bars"])
    sharpe = float(np.mean(np.sign(pred) * real) / (np.std(np.sign(pred) * real) + 1e-12)
                   * np.sqrt(ppy))
    # best single env-feature IC (characterize any structure: momentum at 60/240m etc.)
    feat_names = [f"s{s}_{f}" for s in c["scales"]
                  for f in ["logret", "atr", "park", "oz", "hz", "lz", "cz", "vz"]]
    single_ic = {feat_names[j]: float(spearmanr(X[:, j], y).correlation) for j in range(X.shape[1])}
    top_single = sorted(single_ic.items(), key=lambda kv: -abs(kv[1]))[:5]

    # --- NULL + verdict ---
    null = block_shuffle_null(pred, real, nm["n_boot"], nm["block_bars"])
    checks = {
        "ic_beats_null": null["ic_null_p"] < gate["ic_null_p_max"],
        "pf_beats_null": null["pf_null_p"] < gate["pf_null_p_max"],
        "pf_above_floor": null["obs_pf"] >= gate["pf_floor"],
        "ic_above_floor": null["obs_ic"] >= gate["ic_floor"],
    }
    go = all(checks[k] for k in g["verdict"]["require_all"])
    verdict = "GO (proceed to RL canary)" if go else "NO-GO / FALSIFIED (no frictionless edge — BTC/gold class)"

    report = {
        "instrument": "TX", "base_scale_min": base, "n_oos_bars": int(len(pred)),
        "oos_rank_ic": round(null["obs_ic"], 5), "oos_frictionless_pf": round(null["obs_pf"], 5),
        "oos_directional_sharpe_ann": round(sharpe, 4),
        "ic_null_p": round(null["ic_null_p"], 4), "pf_null_p": round(null["pf_null_p"], 4),
        "ic_null_mean": round(null["ic_null_mean"], 5), "pf_null_mean": round(null["pf_null_mean"], 5),
        "top_single_feature_ic": [{"feature": k, "ic": round(v, 5)} for k, v in top_single],
        "gates": gate, "checks": checks, "verdict": verdict,
    }
    (out_dir / "canary_report.json").write_text(json.dumps(report, indent=2))

    print("\n=== TX gmgp1/sg-1 FALSIFICATION CANARY ===")
    print(f"OOS bars: {len(pred):,}  (base {base}min, {c['wf_folds']}-fold WF, embargo {c['embargo_bars']})")
    print(f"OOS rank-IC      : {null['obs_ic']:+.5f}   (null mean {null['ic_null_mean']:+.5f}, p={null['ic_null_p']:.4f}, floor {gate['ic_floor']})")
    print(f"frictionless PF  : {null['obs_pf']:.4f}    (null mean {null['pf_null_mean']:.4f}, p={null['pf_null_p']:.4f}, floor {gate['pf_floor']})")
    print(f"dir. Sharpe (ann): {sharpe:+.3f}")
    print("top single-feature |IC|:", ", ".join(f"{k}={v:+.4f}" for k, v in top_single))
    print("checks:", {k: ("PASS" if v else "FAIL") for k, v in checks.items()})
    print(f"VERDICT: {verdict}")
    print(f"written: {out_dir/'canary_report.json'}")
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
