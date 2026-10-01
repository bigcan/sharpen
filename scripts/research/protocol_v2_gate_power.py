"""Operating characteristic of the Protocol v2 RL gates: P(pass) for a strategy of known true Sharpe.

Backs the tables in ``docs/research/protocol_v2_audit_2026-09-29.md``. No data, no GPU: every
number is a Monte Carlo over a window-level model, seeded, so the report reproduces exactly.

MODEL
-----
A window of T years yields a realized annualized net Sharpe

    SR_hat = s_true + sd_train * u + market_noise / sqrt(T)

(``u`` = training / seed dispersion; seeds evaluated on one shared window share the market term
with correlation ``rho``). The RL gates grade *bar-level* profit factor (sum of positive per-bar
P&L over sum of negative, flat bars excluded -- ``sharpen/hpo/evaluate.py:300-313``,
``scripts/sg1_arm_gate_backtest.py:171-176``). With x = per-in-market-bar Sharpe and
k = sigma / E|r|,

    PF = (1 + k x) / (1 - k x),    x = SR_hat / sqrt(f * bars_per_year)

where f is the fraction of bars in the market. ``--section map`` checks this against direct
per-bar simulation (normal and t(4) tails). A window is "profitable" iff SR_hat > 0 iff PF > 1.

WHAT IS MODELLED -- and why the chain numbers are UPPER bounds
-------------------------------------------------------------
Documented gates (docs/sharpops.md, formerly docs/protocol_v2.md, section 4): S1 HPO best-of-trials median-of-3 x 500-bar PF;
S2 L1 N seeds on one 2-month test (median PF floor, every seed PF > 1, CV(PF) <= 0.30, the
pre-committed ambiguous band escalating to 2N); S3 walk-forward K one-month windows, all
profitable, median PF floor; S4 60-day OOS PF >= 0.8 x WF-median PF (HOLD).
As-run WF gate G1 (``scripts/sg1_btc_velotrade_ensemble_eval.py:526-674``,
``configs/gmgp1_btc_x2deleak_canary_ensemble.gates.yaml`` g1_solo_baseline): PASS iff at least
``need_seeds`` seeds clear ``pf_floor`` in at least ``need_folds`` folds.
Legs NOT modelled (fixed-lot DD stress, MDD breach, compliance, obs-noise, sensitivity) can only
lower pass rates, so every chain probability printed here is an upper bound.

Run:  python scripts/research/protocol_v2_gate_power.py --section all
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.stats import norm

K_NORMAL = float(np.sqrt(np.pi / 2))  # sigma / E|r|, normal
K_T4 = float(np.sqrt(2.0))            # t(4): E|T| = 1, sd = sqrt(2)
BARS = {"15m crypto 24/7": 35040, "15m gold": 23920, "3m crypto 24/7": 175200}
SRS = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 15.0, 30.0)


def pf_from_sr(sr_hat, nb, f, k=K_NORMAL):
    kx = np.clip(k * np.asarray(sr_hat) / np.sqrt(f * nb), -0.999999, 0.999999)
    return (1 + kx) / (1 - kx)


def sr_for_pf(pf, nb, f, k=K_NORMAL):
    return (pf - 1) / (k * (pf + 1)) * np.sqrt(f * nb)


# --------------------------------------------------------------------------- map check
def section_map(rng) -> None:
    print("\n[map] bar-PF vs annualized Sharpe: analytic vs direct per-bar simulation (4y paths)")
    for label, nb in (("15m crypto 24/7", 35040), ("15m gold", 23920)):
        for f in (1.0, 0.3):
            for sr in (1.0, 3.0, 8.0):
                out = []
                for dist, k in (("normal", K_NORMAL), ("t4", K_T4)):
                    n, pfs = int(nb * 4), []
                    for _ in range(20):
                        z = rng.standard_normal(n) if dist == "normal" else rng.standard_t(4, n) / K_T4
                        r = (sr / np.sqrt(f * nb) + z) * (rng.random(n) < f)
                        pfs.append(r[r > 0].sum() / -r[r < 0].sum())
                    out.append(f"{dist} {float(pf_from_sr(sr, nb, f, k)):.4f}/{np.mean(pfs):.4f}")
                print(f"  {label:15s} f={f:.1f} SR={sr:4.1f}  analytic/direct  " + "  ".join(out))
    print("\n[map] true annualized Sharpe at which the bar-PF point estimate equals the floor "
          "(normal tails; t(4) tails ~11% lower)")
    print(f"  {'bars':16s} {'f':>4s}   PF 1.10   PF 1.20   PF 1.50")
    for label, nb in BARS.items():
        for f in (0.3, 1.0):
            print(f"  {label:16s} {f:4.1f} " + "".join(f"{sr_for_pf(p, nb, f):9.1f}" for p in (1.10, 1.2, 1.5)))


# --------------------------------------------------------------------------- documented chain
def _s1_hpo(rng, s, nb, f, floor, P, n_trials=50, slice_bars=500, n_slices=3):
    sr = s + rng.standard_normal((P, n_trials, n_slices)) / np.sqrt(slice_bars / nb)
    return np.median(pf_from_sr(sr, nb, f), axis=2).max(axis=1) >= floor


def _s2_l1(rng, s, nb, f, floor, P, n=10, T=1 / 6, rho=0.5, sd_train=0.5, cv_max=0.30,
           amb=(0.22, 0.38)):
    z0 = rng.standard_normal((P, 1))

    def draw(m):
        zi = rng.standard_normal((P, m))
        return (s + sd_train * rng.standard_normal((P, m))
                + (np.sqrt(rho) * z0 + np.sqrt(1 - rho) * zi) / np.sqrt(T))

    pf = pf_from_sr(draw(n), nb, f)
    cv = pf.std(axis=1, ddof=1) / pf.mean(axis=1)
    esc = (cv >= amb[0]) & (cv <= amb[1])
    pf2 = np.concatenate([pf, pf_from_sr(draw(n), nb, f)], axis=1)
    cv2 = pf2.std(axis=1, ddof=1) / pf2.mean(axis=1)
    med = np.where(esc, np.median(pf2, axis=1), np.median(pf, axis=1))
    mn = np.where(esc, pf2.min(axis=1), pf.min(axis=1))
    return (med >= floor) & (mn >= 1.0) & (np.where(esc, cv2, cv) <= cv_max), mn >= 1.0


def _s3_wf(rng, s, nb, f, floor, P, K=4, T=1 / 12, sd_train=0.5):
    sr = s + sd_train * rng.standard_normal((P, K)) + rng.standard_normal((P, K)) / np.sqrt(T)
    med = np.median(pf_from_sr(sr, nb, f), axis=1)
    allprof = (sr > 0).all(axis=1)
    return allprof & (med >= floor), allprof, med


def _s4_hold(rng, s, nb, f, wf_med, P, T=60 / 365, hold=0.8):
    return pf_from_sr(s + rng.standard_normal(P) / np.sqrt(T), nb, f) >= hold * wf_med


def chain_table(rng, P, bars="15m crypto 24/7", f=0.6, K=4, rho=0.5, sd_train=0.5) -> None:
    nb = BARS[bars]
    variants = (("doc floors 1.5/1.5/-", 1.5, 1.5, 1.0),
                ("yaml floors 1.2/1.2/1.10", 1.2, 1.2, 1.10),
                ("floors removed", 0.0, 1.0, 1.0))
    print(f"\n[chain] documented S1->S4 chain, {bars}, f={f}, WF {K} x 1 month, rho={rho}, "
          f"sd_train={sd_train}   (P(pass), upper bounds)")
    print(f"  {'true SR':>7s} | " + " | ".join(f"{v[0]:>24s}" for v in variants)
          + " | S2 all seeds>1 | S3 all windows>0 | S4 HOLD")
    for s in SRS:
        cells, diag = [], None
        for _, f1, f2, f3 in variants:
            p1 = _s1_hpo(rng, s, nb, f, f1, P // 4).mean()
            p2, allseeds = _s2_l1(rng, s, nb, f, f2, P, rho=rho, sd_train=sd_train)
            p3, allprof, wfmed = _s3_wf(rng, s, nb, f, f3, P, K=K, sd_train=sd_train)
            hold = _s4_hold(rng, s, nb, f, wfmed, P)
            cells.append(p1 * p2.mean() * (p3 & hold).mean())
            diag = (allseeds.mean(), allprof.mean(), hold.mean())
        print(f"  {s:7.1f} | " + " | ".join(f"{c:24.2e}" for c in cells)
              + f" | {diag[0]:14.3f} | {diag[1]:16.3f} | {diag[2]:7.3f}")
    print("  S1 HPO alone at floor 1.2 (best of 50 trials): "
          + ", ".join(f"SR {s:g}: {_s1_hpo(rng, s, nb, f, 1.2, P // 4).mean():.2f}" for s in (0.0, 1.0)))


# --------------------------------------------------------------------------- as-run WF G1
def _g1(rng, s, nb, f, pf_floor, K, need_folds, seeds, need_seeds, P, rho=0.5, sd_train=0.5,
        T=1 / 12):
    z0 = rng.standard_normal((P, K, 1))
    zi = rng.standard_normal((P, K, seeds))
    sr = (s + sd_train * rng.standard_normal((P, K, seeds))
          + (np.sqrt(rho) * z0 + np.sqrt(1 - rho) * zi) / np.sqrt(T))
    folds_ok = (pf_from_sr(sr, nb, f) >= pf_floor).sum(axis=1)
    return ((folds_ok >= need_folds).sum(axis=1) >= need_seeds).mean()


def section_g1(rng, P) -> None:
    srs = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0)
    print("\n[g1] walk-forward gate G1 AS IMPLEMENTED: PASS iff >= need_seeds seeds clear pf_floor "
          "in >= need_folds folds (1-month folds, 15m crypto, f=0.6)")
    print(f"  {'rule':44s}" + "".join(f"  SR{s:<4g}" for s in srs))
    rules = (("canary: 2 of 5 seeds, >=3/4 folds, PF>=1.10", 1.10, 4, 3, 5),
             ("canary rule, PF>=1.05", 1.05, 4, 3, 5),
             ("canary rule, PF>=1.00", 1.00, 4, 3, 5),
             ("SG-1-BTC: 2 of 3 seeds, >=7/8 folds, PF>=1.10", 1.10, 8, 7, 3),
             ("SG-1-BTC rule, PF>=1.00", 1.00, 8, 7, 3))
    for label, fl, K, need_folds, seeds in rules:
        vals = [_g1(rng, s, BARS["15m crypto 24/7"], 0.6, fl, K, need_folds, seeds, 2, P) for s in srs]
        print(f"  {label:44s}" + "".join(f"{v:8.3f}" for v in vals))


# --------------------------------------------------------------------------- one pooled test
def section_power() -> None:
    print("\n[power] ONE pre-registered one-sided pooled-OOS Sharpe test: Phi(SR*sqrt(T) - z_alpha)")
    for alpha in (0.20, 0.10, 0.05):
        z = norm.ppf(1 - alpha)
        print(f"  alpha={alpha:.2f}   T(yr) | SR 0.5  SR 1.0  SR 2.0  SR 3.0 | MDE@80%")
        for T in (1 / 3, 1.0, 2.0, 4.0, 8.0):
            p = "  ".join(f"{norm.cdf(s * np.sqrt(T) - z):5.2f}" for s in (0.5, 1, 2, 3))
            print(f"               {T:5.2f} | {p} | {(z + norm.ppf(0.8)) / np.sqrt(T):5.2f}")
    for d in (60, 120, 365):
        print(f"  {d:3d}-day window: SE(annualized SR) = {1 / np.sqrt(d / 365):.2f}")
    print("  RL overlay vs linear core on the same path: SE(dSR) = sqrt(2(1-rho)/T); years for 80% "
          "power at dSR 0.3, alpha 0.10:")
    for rho in (0.8, 0.9, 0.95):
        yrs = ((norm.ppf(0.9) + norm.ppf(0.8)) * np.sqrt(2 * (1 - rho)) / 0.3) ** 2
        print(f"    rho={rho:.2f}: {yrs:5.1f} years")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--section", default="all", choices=("map", "chain", "g1", "power", "all"))
    ap.add_argument("--paths", type=int, default=20000)
    a = ap.parse_args()
    rng = np.random.default_rng(20260929)
    if a.section in ("map", "all"):
        section_map(rng)
    if a.section in ("chain", "all"):
        chain_table(rng, a.paths)
        chain_table(rng, a.paths, rho=0.9, sd_train=0.1)
        chain_table(rng, a.paths, bars="3m crypto 24/7", f=0.3, K=8)
    if a.section in ("g1", "all"):
        section_g1(rng, 2 * a.paths)
    if a.section in ("power", "all"):
        section_power()


if __name__ == "__main__":
    main()
