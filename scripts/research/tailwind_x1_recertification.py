"""TAILWIND-v1 X1 re-certification: ONE pre-registered pass (Tier-2 2026-09-29, roadmap X1).

The audit's specification, applied exactly (``docs/research/tailwind-v1_deep_lifecycle_audit_2026-09-29.md``
row X1). Every setting comes from ``configs/tailwind_v1_x1.prereg.yaml``, and every threshold
from the gates file that prereg names; none is authored here.

  * SERIES. The real ``TwoSleeveExecutor`` on the lead-1 challenge config, financed (excess of
    the 3m T-bill plus short borrow), trimmed to [research active start, research_cutoff]. X2
    proved this lead-1 batch series forward-realizable (forward = batch at every step).
  * DSR. In-sample, deflated once, against the matched financed pool (18 graded momentum books
    each combined with BAB, financed at the same borrow, measured over the same window), at every
    N of the ledger bracket. ``null_max_calibration``'s n* is a diagnostic.
  * P(pass). N10 estimator: a disjoint walk over the window that advances on the run it scores,
    with TIMEOUTs right-censored, one horizon, the declared terminal kills, and the challenge
    simulator's ``intraday_mae_mult`` at each registered value. Wilson lower bound binds.
  * Beside it: the momentum-engine DSR, borrow sensitivity, the research basis, the executor-
    basis render and Stage-4. None of these gate X1.

Modes
  ``--mode dry-run``  lineage, pins and the offline guard only; computes no strategy number.
  ``--mode certify``  the single pass. Refuses unless the prereg is ``registered``, signed off,
                      committed and clean, and refuses to overwrite an existing pass artifact.

Exit codes (Tier-2 N3): PASS 0, REVIEW 3, BLOCK 1, FAIL 1.

Offline by construction: every network fetch path is replaced by a raising stub before any data
module runs, and the pinned files are re-hashed after the pass.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import datetime as _dt
import hashlib
import importlib.metadata as _md
import json
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import yaml as _yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sharpen.prop.challenge_simulator import (  # noqa: E402
    DAILY_BREACH,
    DD_BREACH,
    PASS,
    TIMEOUT,
    FirmRules,
    SizingPolicy,
    simulate_path,
)

log = logging.getLogger("tailwind_x1")

PREREG = ROOT / "configs" / "tailwind_v1_x1.prereg.yaml"
PREREG_DOC = ROOT / "docs" / "research" / "tailwind_x1_preregistration_2026-09-29.md"
OUT_DIR = ROOT / "results" / "tailwind_v1"
EXIT = {"PASS": 0, "REVIEW": 3, "BLOCK": 1, "FAIL": 1}
# Code the executor series depends on; a change between the prereg commit and the pass is a
# declared integrity warning (REVIEW), never silent.
DEPENDENCY_PATHS = ("sharpen", "scripts/research", "configs/tailwind_v1_challenge.yaml",
                    "configs/tailwind_v1.gates.yaml", "configs/tailwind_v1_challenge_v2.gates.yaml",
                    "configs/tailwind_v1_stage4.gates.yaml", "configs/tailwind_v1_x1.prereg.yaml")
LIBS = ("numpy", "pandas", "scipy", "pyarrow", "yfinance", "PyYAML")


class IntegrityError(RuntimeError):
    """A pin, lineage or offline-guard check failed: no verdict may be read."""


# ======================================================================================
# pure statistics (unit-tested data-free)
# ======================================================================================
def wilson(k: int, n: int, confidence: float = 0.95) -> tuple[float, float] | None:
    """Two-sided Wilson score interval for k successes in n trials; None when n == 0."""
    if n <= 0:
        return None
    if not 0 <= k <= n:
        raise ValueError(f"k={k} outside [0, n={n}]")
    z = NormalDist().inv_cdf(1.0 - (1.0 - confidence) / 2.0)
    p = k / n
    den = 1.0 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    hw = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
    return max(0.0, ctr - hw), min(1.0, ctr + hw)


def disjoint_walk(r: np.ndarray, lead: FirmRules, policy: SizingPolicy, horizon: int, *,
                  paired: FirmRules | None = None) -> list[dict]:
    """Non-overlapping challenge windows over ``r`` (N10).

    A window starts at the cursor and runs ``simulate_path`` under ``lead`` for at most
    ``horizon`` sessions. A resolved window (pass or breach on day d) moves the cursor d
    sessions on, so the next window starts the session after resolution. A TIMEOUT is
    right-censored: ``horizon_cap`` when the full horizon ran (the cursor jumps the horizon),
    ``data_end`` when the data ran out first (the walk stops). ``paired`` rules, if given, are
    run on the same slice for a paired read (e.g. firm lead, kills paired -> needless)."""
    r = np.asarray(r, dtype=np.float64)
    if not np.all(np.isfinite(r)):
        raise ValueError("returns contain non-finite values")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    rows: list[dict] = []
    i, n = 0, len(r)
    while i < n:
        seg = r[i:i + horizon]
        out, day, _ = simulate_path(seg, lead, policy)
        row = {"i": i, "outcome": out, "days": int(day), "censored": None}
        if paired is not None:
            row["paired"] = simulate_path(seg, paired, policy)[0]
        if out == TIMEOUT:
            row["censored"] = "data_end" if len(seg) < horizon else "horizon_cap"
            rows.append(row)
            if row["censored"] == "data_end":
                break
            i += horizon
            continue
        rows.append(row)
        i += int(day)
    return rows


def summarize_walk(rows: list[dict], confidence: float = 0.95) -> dict:
    """P(pass), daily-breach rate and (if paired) the needless share, one estimator for all."""
    resolved = [x for x in rows if x["censored"] is None]
    n = len(resolved)
    k_pass = sum(x["outcome"] == PASS for x in resolved)
    k_daily = sum(x["outcome"] == DAILY_BREACH for x in resolved)
    k_dd = sum(x["outcome"] == DD_BREACH for x in resolved)
    ci = wilson(k_pass, n, confidence)
    n_cap = sum(x["censored"] == "horizon_cap" for x in rows)
    ci_fc = wilson(k_pass, n + n_cap, confidence)      # M2 diagnostic: horizon-cap TIMEOUT = fail
    out = {
        "n_windows": len(rows), "n_resolved": n,
        "n_censored_horizon_cap": sum(x["censored"] == "horizon_cap" for x in rows),
        "n_censored_data_end": sum(x["censored"] == "data_end" for x in rows),
        "k_pass": k_pass, "k_daily_breach": k_daily, "k_dd_breach": k_dd,
        "p_pass": (k_pass / n) if n else None,
        "wilson_lower": None if ci is None else ci[0],
        "wilson_upper": None if ci is None else ci[1],
        "daily_breach_rate": (k_daily / n) if n else None,
        "p_pass_horizon_timeouts_as_fail": (k_pass / (n + n_cap)) if (n + n_cap) else None,
        "wilson_lower_horizon_timeouts_as_fail": None if ci_fc is None else ci_fc[0],
    }
    if resolved and "paired" in resolved[0]:
        passes = [x for x in resolved if x["outcome"] == PASS]
        needless = sum(x["paired"] in (DD_BREACH, DAILY_BREACH) for x in passes)
        nci = wilson(needless, len(passes), confidence)
        out.update({"n_lead_passes": len(passes), "n_needless": needless,
                    "needless_share": (needless / len(passes)) if passes else None,
                    "needless_wilson_upper": None if nci is None else nci[1]})
    return out


def p_pass_by_mae(r: np.ndarray, firm: FirmRules, kills: FirmRules, maes: list[float],
                  horizon: int, confidence: float = 0.95) -> dict:
    """At each intraday MAE multiplier: the gate read (declared kills), the firm-only
    diagnostic, and the firm walk with the kills paired on the same slices (needless share)."""
    out = {}
    for m in maes:
        pol = SizingPolicy(vol_multiplier=1.0, intraday_mae_mult=float(m))
        out[f"mae_{float(m)}"] = {
            "declared_kills": summarize_walk(disjoint_walk(r, kills, pol, horizon), confidence),
            "firm_only": summarize_walk(disjoint_walk(r, firm, pol, horizon), confidence),
            "firm_with_paired_kills": summarize_walk(
                disjoint_walk(r, firm, pol, horizon, paired=kills), confidence),
        }
    return out


def verdict_from(dsr_by_n: dict, min_dsr: float, p_lower_by_mae: dict, min_p: float,
                 warnings: list[str]) -> str:
    """PASS needs DSR >= min_dsr at every N AND Wilson-lower >= min_p at every MAE. A None
    (undefined) statistic fails closed. PASS with a declared integrity warning is REVIEW."""
    if not dsr_by_n or not p_lower_by_mae:
        raise ValueError("empty gate inputs")
    dsr_ok = all(v is not None and v >= min_dsr for v in dsr_by_n.values())
    p_ok = all(v is not None and v >= min_p for v in p_lower_by_mae.values())
    if not (dsr_ok and p_ok):
        return "BLOCK"
    return "REVIEW" if warnings else "PASS"


def month_end_sessions(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Sessions that are the NYSE month-end (the ex-ante calendar X2 uses), so an incomplete
    trailing month contributes no rebalance."""
    from sharpen.data import trading_calendar as tc

    return pd.DatetimeIndex([d for d in pd.DatetimeIndex(idx).sort_values() if tc.is_month_end(d)])


# ======================================================================================
# lineage, pins and the offline guard
# ======================================================================================
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True,
                          check=True).stdout.strip()


def check_pins(prereg: dict) -> dict:
    """Every pinned file must match its registered sha256, and the executor manifest its
    registered content hash. Raises IntegrityError on the first mismatch."""
    got = {}
    for rel, want in prereg["pins"].items():
        p = ROOT / rel
        if not p.exists():
            raise IntegrityError(f"pinned file missing: {rel}")
        h = sha256(p)
        if h != want:
            raise IntegrityError(f"pin mismatch {rel}: {h[:16]} != registered {want[:16]}")
        got[rel] = h
    man_path = ROOT / "results" / "tailwind_v1" / "ohlcv_daily.manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    if man.get("content_sha256_16") != prereg["manifest_content_sha256_16"]:
        raise IntegrityError(f"manifest content hash {man.get('content_sha256_16')} != registered "
                             f"{prereg['manifest_content_sha256_16']}")
    return {"files_sha256": got, "manifest_sha256": sha256(man_path),
            "manifest_content_sha256_16": man["content_sha256_16"],
            "manifest_status": man.get("status"), "manifest_reconstructed": man.get("reconstructed")}


def _raise_fetch(*_a, **_k):
    raise IntegrityError("network fetch attempted during X1 (offline guard)")


@contextlib.contextmanager
def offline_guard(prereg: dict):
    """Replace every fetch path with a raising stub for the duration; re-hash the pins after."""
    import yfinance

    from sharpen.data import cross_asset_loader as cal
    from sharpen.data import treasury_curve_loader as tcl

    saved = [(yfinance, "download", yfinance.download), (cal, "fetch_ohlcv_wide", cal.fetch_ohlcv_wide),
             (tcl, "_fetch_yahoo_curve", tcl._fetch_yahoo_curve)]
    for mod, name, _ in saved:
        setattr(mod, name, _raise_fetch)
    try:
        yield
    finally:
        for mod, name, fn in saved:
            setattr(mod, name, fn)
        check_pins(prereg)          # a fetch that slipped past the stubs would change a hash


def lineage(prereg: dict, prereg_commit: str | None) -> dict:
    head = _git("rev-parse", "HEAD")
    porcelain = _git("status", "--porcelain")
    dirty = [ln[3:] for ln in porcelain.splitlines() if ln.strip()]
    dep_changed: list[str] = []
    if prereg_commit:
        dep_changed = [p for p in _git("diff", "--name-only", prereg_commit, "HEAD", "--",
                                       *DEPENDENCY_PATHS).splitlines() if p]
    dep_dirty = [p for p in dirty if p.replace("\\", "/").startswith(DEPENDENCY_PATHS)]
    libs = {}
    for name in LIBS:
        try:
            libs[name] = _md.version(name)
        except _md.PackageNotFoundError:
            libs[name] = None
    cfg_files = ["configs/tailwind_v1_challenge.yaml", "configs/tailwind_v1.gates.yaml",
                 "configs/tailwind_v1_challenge_v2.gates.yaml", "configs/tailwind_v1_stage4.gates.yaml",
                 "configs/tailwind_v1_x1.prereg.yaml"]
    return {
        "git_head": head, "git_dirty": bool(dirty), "git_dirty_paths": dirty,
        "prereg_commit": prereg_commit,
        "dependency_paths_changed_since_prereg": dep_changed,
        "dependency_paths_uncommitted": dep_dirty,
        "script_sha256": sha256(Path(__file__)),
        "prereg_doc_sha256": sha256(PREREG_DOC) if PREREG_DOC.exists() else None,
        "config_sha256": {c: sha256(ROOT / c) for c in cfg_files},
        "python": platform.python_version(), "libs": libs,
        "cwd": str(Path.cwd()), "run_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "window": prereg["window"], "lead": prereg["basis"]["decision_lead_bars"],
        "financing": prereg["financing"],
    }


# ======================================================================================
# the pass
# ======================================================================================
def _gate_value(ref: dict) -> object:
    node = _yaml.safe_load((ROOT / ref["file"]).read_text(encoding="utf-8"))
    for part in ref["key"].split("."):
        node = node[part]
    return node


def _executor_cfg(base: dict, fin: dict, bps: float, drop: list[str] | None = None) -> dict:
    c = copy.deepcopy(base)
    c["financing"] = {"model": fin["model"], "tenor": fin["tenor"], "day_count": fin["day_count"],
                      "short_borrow_bps": float(bps)}
    for s in drop or []:
        del c["sleeves"][s]
    return c


def _dsr_block(observed: pd.Series, pool: dict[str, pd.Series], ns: list[int]) -> dict:
    from sharpen.crypto.eval.statistics import deflated_sharpe_ratio, excess_kurtosis, skewness
    import xsec_momentum_falsification as mom

    ann = mom.ANN
    cd = observed.dropna().to_numpy(dtype=np.float64)
    obs = mom.sharpe(observed) / np.sqrt(ann)
    trials = [mom.sharpe(s) / np.sqrt(ann) for s in pool.values()]
    g1, g2 = skewness(cd.tolist()), excess_kurtosis(cd.tolist())
    by_n = {}
    for n in ns:
        d = deflated_sharpe_ratio(obs, trials, n_obs=len(cd), skew=g1, excess_kurt=g2,
                                  n_trials=int(n), periods_per_year=ann)
        by_n[str(n)] = None if d is None else {"dsr": float(d["dsr"]), "sr_star_ann": float(d["sr_star_ann"])}
    return {"observed_sharpe_ann": float(mom.sharpe(observed)), "n_obs": int(len(cd)),
            "skew": float(g1), "excess_kurtosis": float(g2), "n_pool": len(trials),
            "pool_sharpe_sd_ann": float(np.std([t * np.sqrt(ann) for t in trials], ddof=1)),
            "by_n": by_n}


def research_pools(spec, bps: float, w0: pd.Timestamp, w1: pd.Timestamp) -> dict:
    """Financed research books over the window: the matched combined pool, the momentum-only
    pool, and the deployed research-basis combined book (a diagnostic)."""
    import tailwind_dsr_consistent as tdc

    atb, pf, mom = tdc._m()
    close, rets, bench, rebal = atb._mom_frame()
    books = mom.build_books(close, rets, bench)
    if len(books) != 18 or "TSMOM_pooled_monthly" not in books:
        raise IntegrityError(f"research grid is {len(books)} books; the registered pool is 18")
    bab = mom.run_book("defensive_bab", atb._bab_weights(close, rets, rebal), rets, bench)
    cash, borrow = tdc._rates(rets.index, spec, bps)
    ex = {b: tdc._excess(bk, rets, cash, borrow) for b, bk in books.items()}
    ex_def = tdc._excess(bab, rets, cash, borrow)
    comb_full = {b: pf.risk_parity([s, ex_def])[0] for b, s in ex.items()}
    return {"combined": {b: s.loc[w0:w1] for b, s in comb_full.items()},
            "momentum": {b: s.loc[w0:w1] for b, s in ex.items()},
            "deployed_full": comb_full["TSMOM_pooled_monthly"]}


def run_pass(prereg: dict) -> dict:
    import tailwind_dsr_consistent as tdc
    import tailwind_executor_recompute as ter
    import tailwind_forward_path_render as tfr
    import tailwind_stage4_recent_oos as s4

    from sharpen.crypto.eval.statistics import block_bootstrap_sharpe_ci
    from sharpen.data import financing as fin_mod
    from sharpen.envs.allocator_factory import decision_lead_bars, execution_stamp

    w0, w1 = pd.Timestamp(prereg["window"]["start"]), pd.Timestamp(prereg["window"]["end"])
    fin = prereg["financing"]
    base = _yaml.safe_load((ROOT / prereg["basis"]["config"]).read_text(encoding="utf-8"))
    if decision_lead_bars(base) != int(prereg["basis"]["decision_lead_bars"]):
        raise IntegrityError("config decision_lead_bars != registered lead")
    cspec = fin_mod.financing_spec(base)
    if (cspec.model, cspec.tenor, int(cspec.day_count), float(cspec.short_borrow_bps)) != (
            fin["model"], fin["tenor"], int(fin["day_count"]), float(fin["borrow_bps_primary"])):
        raise IntegrityError(f"config financing {cspec} != registered {fin}")
    b_primary = float(fin["borrow_bps_primary"])
    borrows = [b_primary] + [float(b) for b in fin["borrow_bps_sensitivity"]]

    dsr_cfg = prereg["dsr"]
    ledger_full = int(prereg["ledger"]["totals"]["full"])
    ns = [int(n) for n in dsr_cfg["n_bracket"]]
    if ledger_full not in ns:
        raise IntegrityError("the ledger's full count is not in the registered bracket")
    ns_all = sorted(set(ns) | {int(n) for n in dsr_cfg.get("n_diagnostic", [])})
    min_dsr = float(_gate_value(dsr_cfg["min_dsr_source"]))

    # ---- series -------------------------------------------------------------------------
    series, pools, traj = {}, {}, {}
    for bps in borrows:
        s, _, detail = ter.build_executor_series(_executor_cfg(base, fin, bps))
        series[bps] = s.dropna()
        traj[bps] = detail["trajectory"]
        pools[bps] = research_pools(cspec, bps, w0, w1)
    s_mom = ter.build_executor_series(_executor_cfg(
        base, fin, b_primary, prereg["basis"]["momentum_engine_drop_sleeves"]))[0].dropna()

    # window integrity: the research book must go live exactly at w0 and end at w1
    dep = pools[b_primary]["deployed_full"].dropna()
    nz = dep[dep != 0]
    if nz.index[0] != w0 or dep.index[-1] != w1:
        raise IntegrityError(f"research active window {nz.index[0].date()}..{dep.index[-1].date()} "
                             f"!= registered {w0.date()}..{w1.date()}")
    for bps, s in series.items():
        if w0 not in s.index or w1 not in s.index:
            raise IntegrityError(f"executor series (borrow {bps}) lacks a window endpoint")
    ex_w = {bps: s.loc[w0:w1] for bps, s in series.items()}
    mom_w = s_mom.loc[w0:w1]

    # ---- DSR ----------------------------------------------------------------------------
    primary = pools[b_primary]
    dsr_exec = _dsr_block(ex_w[b_primary], primary["combined"], ns_all)
    dsr_mom = _dsr_block(mom_w, primary["momentum"], ns_all)
    dsr_research = _dsr_block(primary["combined"]["TSMOM_pooled_monthly"], primary["combined"], ns_all)
    corr = tdc._active_corr(primary["combined"])
    cal = tdc.null_max_calibration(corr)
    n_star = {"n_star": cal["n_star"], "k": cal["k"],
              "estimator_over_true_null_max": {str(n): tdc.estimator_ratio(cal, n) for n in ns_all}}
    sens = {}
    for bps in borrows[1:]:
        d = _dsr_block(ex_w[bps], pools[bps]["combined"], ns_all)
        sens[f"borrow_{int(bps)}bp"] = d

    # ---- P(pass) ------------------------------------------------------------------------
    pp_cfg = prereg["p_pass"]
    hard = _gate_value(pp_cfg["firm_source"])
    ks = pp_cfg["kills_source"]
    kill_dd_pct, kill_daily_pct = (float(_gate_value({"file": ks["file"], "key": k})) for k in ks["keys"])
    gross_ceiling, env_gross = (float(_gate_value(g)) for g in prereg["render"]["gross_sources"])
    daily_budget = float(_gate_value(pp_cfg["daily_breach_budget_source"]))
    horizon = int(_gate_value(pp_cfg["horizon_source"]))
    min_p = float(_gate_value(pp_cfg["min_p_pass_source"]))
    target = float(hard["profit_target_step1_pct"]) / 100.0
    firm = FirmRules(name="firm", profit_target=target,
                     max_total_dd=float(hard["max_total_loss_pct"]) / 100.0,
                     daily_loss_limit=float(hard["daily_loss_limit_pct"]) / 100.0,
                     max_days=None, min_trading_days=int(hard["min_trading_days"]),
                     dd_mode=hard["dd_mode"])
    kills = FirmRules(name="declared_kills", profit_target=target,
                      max_total_dd=kill_dd_pct / 100.0,
                      daily_loss_limit=kill_daily_pct / 100.0,
                      max_days=None, min_trading_days=int(hard["min_trading_days"]),
                      dd_mode=hard["dd_mode"])
    if kills.max_total_dd > firm.max_total_dd or kills.daily_loss_limit > firm.daily_loss_limit:
        raise IntegrityError("declared kills are looser than the firm limits")
    conf = float(pp_cfg["confidence"])
    maes = [float(m) for m in pp_cfg["intraday_mae_mult"]]

    def _pp(r: pd.Series) -> dict:
        return p_pass_by_mae(r.to_numpy(dtype=np.float64), firm, kills, maes, horizon, conf)

    p_pass = _pp(ex_w[b_primary])
    p_sens = {f"borrow_{int(b)}bp": _pp(ex_w[b]) for b in borrows[1:]}
    p_lower = {k: v["declared_kills"]["wilson_lower"] for k, v in p_pass.items()}

    # ---- render (executor basis) --------------------------------------------------------
    r_w = ex_w[b_primary]
    t = traj[b_primary]
    gross = pd.Series(np.asarray(t.gross_exposure, dtype=np.float64),
                      index=pd.to_datetime(np.asarray(t.timestamps, dtype=np.int64), unit="s")).loc[w0:w1]
    render = {
        "path_stats": tfr.path_stats(r_w, kills.max_total_dd, kills.daily_loss_limit,
                                     firm.max_total_dd, firm.daily_loss_limit),
        "needless_termination": {k: v["firm_with_paired_kills"] for k, v in p_pass.items()},
        "max_needless_share": float(_gate_value(prereg["render"]["needless_source"])),
        "daily_breach_budget": daily_budget,
        "gross": {"mean": float(gross.mean()), "p95": float(gross.quantile(0.95)),
                  "max": float(gross.max()), "render_basis_ceiling": gross_ceiling,
                  "env_max_gross_exposure": env_gross},
    }
    render["checks"] = {
        "needless_ok": {k: (v["needless_share"] is not None
                            and v["needless_share"] <= render["max_needless_share"])
                        for k, v in render["needless_termination"].items()},
        "daily_breach_ok": {k: (v["declared_kills"]["daily_breach_rate"] is not None
                                and v["declared_kills"]["daily_breach_rate"] <= render["daily_breach_budget"])
                            for k, v in p_pass.items()},
        "gross_ok": bool(render["gross"]["max"] <= gross_ceiling),
    }
    flags = [k for k, v in render["checks"].items()
             if not (all(v.values()) if isinstance(v, dict) else v)]
    render["decision"] = "RENDER_CLEAR" if not flags else "RENDER_FLAGS"
    render["failed_checks"] = flags

    # ---- Stage-4 (executor basis) -------------------------------------------------------
    s4_cfg = _yaml.safe_load((ROOT / prereg["stage4"]["gates"]).read_text(encoding="utf-8"))["stage4_recent_oos"]
    cutoff = pd.Timestamp(s4_cfg["research_cutoff"])
    full = series[b_primary]
    oos = full[full.index >= cutoff]
    baseline = s4.subperiod_baseline(full, s4_cfg["subperiods"], cutoff)
    oos_stats = s4.window_stats(oos, "recent_oos_cutoff_anchored_executor")
    n_rebal = int(len(month_end_sessions(oos.index)))
    boot = block_bootstrap_sharpe_ci(oos.to_numpy(dtype=np.float64), block=s4_cfg["sharpe_ci_block_days"],
                                     n_boot=s4_cfg["sharpe_ci_resamples"], periods_per_year=252)
    turn = pd.Series(np.asarray(t.turnovers, dtype=np.float64),
                     index=pd.to_datetime(np.asarray(t.timestamps, dtype=np.int64), unit="s"))
    stage4 = {
        "cutoff": str(cutoff.date()), "oos": oos_stats, "baseline": baseline,
        "n_rebalances_oos": n_rebal, "min_rebalances_for_power": s4_cfg["min_rebalances_for_power"],
        "power_sufficient": n_rebal >= s4_cfg["min_rebalances_for_power"],
        "sharpe_ci95": None if boot is None else [float(boot["ci_low"]), float(boot["ci_high"])],
        "p_sharpe_lt_0": None if boot is None else float(boot["p_sharpe_lt_0"]),
        "verdict": s4.verdict_for(oos_stats["pf"], baseline["median_pf"], s4_cfg),
        "compliance": s4.compliance_block(oos, turn, s4_cfg["compliance"], n_rebal),
        "deploy_gating": False,
        "note": "curve carried forward past its 2026-06-12 end inside this window (manifest WARN)",
    }

    # ---- verdict ------------------------------------------------------------------------
    return {
        "execution_stamp": execution_stamp(_executor_cfg(base, fin, b_primary)),
        "window": {"start": str(w0.date()), "end": str(w1.date()), "n_days": int(len(r_w))},
        "dsr": {"min_dsr": min_dsr, "n_primary": ledger_full, "n_bracket": ns,
                "executor_combined": dsr_exec, "momentum_engine": dsr_mom,
                "research_basis_diagnostic": dsr_research, "n_star_diagnostic": n_star,
                "borrow_sensitivity": sens},
        "p_pass": {"min_p_pass": min_p, "horizon": horizon, "binding": "declared_kills.wilson_lower",
                   "by_mae": p_pass, "borrow_sensitivity": p_sens},
        "render_executor_basis": render,
        "stage4_executor_basis": stage4,
        "_gate_inputs": {"dsr_by_n": {str(n): (dsr_exec["by_n"][str(n)] or {}).get("dsr") for n in ns},
                         "min_dsr": min_dsr, "p_lower_by_mae": p_lower, "min_p": min_p},
    }


# ======================================================================================
# CLI
# ======================================================================================
def _jsonable(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, _dt.date)):
        return str(o)
    raise TypeError(type(o))


def _prereg_commit_if_clean() -> str | None:
    rel = str(PREREG.relative_to(ROOT)).replace("\\", "/")
    if not _git("ls-files", rel) or _git("status", "--porcelain", "--", rel):
        return None
    return _git("log", "-1", "--format=%H", "--", rel) or None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=("dry-run", "certify"), required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    os.chdir(ROOT)                       # N6: the configs' cache_dir is repo-relative
    prereg = _yaml.safe_load(PREREG.read_text(encoding="utf-8"))
    commit = _prereg_commit_if_clean()
    out: dict = {"schema": "tailwind_x1_recertification/v1", "strategy": prereg["strategy"],
                 "mode": args.mode}
    try:
        out["pins"] = check_pins(prereg)
        out["lineage"] = lineage(prereg, commit)
        if args.mode == "dry-run":
            with offline_guard(prereg):
                pass
            log.info(json.dumps(out, indent=1, default=_jsonable))
            log.info("dry-run OK: pins match, offline guard armed and released, no number computed")
            return 0
        if prereg.get("status") != "registered" or not prereg.get("operator_signoff"):
            raise IntegrityError("prereg is not registered and signed off; certify refused")
        if commit is None:
            raise IntegrityError("prereg is uncommitted or dirty; certify refused")
        dest = OUT_DIR / f"x1_recertification_{_dt.date.today().isoformat()}.json"
        if any(OUT_DIR.glob("x1_recertification_*.json")):
            raise IntegrityError("an X1 pass artifact already exists; X1 is a single pass")
        with offline_guard(prereg):
            res = run_pass(prereg)
    except IntegrityError as e:
        out["verdict"] = {"decision": "FAIL", "reason": str(e)}
        log.error("X1 FAIL: %s", e)
        if args.mode == "certify":      # recorded, but outside the single-pass artifact name
            stamp = _dt.datetime.now().strftime("%Y-%m-%dT%H%M%S")
            (OUT_DIR / f"x1_attempt_FAIL_{stamp}.json").write_text(
                json.dumps(out, indent=2, default=_jsonable), encoding="utf-8")
        return EXIT["FAIL"]

    warnings = []
    if out["lineage"]["dependency_paths_changed_since_prereg"] or out["lineage"]["dependency_paths_uncommitted"]:
        warnings.append("executor-path code changed or uncommitted since the prereg commit")
    curve = json.loads((ROOT / "results/tailwind_v1/treasury_curve.manifest.json").read_text(encoding="utf-8"))
    if curve.get("status") != "PASS":
        warnings.append(f"treasury curve manifest status {curve.get('status')}")
    gi = res.pop("_gate_inputs")
    decision = verdict_from(gi["dsr_by_n"], gi["min_dsr"], gi["p_lower_by_mae"], gi["min_p"], warnings)
    out.update(res)
    out["verdict"] = {
        "decision": decision, "exit_code": EXIT[decision], "integrity_warnings": warnings,
        "dsr_by_n": gi["dsr_by_n"], "dsr_pass": all(v is not None and v >= gi["min_dsr"]
                                                    for v in gi["dsr_by_n"].values()),
        "p_pass_wilson_lower_by_mae": gi["p_lower_by_mae"],
        "p_pass_pass": all(v is not None and v >= gi["min_p"] for v in gi["p_lower_by_mae"].values()),
        "not_authorising": ("X1 authorises nothing. A PASS would still need a Tier-2 audit on the "
                            "final tree and an operator gate-record decision before any paper or "
                            "capital step."),
    }
    dest.write_text(json.dumps(out, indent=2, default=_jsonable), encoding="utf-8")
    log.info("X1 %s (exit %d) -> %s", decision, EXIT[decision], dest.relative_to(ROOT))
    return EXIT[decision]


if __name__ == "__main__":
    sys.exit(main())
