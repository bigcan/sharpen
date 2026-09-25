"""The ONE look at the sealed clean window. Runs every pre-registered computation in a single pass.

Refuses to run while sealed (seal.backward_view raises). Records a look entry keyed by the SHA-256 of the code,
the strategy config, the gates file and the pre-registration; a second run is allowed only with IDENTICAL hashes
(a byte-for-byte replay for reporting), never with changed rules.

Usage: cd scripts && python -m research.finrlx_strategy.one_look
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm

from sharpen.crypto.eval import statistics as st

from . import book, common, ledger, metrics, pnl, proxy_panel, seal
from .paths import RESULTS

log = logging.getLogger("finrlx.one_look")
REPO = Path(__file__).resolve().parents[3]
GATES = REPO / "configs" / "finrlx_strategy.gates.yaml"
PKG = Path(__file__).resolve().parent
LOOKS = RESULTS / "looks.jsonl"
OUT = RESULTS / "one_look"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fingerprint() -> dict:
    gates = yaml.safe_load(GATES.read_text(encoding="utf-8"))
    files = sorted(PKG.glob("*.py")) + [common.CONFIG, GATES, REPO / gates["prereg"]["path"],
                                         REPO / "sharpen" / "features" / "cross_asset_signals.py"]
    return {str(f.relative_to(REPO)).replace("\\", "/"): _sha(f) for f in files}


def check_one_look(fp: dict) -> None:
    if not LOOKS.exists():
        return
    for line in LOOKS.read_text(encoding="utf-8").splitlines():
        prev = json.loads(line)["fingerprint"]
        if prev != fp:
            changed = sorted(k for k in set(prev) | set(fp) if prev.get(k) != fp.get(k))
            raise RuntimeError(f"clean window already looked at with different code/rules; changed: {changed}. "
                               "A wrong rule gets a NEW pre-registered test, not a second look.")


def subperiod_alphas(book_df: pd.DataFrame, bench: pd.DataFrame, n: int) -> list[dict]:
    idx = book_df.index
    edges = np.linspace(0, len(idx), n + 1).astype(int)
    out = []
    for i in range(n):
        s, e = idx[edges[i]], idx[edges[i + 1] - 1]
        a = metrics.alpha_vs(book_df.loc[s:e], bench.loc[s:e])
        out.append({"start": str(s.date()), "end": str(e.date()), "alpha_ann": a["alpha_ann"], "t": a["t_alpha"],
                    "sharpe": metrics.sharpe(book_df.loc[s:e, "excess"]),
                    "sharpe_spy": metrics.sharpe(bench.loc[s:e, "excess"])})
    return out


def residual(book_df: pd.DataFrame, bench: pd.DataFrame) -> pd.Series:
    """Daily alpha stream: book excess minus beta x SPY excess (beta from the monthly regression)."""
    beta = metrics.alpha_vs(book_df, bench)["beta"]
    return book_df["excess"] - beta * bench["excess"].reindex(book_df.index)


def ts_ic(close: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, h: int = 21) -> dict:
    """Time-series power of the trend conviction: pooled Spearman corr of conviction at month-end t with the
    same asset's next-h-day return (from t+1), monthly sampling, t-stat from the monthly mean of per-date ICs."""
    from sharpen.features import cross_asset_signals as cas
    long = cas.compute(close)
    conv = long.pivot(index="date", columns="ticker", values="trend_conviction").reindex_like(close)
    fwd = close.shift(-(h + 1)) / close.shift(-1) - 1.0
    me = pd.Series(close.index, index=close.index).groupby(close.index.to_period("M")).max()
    me = me[(me >= start) & (me <= end)]
    xs, ys = [], []
    for d in me:
        c, f = conv.loc[d], fwd.loc[d]
        ok = c.notna() & f.notna() & (c != 0)
        xs.extend(c[ok].tolist())
        ys.extend(f[ok].tolist())
    x, y = pd.Series(xs), pd.Series(ys)
    rho = float(x.rank().corr(y.rank()))
    # sign hit-rate by month (time-series): share of asset-months where sign(conv) == sign(fwd)
    hit = float((np.sign(x) == np.sign(y)).mean())
    n = len(x)
    return {"pooled_spearman": rho, "n_asset_months": n, "sign_hit_rate": hit,
            "t_naive": rho * np.sqrt(max(n - 2, 1)) / np.sqrt(max(1 - rho ** 2, 1e-12))}


def main() -> dict:
    warnings.filterwarnings("ignore", category=FutureWarning)
    ok, why = seal.prereg_status()
    if not ok:
        raise seal.SealedError(f"refusing the look: {why}")
    fp = fingerprint()
    check_one_look(fp)
    cfg, gates = common.load_config(), yaml.safe_load(GATES.read_text(encoding="utf-8"))
    pr = yaml.safe_load((REPO / gates["prereg"]["path"]).read_text(encoding="utf-8").split("```yaml")[1].split("```")[0])

    close, rf, provenance = proxy_panel.load(window="backward")
    start, end = pd.Timestamp(pr["window"]["start"]), pd.Timestamp(pr["window"]["end"])
    rets = close.pct_change(fill_method=None)
    cm_base = common.cost_model(cfg, proxies=True)
    cm_free = common.cost_model(cfg, proxies=True, mult=0.0)
    cm_harsh = common.cost_model(cfg, proxies=True, mult=cfg["costs"]["harsh_multiplier"])
    bench = pnl.buy_and_hold(rets, rf, "SPY", cm_base, start=start, end=end).loc[start:end]

    pre = book.raw_trend_weights(close)
    specs = {common.spec_name(s): s for s in common.grid_specs(cfg)}
    evs = {}
    for name, spec in specs.items():
        tg, diag = book.build_targets(close, spec, precomputed=pre)
        evs[name] = {"targets": tg, "diag": diag,
                     "daily": pnl.run(tg, rets, rf, cm_base, start=start, end=end).loc[start:end]}
    sel = pr["h1"]["cell"]
    assert sel in evs, sel
    b = evs[sel]["daily"]

    # ---- H1 money + alpha
    m = metrics.money(b).__dict__
    a = metrics.alpha_vs(b, bench)
    free = pnl.run(evs[sel]["targets"], rets, rf, cm_free, start=start, end=end).loc[start:end]
    harsh = pnl.run(evs[sel]["targets"], rets, rf, cm_harsh, start=start, end=end).loc[start:end]
    sr_spy = metrics.sharpe(bench["excess"])

    def sr_minus_spy(mult: float) -> float:
        d = pnl.run(evs[sel]["targets"], rets, rf, common.cost_model(cfg, proxies=True, mult=mult),
                    start=start, end=end).loc[start:end]
        return metrics.sharpe(d["excess"]) - sr_spy

    def alpha_at(mult: float) -> float:
        d = pnl.run(evs[sel]["targets"], rets, rf, common.cost_model(cfg, proxies=True, mult=mult),
                    start=start, end=end).loc[start:end]
        return metrics.alpha_vs(d, bench)["alpha_ann"]

    be_sr = metrics.breakeven_multiplier(sr_minus_spy)
    be_alpha = metrics.breakeven_multiplier(alpha_at)

    # ---- Sharpen battery
    n_trials = max(int(pr["deflation"]["n_trials"]),
                   ledger.n_trials(prior_family=pr["deflation"]["prior_family_trials"]))
    grid_sr = [metrics.sharpe(v["daily"]["excess"]) for v in evs.values()]
    grid_ir = [metrics.sharpe(residual(v["daily"], bench)) for v in evs.values()]
    bat_book = metrics.sharpen_battery(b["excess"], trial_sharpes_ann=grid_sr, n_trials=n_trials)
    bat_alpha = metrics.sharpen_battery(residual(b, bench), trial_sharpes_ann=grid_ir, n_trials=n_trials)
    perf = common.monthly_perf_matrix({k: v for k, v in evs.items()})
    pbo = st.probability_of_backtest_overfitting(perf.to_numpy(), n_splits=16)
    boot = metrics.paired_block_bootstrap(b["excess"], bench["excess"])
    subs = subperiod_alphas(b, bench, gates["h1_beats_spy"]["subperiods"])
    tsic = ts_ic(close, start, end)

    # ---- H2 / H3 (paired, same window)
    def uplift(x: str, y: str) -> dict:
        bx, by = evs[x]["daily"]["excess"], evs[y]["daily"]["excess"]
        bb = metrics.paired_block_bootstrap(bx, by)
        return {"cell": x, "baseline": y, "sharpe": metrics.sharpe(bx), "sharpe_baseline": metrics.sharpe(by),
                "uplift": metrics.sharpe(bx) - metrics.sharpe(by), "p_uplift_le_0": bb["p_dsr_le_0"],
                "ci": bb["dsr_ci"]}
    h2 = uplift(pr["h2"]["cell"], pr["h2"]["baseline"])
    h3 = uplift(pr["h3"]["cell"], pr["h3"]["baseline"])

    g1 = gates["h1_beats_spy"]
    n_pos = sum(1 for s in subs if s["alpha_ann"] > 0)
    legs = {
        "sharpe_exceeds_spy": m["sharpe"] > sr_spy,
        "alpha_t": a["t_alpha"] >= g1["alpha_t_min"],
        "alpha_dsr": (bat_alpha["dsr"] or {}).get("dsr", 0.0) >= g1["alpha_dsr_min"],
        "book_dsr": (bat_book["dsr"] or {}).get("dsr", 0.0) >= g1["book_dsr_min"],
        "psr": bat_book["psr"] >= g1["psr_min"],
        "pbo": (pbo or {}).get("pbo", 1.0) <= g1["pbo_max"],
        "cost_gap": (metrics.sharpe(free["excess"]) - m["sharpe"]) <= g1["cost_gap_max"],
        "subperiods": n_pos >= g1["min_subperiods_alpha_positive"],
    }
    h1_pass = all(legs.values())
    fam = gates["family"]["familywise_alpha"]
    h2_pass = h1_pass and h2["uplift"] >= gates["h2_allocator"]["min_sharpe_uplift"] and \
        h2["p_uplift_le_0"] <= gates["h2_allocator"]["max_p_uplift_le_0"]
    h3_pass = h1_pass and h3["uplift"] >= gates["h3_vol_managed_core"]["min_sharpe_uplift"] and \
        h3["p_uplift_le_0"] <= gates["h3_vol_managed_core"]["max_p_uplift_le_0"]

    res = {
        "window": [str(start.date()), str(end.date())], "provenance": provenance, "n_trials": n_trials,
        "h1": {"cell": sel, "money": m, "alpha": a, "sharpe_spy": sr_spy, "money_spy": metrics.money(bench).__dict__,
               "sharpe_frictionless": metrics.sharpe(free["excess"]), "sharpe_harsh": metrics.sharpe(harsh["excess"]),
               "breakeven_cost_multiplier_sr": be_sr, "breakeven_cost_multiplier_alpha": be_alpha,
               "battery_book": bat_book, "battery_alpha": bat_alpha, "pbo_grid": pbo, "bootstrap_vs_spy": boot,
               "subperiods": subs, "ts_ic": tsic, "p_one_sided_alpha": float(norm.sf(a["t_alpha"])),
               "legs": legs, "pass": h1_pass},
        "h2": {**h2, "pass": h2_pass, "tested": h1_pass, "level": fam / 2},
        "h3": {**h3, "pass": h3_pass, "tested": h1_pass, "level": fam / 2},
        "grid": {k: common.compact({"daily": v["daily"], "money": metrics.money(v["daily"]).__dict__,
                                    "alpha": metrics.alpha_vs(v["daily"], bench),
                                    "sharpe_spy": sr_spy}) for k, v in evs.items()},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "one_look.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    b.to_parquet(OUT / "h1_daily.parquet")
    bench.to_parquet(OUT / "spy_daily.parquet")
    for k, v in evs.items():
        ledger.append(name=k, window="backward", spec=specs[k].as_dict(), metrics=res["grid"][k], kind="one_look")
    with LOOKS.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                            "fingerprint": fp, "h1_pass": h1_pass}) + "\n")
    log.info("H1 %s legs=%s", "PASS" if h1_pass else "FAIL", legs)
    return res


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
