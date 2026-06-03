"""R0-regime ADDENDUM — strictly-causal, Prism-free regime detector.

Addresses the objection: "the Prism GAHMM labels came from a pre-X2-fix research
run + the GAHMM is fit-on-full-history (its own leak) + the parquet is partially
degraded (Chronos dead, posteriors stubbed) — how can the NO-GO be trusted?"

This script removes Prism ENTIRELY. It builds a latent-state regime detector
(2-state Gaussian mixture, the leak-free analogue of Prism's GAHMM) fit by
EXPANDING WINDOW: for trading day D the model is fit only on daily features
through D-1, and the regime for D is the component of the D-1 observation. State
identity is pinned by sorting components on the feature mean, so labels are
consistent across refits (no label-switching). Daily close is resampled from the
clean GC 15m OHLCV (no Prism `close`). Then we re-run the decisive tests.

Logic note: leakage can only MANUFACTURE apparent separation (bias toward GO); it
cannot hide a real one. The main test already got NO-GO with optimistic leaky
Prism labels. If a provably-causal, Prism-free detector ALSO gives NO-GO, the
"leaky inputs" objection is fully retired.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture

sys.argv = ["x"]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import r0_regime_separation as R  # noqa: E402  (after sys.path insert by design)

OUT = R.OUT
WIN_FOLDS, BREAKEVEN_FOLDS, INIT = R.WIN_FOLDS, R.BREAKEVEN_FOLDS, R.INIT_CAPITAL


def clean_daily_close() -> pd.Series:
    """Daily close resampled from the clean GC 15m OHLCV — no Prism dependency."""
    gc = pd.read_parquet(R.GC)[["timestamp", "close"]].copy()
    gc["timestamp"] = pd.to_datetime(gc["timestamp"])
    gc = gc.sort_values("timestamp")
    daily = gc.set_index("timestamp")["close"].resample("1D").last().dropna()
    return daily


def causal_gmm_regime(feature: pd.Series, *, min_train: int = 60, seed: int = 0) -> pd.Series:
    """Expanding-window 2-state GMM. For each date D, fit on feature[:D-1], assign
    D-1's observation to a component; label 1 = HIGH-feature component (pinned by
    sorting on component means). Strictly causal: the label used for trading on D
    is derived only from data <= D-1."""
    f = feature.dropna()
    idx = f.index
    labels = pd.Series(index=idx, dtype=float)
    vals = f.to_numpy().reshape(-1, 1)
    for i in range(len(f)):
        if i < min_train:
            continue
        train = vals[:i]  # strictly before position i  => causal for "label at i"
        gm = GaussianMixture(n_components=2, covariance_type="full",
                             random_state=seed, n_init=1, max_iter=100)
        gm.fit(train)
        high_comp = int(np.argmax(gm.means_.ravel()))  # pin state identity by mean
        comp = int(gm.predict(vals[i:i + 1])[0])
        labels.iloc[i] = 1.0 if comp == high_comp else 0.0
    return labels


def assign_causal_label_to_trading_days(label_by_date: pd.Series,
                                        trading_dates: pd.Series) -> pd.Series:
    """For trading day D use the most recent regime label with date < D (causal)."""
    lab = label_by_date.dropna()
    left = pd.DataFrame({"date": pd.to_datetime(pd.Series(trading_dates.unique()))}).sort_values("date")
    right = lab.rename("regime").reset_index()
    right.columns = ["date", "regime"]
    right["date"] = pd.to_datetime(right["date"])
    right = right.sort_values("date")
    m = pd.merge_asof(left, right, on="date", direction="backward", allow_exact_matches=False)
    return m.set_index("date")["regime"]


def within_period(daily: pd.DataFrame, col: str) -> list:
    rows = []
    for period in ["WIN", "BREAKEVEN"]:
        d = daily[daily.period == period].dropna(subset=[col])
        a = d[d[col] == 0.0]["pnl"]  # low-feature regime
        b = d[d[col] == 1.0]["pnl"]  # high-feature regime
        if len(a) >= 5 and len(b) >= 5:
            mwu = stats.mannwhitneyu(a, b, alternative="two-sided")
            rows.append({"regime_var": col, "period": period, "n": len(d),
                         "low_mean_pnl": float(a.mean()), "high_mean_pnl": float(b.mean()),
                         "low_pf": R.pf(a.values), "high_pf": R.pf(b.values),
                         "mwu_p": float(mwu.pvalue)})
        else:
            rows.append({"regime_var": col, "period": period, "n": len(d),
                         "low_mean_pnl": None, "high_mean_pnl": None,
                         "low_pf": None, "high_pf": None, "mwu_p": None})
    return rows


def oos_gate(bar: pd.DataFrame, daily: pd.DataFrame, col: str, trade_when_high: bool) -> dict:
    """A-priori gate (trade only in low- or high-feature regime), calibrated by a
    fixed a-priori choice, scored on held-out folds 2-3 at bar-level PF."""
    target = 1.0 if trade_when_high else 0.0
    mask = daily[col] == target
    oos_pf, oos_ret, tim = R.bar_pf_under_daily_mask(bar, daily, mask, [2, 3])
    ung_pf, ung_ret, _ = R.bar_pf_under_daily_mask(
        bar, daily, pd.Series(True, index=daily.index), [2, 3])
    return {"gate": f"{col}_{'high' if trade_when_high else 'low'}",
            "oos_pf_bar": oos_pf, "oos_pf_ungated": ung_pf,
            "oos_ret_pct": oos_ret, "oos_time_in_mkt": tim}


def causality_assert(close: pd.Series, vol: pd.Series):
    """Confirm the regime detector cannot see the future: perturbing close at date
    dp must not change ANY causal label dated STRICTLY BEFORE dp. (The label AT dp
    legitimately may change — bar dp enters its own rolling window.)"""
    base = causal_gmm_regime(vol, seed=0)
    pos = max(70, len(close) // 2)
    dp = close.index[pos]
    c2 = close.copy()
    c2.iloc[pos] *= 1.5
    vol2 = np.log(c2).diff().rolling(20).std()
    pert = causal_gmm_regime(vol2, seed=0)
    common = base.dropna().index.intersection(pert.dropna().index)
    before = common[common < dp]
    diff = int((base.loc[before] != pert.loc[before]).sum())
    if diff != 0:
        sys.exit(f"FATAL: causal regime label leaked future ({diff} labels dated < {dp} moved)")
    after_changed = int((base.loc[common[common >= dp]]
                         != pert.loc[common[common >= dp]]).sum())
    return {"labels_before_perturb_unchanged": True, "n_before": int(len(before)),
            "labels_at_or_after_perturb_changed": after_changed,
            "perturb_date": str(dp.date())}


def run(rule: str) -> dict:
    bar = R.load_bar_panel(rule)
    daily = bar.groupby(["fold", "date"], as_index=False).agg(pnl=("pnl", "sum"))
    daily["period"] = np.where(daily["fold"].isin(WIN_FOLDS), "WIN", "BREAKEVEN")

    close = clean_daily_close()
    ret = np.log(close).diff()
    vol20 = ret.rolling(20).std()
    mom10 = close.pct_change(10)

    vol_reg_by_date = causal_gmm_regime(vol20, seed=0)     # 1=high-vol (turbulent)
    trend_reg_by_date = causal_gmm_regime(mom10, seed=0)   # 1=up-trend

    daily["gmm_vol_regime"] = assign_causal_label_to_trading_days(
        vol_reg_by_date, daily["date"]).reindex(daily["date"]).to_numpy()
    daily["gmm_trend_regime"] = assign_causal_label_to_trading_days(
        trend_reg_by_date, daily["date"]).reindex(daily["date"]).to_numpy()

    wp = within_period(daily, "gmm_vol_regime") + within_period(daily, "gmm_trend_regime")
    gates = [
        oos_gate(bar, daily, "gmm_vol_regime", trade_when_high=False),   # trade only calm
        oos_gate(bar, daily, "gmm_trend_regime", trade_when_high=True),  # trade only up-trend
        oos_gate(bar, daily, "gmm_trend_regime", trade_when_high=False),
        oos_gate(bar, daily, "gmm_vol_regime", trade_when_high=True),
    ]
    # decisive flags
    sig_within = [r for r in wp if r["mwu_p"] is not None and r["mwu_p"] < 0.05]
    vars_both = [v for v in ["gmm_vol_regime", "gmm_trend_regime"]
                 if {"WIN", "BREAKEVEN"}.issubset(
                     {r["period"] for r in sig_within if r["regime_var"] == v})]
    oos_win = [g for g in gates if g["oos_pf_bar"] >= 1.20 and
               g["oos_pf_bar"] > g["oos_pf_ungated"] + 0.10]
    decision = "GO" if (vars_both and oos_win) else ("AMBIGUOUS" if (vars_both or oos_win) else "NO_GO")
    return {"rule": rule, "within_period": wp, "oos_gates": gates,
            "vars_sig_within_BOTH": vars_both,
            "oos_winners": [g["gate"] for g in oos_win], "decision": decision,
            "regime_balance": {
                "gmm_vol_high_frac": float(np.nanmean(daily["gmm_vol_regime"])),
                "gmm_trend_up_frac": float(np.nanmean(daily["gmm_trend_regime"]))}}


def main():
    close = clean_daily_close()
    vol20 = np.log(close).diff().rolling(20).std()
    tw = causality_assert(close, vol20)
    print(f"[causal-detector tripwire] {tw}")
    results = {}
    for rule in ["ens_pf_weighted", "solo_456", "ens_mean"]:
        r = run(rule)
        results[rule] = r
        print(f"\nRULE {rule}: decision={r['decision']}  "
              f"sig_within_BOTH={r['vars_sig_within_BOTH']}  oos_winners={r['oos_winners']}")
        for row in r["within_period"]:
            if row["mwu_p"] is not None:
                print(f"   {row['regime_var']:18s} {row['period']:10s} n={row['n']:3d} "
                      f"low_pf={row['low_pf']:.2f} high_pf={row['high_pf']:.2f} MWU_p={row['mwu_p']:.3f}")
        for g in r["oos_gates"]:
            print(f"   OOS {g['gate']:24s} pf_bar={g['oos_pf_bar']:.3f} "
                  f"(ungated {g['oos_pf_ungated']:.3f}) time_in_mkt={g['oos_time_in_mkt']:.2f}")
    primary = results["ens_pf_weighted"]["decision"]
    agree = all(results[r]["decision"] == primary for r in results)
    verdict = {"test": "R0-regime causal-detector addendum (Prism-free, leak-free)",
               "decision": primary, "robust_across_rules": bool(agree),
               "per_rule": {r: results[r]["decision"] for r in results},
               "tripwire": tw, "detail": results}
    (OUT / "causal_detector_addendum.json").write_text(json.dumps(verdict, indent=2))
    print(f"\n{'#'*64}\nCAUSAL-DETECTOR ADDENDUM DECISION: {primary} (robust={agree})\n{'#'*64}")
    return verdict


if __name__ == "__main__":
    main()
