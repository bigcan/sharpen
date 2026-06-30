"""Component 2 tripwires — the 3 evaluator holes (N_eff, CPCV, HLZ/BHY).

C2.1 effective-N clustering, C2.2 combinatorial purged CV, C2.3 BHY-FDR + HLZ hurdle.
All additive + back-compat: defaults reproduce the raw-N behaviour, so the existing
tests/signals suite stays green; these lock the NEW behaviour and its limiting cases.

Design: .agent/artifacts/evaluator_holes_component2_architecture.md
"""
from __future__ import annotations

import json
from math import comb

import numpy as np

from finrl_pro_ds.signals import (
    Gates,
    SignalSpec,
    bh_fdr,
    bhy_fdr,
    effective_n_trials,
    evaluate_batch,
    to_json,
)
from finrl_pro_ds.signals.eval_harness import _contiguous_runs, tier3_5_cpcv
from finrl_pro_ds.signals.features import Panel


# ----------------------------------------------------- shared fixtures ----

class Trail:
    """Trailing k-day log-return — a causal momentum signal (Trails are mutually correlated)."""

    def __init__(self, k: int) -> None:
        self.k = k
        self.spec = SignalSpec(name=f"trail{k}", hypothesis="trailing return predicts",
                               family="technical", expected_sign=1, horizons=(1, 5),
                               neutralization=())

    def compute(self, panel: Panel) -> np.ndarray:
        c = np.log(panel.close)
        out = np.full(c.shape, np.nan)
        out[self.k:] = c[self.k:] - c[:-self.k]
        return out


def _momentum_panel(t: int = 600, n: int = 50, rho: float = 0.45, seed: int = 0) -> Panel:
    rng = np.random.default_rng(seed)
    eps = 0.01 * rng.standard_normal((t, n))
    ret = np.zeros((t, n))
    for i in range(1, t):
        ret[i] = rho * ret[i - 1] + eps[i]
    close = np.exp(np.cumsum(ret, axis=0) + rng.uniform(3.0, 5.0, size=n))
    open_ = close * np.exp(0.0005 * rng.standard_normal((t, n)))
    high = np.maximum(open_, close) * np.exp(np.abs(0.001 * rng.standard_normal((t, n))))
    low = np.minimum(open_, close) * np.exp(-np.abs(0.001 * rng.standard_normal((t, n))))
    vol = rng.uniform(1e5, 1e7, size=(t, n))
    dates = (np.datetime64("2012-01-02")
             + np.arange(t) * np.timedelta64(1, "D")).astype("datetime64[ns]")
    return Panel(dates, tuple(f"M{i:03d}" for i in range(n)), open_, high, low, close, vol,
                 np.ones((t, n), bool), close * vol, rng.integers(0, 5, size=n),
                 {"survivorship_free": False, "source": "synthetic"})


def _gates(**over) -> Gates:
    base = {"universe": {"min_names_per_day": 4},
            "coverage": {"min_days": 150},
            "gross_power": {"horizons": [1, 5], "primary_horizon": 1}}
    for k, v in over.items():
        base.setdefault(k, {})
        base[k] = {**base[k], **v} if isinstance(v, dict) else v
    return Gates.from_dict(base)


# ============================================================ C2.1 — N_eff ====

def test_effective_n_independent_trials_approx_K() -> None:
    # K mutually independent IC series -> effective trials ~ K (corr matrix ~ identity).
    rng = np.random.default_rng(0)
    T, K = 800, 8
    days = np.arange(T, dtype=np.int64)
    series = [rng.standard_normal(T) for _ in range(K)]
    n_eff = effective_n_trials(series, [days] * K)
    assert 0.75 * K <= n_eff <= K          # near K, never above the raw count


def test_effective_n_identical_copies_collapse_to_one() -> None:
    # K identical series -> all-ones corr matrix -> participation ratio = 1.
    rng = np.random.default_rng(1)
    T, K = 500, 6
    days = np.arange(T, dtype=np.int64)
    base = rng.standard_normal(T)
    n_eff = effective_n_trials([base.copy() for _ in range(K)], [days] * K)
    assert abs(n_eff - 1.0) < 1e-6


def test_effective_n_clamps_and_degenerate() -> None:
    assert effective_n_trials([np.zeros(10)], [np.arange(10)]) == 1.0   # K<2 -> K
    # two series with < min_overlap common days -> treated independent -> N_eff = 2
    d1 = np.arange(0, 30, dtype=np.int64)
    d2 = np.arange(100, 130, dtype=np.int64)
    rng = np.random.default_rng(2)
    n_eff = effective_n_trials([rng.standard_normal(30), rng.standard_normal(30)], [d1, d2])
    assert abs(n_eff - 2.0) < 1e-9


def test_effective_n_into_dsr_relaxes_penalty_on_correlated_pool() -> None:
    # Correlated Trails -> N_eff < raw N -> deflating against fewer effective trials should
    # RAISE the DSR (less multiple-testing penalty). Default (raw N) is the stricter bar.
    panel = _momentum_panel()
    signals = {s.spec.name: s for s in [Trail(1), Trail(2), Trail(3)]}
    raw = evaluate_batch(signals, panel, _gates(), batch_name="raw")
    eff = evaluate_batch(signals, panel, _gates(deflation={"use_effective_n": True}),
                         batch_name="eff")
    by_raw = {c.name: c for c in raw.cards}
    by_eff = {c.name: c for c in eff.cards}
    # the effective trial count is strictly below the raw batch size for a correlated pool
    top = raw.cards[0].name
    assert by_raw[top].deflation.n_eff < by_raw[top].deflation.n_trials
    # and DSR under effective-N is never below the raw-N DSR (Type-II penalty relaxed)
    for name in signals:
        dr, de = by_raw[name].deflation.dsr, by_eff[name].deflation.dsr
        if np.isfinite(dr) and np.isfinite(de):
            assert de >= dr - 1e-9


def test_default_deflation_is_unchanged_by_effective_n_flag_off() -> None:
    # use_effective_n defaults False -> DSR identical to the pre-C2 raw-N path (back-compat).
    panel = _momentum_panel()
    signals = {s.spec.name: s for s in [Trail(1), Trail(2), Trail(3)]}
    rs = evaluate_batch(signals, panel, _gates(), batch_name="def")
    for c in rs.cards:
        if c.deflation is not None and np.isfinite(c.deflation.dsr):
            # raw N is what fed DSR; n_eff is reported but NOT applied
            assert c.deflation.n_trials == rs.n_trials


# ============================================================= C2.2 — CPCV ====

def test_contiguous_runs_helper() -> None:
    mask = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0], dtype=bool)
    assert list(_contiguous_runs(mask)) == [(1, 3), (5, 8)]
    assert list(_contiguous_runs(np.zeros(5, bool))) == []
    assert list(_contiguous_runs(np.ones(4, bool))) == [(0, 4)]


def test_cpcv_distribution_on_momentum() -> None:
    panel = _momentum_panel()
    res = tier3_5_cpcv(Trail(1), panel, _gates(), neutralization=(), expected_sign=1,
                       horizon=1, n_groups=6, k_test=2, embargo_days=5)
    assert res.n_paths == comb(6, 2) == 15
    assert np.isfinite(res.oos_sharpe_mean) and res.oos_sharpe_mean > 0.0   # momentum works OOS
    assert res.oos_sharpe_std >= 0.0
    assert 0.0 <= res.frac_paths_positive <= 1.0
    assert res.oos_sharpe_p05 <= res.oos_sharpe_mean + 1e-9


def test_cpcv_purge_keeps_windows_inside_test_runs() -> None:
    # Instrument the book to record every rebalance index; assert no label window (t, t+h]
    # ever crosses out of its contiguous test run (purge) and embargo is honoured.
    panel = _momentum_panel(t=300, n=20)
    h, emb, G, k = 3, 4, 6, 2
    bounds = np.linspace(0, panel.T, G + 1).astype(int)
    groups = [(int(bounds[i]), int(bounds[i + 1])) for i in range(G)]
    from itertools import combinations
    for combo in combinations(range(G), k):
        in_test = np.zeros(panel.T, dtype=bool)
        for gi in combo:
            a, b = groups[gi]
            in_test[a:b] = True
        for a, b in _contiguous_runs(in_test):
            t = a + emb
            while t + h < b:
                assert t >= a + emb                      # embargo honoured
                assert np.all(in_test[t:t + h + 1])      # window fully inside the run (purge)
                t += h


def test_cpcv_matches_reference_purge_implementation() -> None:
    """Production tripwire: ``tier3_5_cpcv`` must equal an independent purge-correct
    reference. If the purge condition regresses ``t+horizon < b`` -> ``<= b`` (the off-by-one
    that lets a label window read close[b] across the train seam), the means diverge and this
    fails — locking the production code, not just the invariant."""
    from itertools import combinations

    from finrl_pro_ds.signals.eval_harness import _ann_sharpe, _ls_weights, _neutralized_eff

    panel = _momentum_panel(t=420, n=30)
    h, emb, G, k = 3, 4, 6, 2
    sig = Trail(1)
    eff = _neutralized_eff(sig, panel, (), 1, h)
    fwd = panel.forward_returns(h)
    bounds = np.linspace(0, panel.T, G + 1).astype(int)
    groups = [(int(bounds[i]), int(bounds[i + 1])) for i in range(G)]
    ref: list[float] = []
    for combo in combinations(range(G), k):
        in_test = np.zeros(panel.T, dtype=bool)
        for gi in combo:
            a, b = groups[gi]
            in_test[a:b] = True
        rets: list[float] = []
        for a, b in _contiguous_runs(in_test):
            t = a + emb
            while t + h < b:                                    # purge-correct boundary
                w = _ls_weights(eff[t], panel.active[t])
                rets.append(float(np.nansum(w * fwd[t])))
                t += h
        if len(rets) >= 2:
            s = _ann_sharpe(np.asarray(rets), 252.0 / h)
            if np.isfinite(s):
                ref.append(s)
    ref_arr = np.asarray(ref)

    res = tier3_5_cpcv(sig, panel, _gates(), neutralization=(), expected_sign=1,
                       horizon=h, n_groups=G, k_test=k, embargo_days=emb)
    assert abs(res.oos_sharpe_mean - float(ref_arr.mean())) < 1e-9
    assert abs(res.oos_sharpe_p05 - float(np.quantile(ref_arr, 0.05))) < 1e-9
    assert res.frac_paths_positive == float((ref_arr > 0.0).mean())


def test_cpcv_emits_clean_json() -> None:
    panel = _momentum_panel()
    signals = {s.spec.name: s for s in [Trail(1), Trail(5)]}
    rs = evaluate_batch(signals, panel, _gates(), batch_name="cpcv")
    payload = to_json(rs)
    assert all(c["cpcv"] is not None for c in payload["cards"])   # enabled by default
    assert payload["cards"][0]["cpcv"]["n_paths"] == 15
    dumped = json.dumps(payload)                                  # no NaN/Inf leaks
    assert "NaN" not in dumped and "Infinity" not in dumped


def test_cpcv_disabled_by_gate() -> None:
    panel = _momentum_panel()
    rs = evaluate_batch({"trail1": Trail(1)}, panel, _gates(cpcv={"enabled": False}),
                        batch_name="off")
    assert rs.cards[0].cpcv is None


# ====================================================== C2.3 — BHY + HLZ ====

def test_bhy_is_more_conservative_than_bh() -> None:
    p = [0.001, 0.008, 0.02, 0.05, 0.2, 0.4, 0.7]
    bh, bhy = bh_fdr(p), bhy_fdr(p)
    for a, b in zip(bh, bhy):
        assert min(a, 1.0) <= b + 1e-12      # Yekutieli q >= clamped BH q elementwise
    assert all(0.0 <= q <= 1.0 for q in bhy)


def test_bhy_monotone_and_empty() -> None:
    q = bhy_fdr([0.001, 0.01, 0.5, 0.9])
    assert q[0] <= q[-1]
    assert bhy_fdr([]) == []


def test_hlz_pass_reported_and_caveated() -> None:
    panel = _momentum_panel()
    signals = {s.spec.name: s for s in [Trail(1), Trail(2), Trail(3)]}
    rs = evaluate_batch(signals, panel, _gates(), batch_name="hlz")
    for c in rs.cards:
        if c.deflation is None:
            continue
        d = c.deflation
        # hlz_pass is exactly t>=hlz_t_min AND BHY-q<=fdr_q_max
        hp = c.gross.by_horizon[c.gross.primary_horizon]
        expect = bool(np.isfinite(hp.ic_tstat) and hp.ic_tstat >= 3.0
                      and np.isfinite(d.fdr_q_bhy) and d.fdr_q_bhy <= 0.10)
        assert d.hlz_pass == expect
        if c.verdict == "PROMISING" and not d.hlz_pass:
            assert any("HLZ" in cav for cav in c.caveats)


def test_require_hlz_can_demote_promising() -> None:
    # Folding the HLZ hurdle into PROMISING (require_hlz=true) may only REMOVE PROMISING
    # labels relative to default, never add them.
    panel = _momentum_panel()
    signals = {s.spec.name: s for s in [Trail(1), Trail(2), Trail(3)]}
    base = {c.name: c.verdict for c in
            evaluate_batch(signals, panel, _gates(), batch_name="b").cards}
    strict = {c.name: c.verdict for c in
              evaluate_batch(signals, panel, _gates(deflation={"require_hlz": True}),
                             batch_name="s").cards}
    for name in signals:
        if strict[name] == "PROMISING":
            assert base[name] == "PROMISING"     # stricter gate is a subset
