"""crucible-v14.0 statistical tripwires (pre-release audit 2026-09-18).

1. Overlapping h-bar labels: the IC t-stat used sqrt(n_days) as if daily ICs were independent. On a
   no-edge persistent signal at h=21 the null t had sd ~4, so t>=3 fired ~27% of the time.
2. BH/BHY used m = batch size instead of the declared family size.
3. Tier-5 "residual Sharpe" was the Sharpe of the OLS residual, which has mean exactly 0.
"""
from __future__ import annotations

import numpy as np

from sharpen.signals._ic import bh_fdr, cross_sectional_ic, hac_effective_n


def _null_tstat(h: int, seed: int, T: int = 1200, N: int = 60) -> float:
    rng = np.random.default_rng(seed)
    sig = np.zeros((T, N))
    e = rng.standard_normal((T, N))
    for t in range(1, T):                                   # persistent signal, rho = 0.98
        sig[t] = 0.98 * sig[t - 1] + e[t]
    r = rng.standard_normal((T + h, N)) * 0.01               # iid returns: NO edge
    fwd = np.stack([r[t + 1:t + 1 + h].sum(axis=0) for t in range(T)])
    return cross_sectional_ic(sig, fwd, overlap=h).ic_tstat


def test_overlap_one_is_the_iid_formula() -> None:
    x = np.random.default_rng(0).standard_normal(500)
    assert hac_effective_n(x, 0) == 500.0


def test_null_tstat_is_calibrated_under_overlap() -> None:
    ts = np.array([_null_tstat(21, s) for s in range(40)])
    # IID sqrt(n) gave sd ~4 here; the HAC count must bring it near 1 and keep t>=3 rare.
    assert ts.std() < 1.8
    assert np.mean(ts >= 3.0) <= 0.05


def test_hac_never_inflates_the_count() -> None:
    x = np.tile([1.0, -1.0], 200)                            # strong NEGATIVE autocorrelation
    assert hac_effective_n(x, 5) <= x.size


def test_bh_padding_to_declared_family() -> None:
    p = [0.004]
    padded = bh_fdr(p + [1.0] * 49)[:1]
    assert abs(padded[0] - 0.2) < 1e-12                      # 0.004 * 50 / 1
    assert abs(bh_fdr(p)[0] - 0.004) < 1e-12                 # the batch-of-1 number it replaced


def test_tier5_residual_sharpe_is_not_identically_zero() -> None:
    from sharpen.signals.eval_harness import FactorBook, _daily_ls_returns, _neutralized_eff, \
        tier5_orthogonality
    from tests.signals.test_eval_tiers import Trail, _momentum_panel

    p = _momentum_panel()
    eff = _neutralized_eff(Trail(1), p, (), 1, 1)
    sd, sr = _daily_ls_returns(eff, p)
    rng = np.random.default_rng(0)
    book = FactorBook(sd, ("RND1", "RND2"), rng.standard_normal((len(sd), 2)) * 1e-6)
    o = tier5_orthogonality(Trail(1), p, book, neutralization=(), expected_sign=1, horizon=1)
    own = float(sr.mean() / sr.std(ddof=1) * np.sqrt(252))
    assert abs(o.residual_sharpe) > 0.1
    assert abs(o.residual_sharpe - own) < 0.05 * abs(own)   # orthogonal factors leave the alpha
