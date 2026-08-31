import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, r'C:\FinRL\FinRL-Pro_DS')
from finrl_pro_ds.crypto.eval.statistics import block_bootstrap_sharpe_ci

src = open(r'C:\FinRL\FinRL-Pro_DS\scripts\research\vix_voltarget_eval.py', encoding='utf-8').read()
exec(src.split('def main')[0])
from pathlib import Path as _P
VIXC = _P('C:/FinRL/FinRL-Pro_DS/data/raw/cross_asset_panel/n4_vix_yf.parquet')

df = pd.read_parquet(VIXC)
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date').sort_index()
# `build` is injected by the exec() of vix_voltarget_eval.py above (defined at its
# line 48, ahead of `def main`), so ruff cannot see the binding. Verified present.
n1, n2, gross, w = build(df, use_signal=True)  # noqa: F821

N = 22
print("Multiplicity-adjusted confidence intervals (block bootstrap, 20k draws)")
for lbl, s in [('gross', gross), ('net@3%', n1)]:
    for a, nm in [(0.05, '95%'), (0.05 / N, 'Bonferroni/22')]:
        d = block_bootstrap_sharpe_ci(s.dropna().tolist(), block=21, n_boot=20000,
                                      alpha=a, periods_per_year=252)
        print(f"  {lbl:8s} {nm:14s} CI [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]"
              f"   P(SR<=0)={d['p_sharpe_lt_0']:.4f}")

r = n1.dropna().to_numpy()
n = len(r)
sr = r.mean() / r.std(ddof=1)
g1 = float(stats.skew(r))
g2 = float(stats.kurtosis(r, fisher=False))
e = 0.5772156649
sr0 = np.sqrt(1.0 / n) * ((1 - e) * stats.norm.ppf(1 - 1.0 / N)
                          + e * stats.norm.ppf(1 - 1.0 / (N * np.e)))
dsr = stats.norm.cdf(((sr - sr0) * np.sqrt(n - 1))
                     / np.sqrt(1 - g1 * sr + ((g2 - 1) / 4.0) * sr ** 2))
print(f"\nDeflated Sharpe (Bailey/Lopez de Prado), n_trials={N}")
print(f"  net@3% daily SR {sr:.5f}  skew {g1:+.2f}  kurtosis {g2:.1f}  n={n}")
print(f"  SR* hurdle annualised {sr0*np.sqrt(252):+.4f}   observed annualised {sr*np.sqrt(252):+.4f}")
print(f"  DSR = {dsr:.4f}")
