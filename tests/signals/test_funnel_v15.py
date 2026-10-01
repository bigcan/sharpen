"""crucible-v15.0 funnel tripwires (sharpen/signals), deep audit 2026-09-30. Each fails pre-v15.

* the PRE-REGISTERED neutralization controls (gates `neutralization.controls` / `winsor_pct`) were parsed
  and never applied — every scorecard used the spec's own steps (no `size`) at the default winsorization;
* the block-bootstrap p-value could be exactly 0, which BH/BHY leave at q = 0 for ANY family size;
* PSR / MinTRL used the raw count on overlapping h-day labels (the t-stat and DSR already use HAC);
* the Tier-0 truncation tripwire never probed the first quarter of the panel, so a warm-up leak passed.
"""
from __future__ import annotations

import numpy as np

from sharpen.signals import eval_harness as eh
from sharpen.signals import scorecard as sc
from sharpen.signals._ic import block_bootstrap_mean
from sharpen.signals.features import make_synthetic_panel
from sharpen.signals.gates import Gates
from sharpen.signals.spec import SignalSpec


class _Sig:
    def __init__(self, fn, *, name="t", neutralization=("winsor", "zscore", "sector")):
        self._fn = fn
        self.spec = SignalSpec(name=name, hypothesis="h", family="technical", expected_sign=1,
                               neutralization=neutralization)

    def compute(self, panel):
        return self._fn(panel)


def test_declared_gate_controls_are_applied(monkeypatch) -> None:
    panel = make_synthetic_panel(T=300, N=30, seed=1)
    seen = {}
    real = sc.compute_scores

    def spy(sig, p, ns, **kw):
        seen["ns"], seen["kw"] = ns, kw
        return real(sig, p, ns, **kw)

    monkeypatch.setattr(sc, "compute_scores", spy)
    gates = Gates.from_dict({"coverage": {"min_days": 100}, "universe": {"min_names_per_day": 5}})
    card = sc.evaluate_signal(_Sig(lambda p: np.log(p.adv_usd)), panel, gates)
    assert card.hygiene.passed, card.hygiene.reasons
    assert set(gates.neutralization) <= set(seen["ns"]) and "size" in seen["ns"]
    assert seen["kw"].get("winsor_pct") == gates.winsor_pct


def test_bootstrap_p_is_never_exactly_zero() -> None:
    s = 0.05 + 0.001 * np.random.default_rng(0).standard_normal(400)     # overwhelmingly positive
    out = block_bootstrap_mean(s, block=10, n_boot=999, seed=0)
    assert out is not None and out["p_le_0"] >= 1.0 / 1000.0


def test_psr_and_mintrl_use_the_effective_count() -> None:
    rng = np.random.default_rng(3)
    # an MA(20) null-ish IC series (overlapping 21-day labels) with a small positive mean
    e = rng.standard_normal(1500)
    series = 0.02 + np.convolve(e, np.ones(21) / 21.0, mode="valid") * 0.1
    psr_h1, mintrl_h1 = eh._psr_mintrl_hac(series, 1)
    psr_h21, mintrl_h21 = eh._psr_mintrl_hac(series, 21)
    assert psr_h21 <= psr_h1 and mintrl_h21 > mintrl_h1                 # overlap costs evidence


def test_tier0_probes_the_warm_up() -> None:
    """A trailing mean back-filled over its warm-up reads FUTURE bars only in the first window rows."""
    panel = make_synthetic_panel(T=800, N=10, seed=2)
    import pandas as pd

    def bfilled(p):
        return pd.DataFrame(p.close).rolling(120).mean().bfill().to_numpy()

    ok, why = eh.assert_causal(_Sig(bfilled), panel, n_probes=8, seed=0)
    assert not ok, "a leak confined to the warm-up must be probed"
