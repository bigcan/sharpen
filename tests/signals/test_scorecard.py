"""Unit tests: end-to-end batch evaluation, deflation, ranking, and emit."""
from __future__ import annotations

import json

import numpy as np

from finrl_pro_ds.signals import (
    Gates,
    SignalSpec,
    evaluate_batch,
    to_json,
    to_markdown,
    write_scorecard,
)
from finrl_pro_ds.signals.features import Panel


class Trail:
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


class VolNoise:
    """log-volume — causal but unrelated to forward returns (a null signal)."""

    spec = SignalSpec(name="volnoise", hypothesis="volume predicts (it does not)",
                      family="technical", expected_sign=1, horizons=(1, 5), neutralization=())

    def compute(self, panel: Panel) -> np.ndarray:
        return np.log(np.where(panel.volume > 0, panel.volume, np.nan))


class Leaky:
    spec = SignalSpec(name="leaky", hypothesis="peeks ahead", family="technical",
                      expected_sign=1, horizons=(1,), neutralization=())

    def compute(self, panel: Panel) -> np.ndarray:
        c = panel.close
        out = np.full(c.shape, np.nan)
        out[:-1] = c[1:] / c[:-1] - 1.0
        return out


def _momentum_panel(t: int = 500, n: int = 40, rho: float = 0.45, seed: int = 0) -> Panel:
    rng = np.random.default_rng(seed)
    eps = 0.01 * rng.standard_normal((t, n))
    ret = np.zeros((t, n))
    for i in range(1, t):
        ret[i] = rho * ret[i - 1] + eps[i]
    close = np.exp(np.cumsum(ret, axis=0) + rng.uniform(3.0, 5.0, size=n))
    open_ = close * np.exp(0.0005 * rng.standard_normal((t, n)))
    high = np.maximum(open_, close) * np.exp(np.abs(0.001 * rng.standard_normal((t, n))))
    low = np.minimum(open_, close) * np.exp(-np.abs(0.001 * rng.standard_normal((t, n))))
    volume = rng.uniform(1e5, 1e7, size=(t, n))
    dates = (np.datetime64("2012-01-02")
             + np.arange(t) * np.timedelta64(1, "D")).astype("datetime64[ns]")
    return Panel(dates, tuple(f"M{i:03d}" for i in range(n)), open_, high, low, close,
                 volume, np.ones((t, n), bool), close * volume,
                 rng.integers(0, 5, size=n), {"survivorship_free": False, "source": "synthetic"})


def _gates() -> Gates:
    return Gates.from_dict({
        "universe": {"min_names_per_day": 4},
        "coverage": {"min_days": 150},
        "gross_power": {"horizons": [1, 5], "primary_horizon": 1},
    })


def test_batch_ranks_alpha_above_noise_and_excludes_leak() -> None:
    panel = _momentum_panel()
    signals = {s.spec.name: s for s in [Trail(1), Trail(5), VolNoise(), Leaky()]}
    rs = evaluate_batch(signals, panel, _gates(), batch_name="unit")

    assert rs.n_trials == 3                          # 3 causal signals deflated; leaky excluded
    by_name = {c.name: c for c in rs.cards}
    assert by_name["leaky"].verdict == "GATE_FAIL"
    assert not by_name["leaky"].hygiene.causal
    assert rs.cards[-1].name == "leaky"              # GATE_FAIL sinks to the end

    evaluated = [c for c in rs.cards if c.verdict != "GATE_FAIL"]
    assert evaluated[0].name == "trail1"             # strongest momentum alpha ranks #1
    assert evaluated[0].rank_key >= evaluated[-1].rank_key
    # the null signal must not out-rank the real alpha
    assert by_name["trail1"].rank_key >= by_name["volnoise"].rank_key
    for c in evaluated:
        assert c.verdict in {"PROMISING", "LOGGED"}


def test_scorecard_caveats_and_emit(tmp_path) -> None:
    panel = _momentum_panel()
    signals = {s.spec.name: s for s in [Trail(1), VolNoise()]}
    rs = evaluate_batch(signals, panel, _gates(), batch_name="emit")

    # survivorship caveat is stamped on evaluated cards (synthetic panel = not survivorship-free)
    top = [c for c in rs.cards if c.verdict != "GATE_FAIL"][0]
    assert any("UPPER BOUND" in cav for cav in top.caveats)

    md = to_markdown(rs)
    assert "Signal Scorecard — emit" in md and "trail1" in md
    payload = to_json(rs)
    assert payload["n_trials"] == 2
    # JSON must be valid (no NaN/Inf leaking through)
    dumped = json.dumps(payload)
    assert "NaN" not in dumped and "Infinity" not in dumped

    jp, mp = write_scorecard(rs, tmp_path / "batch")
    assert jp.exists() and mp.exists()
    assert json.loads(jp.read_text())["batch_name"] == "emit"


def test_gates_yaml_loads() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    g = Gates.from_yaml(root / "configs" / "signal_eval.gates.yaml")
    assert g.primary_horizon == 5
    assert g.horizons == (1, 5, 10, 21, 63)
    assert g.survivorship_free_required is True
    assert g.neutralization == ("winsor", "zscore", "sector", "size")
