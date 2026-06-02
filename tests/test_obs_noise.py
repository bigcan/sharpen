"""Unit tests for ``finrl_pro_ds.eval.obs_noise`` (v2.7-B B1, S553-cont-25).

Covers the pure core: the OHLC noise function (OBSNOISE-1..4 + MATH-OBS-1),
sigma<->bps mapping (MATH-OBS-2), the (fold,sigma,k) seed derivation
(MATH-OBS-5), the fold aggregator (ADR-6 + the load-bearing MATH-OBS-4 abs-
before-quantile MDD ordering), the gate resolver (ADR-6), and the noised-parquet
writer. The impure orchestrator ``run_obs_noise_stage`` (B2) is exercised by the
B2 integration smoke test.

Architecture: ``.agent/artifacts/protocol_v27_b_obs_noise_stage_3_5_architecture.md``
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finrl_pro_ds.eval.obs_noise import (  # noqa: E402
    DEFAULT_NOISE_BASE_SEED,
    OHLC_COLS,
    PF_CAP,
    FoldNoiseResult,
    InvariantViolation,
    NoiseSpec,
    ObsNoiseVerdict,
    aggregate_fold,
    apply_ohlc_noise,
    bps_to_sigma,
    derive_noise_seed,
    resolve_obs_noise_gate,
    sigma_to_bps,
    write_noised_parquet,
)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _synthetic_ohlcv(n: int = 240, start: str = "2025-09-01", freq: str = "1min",
                     seed: int = 7, base: float = 100.0) -> pd.DataFrame:
    """A clean, strictly-positive, well-ordered 1-min OHLCV frame."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start=start, periods=n, freq=freq, tz="UTC")
    close = base * np.cumprod(1.0 + rng.normal(0.0, 0.001, size=n))
    open_ = np.concatenate([[base], close[:-1]])
    span = np.abs(rng.normal(0.0, 0.002, size=n)) * close
    high = np.maximum(open_, close) + span
    low = np.minimum(open_, close) - span
    low = np.clip(low, 1e-6, None)
    volume = rng.uniform(1.0, 100.0, size=n)
    return pd.DataFrame({
        "timestamp": ts, "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    })


SIGMA_50BPS = 0.005
SIGMA_10BPS = 0.001
MID = "2025-09-01 02:00:00+00:00"  # ~120 of 240 bars masked at default freq


# ---------------------------------------------------------------------------
# sigma <-> bps (MATH-OBS-2)
# ---------------------------------------------------------------------------


def test_sigma_to_bps_mapping():
    assert sigma_to_bps(0.001) == pytest.approx(10.0)
    assert sigma_to_bps(0.005) == pytest.approx(50.0)
    assert bps_to_sigma(10.0) == pytest.approx(0.001)
    assert bps_to_sigma(50.0) == pytest.approx(0.005)
    assert NoiseSpec(0.005, "50bps", 10).bps == pytest.approx(50.0)


def test_close_noise_relative_std_matches_sigma():
    """MATH-OBS-2 / verdict §1: per-bar close rel-std ~= sigma (lognormal)."""
    n = 40000
    ts = pd.date_range("2025-09-01", periods=n, freq="1min", tz="UTC")
    df = pd.DataFrame({
        "timestamp": ts, "open": 100.0, "high": 100.0, "low": 100.0,
        "close": 100.0, "volume": 1.0,
    })
    out = apply_ohlc_noise(df, sigma=SIGMA_50BPS, seed=1, noise_start="2025-09-01")
    rel = out["close"].to_numpy() / 100.0 - 1.0
    assert rel.std() == pytest.approx(SIGMA_50BPS, rel=0.05)


# ---------------------------------------------------------------------------
# apply_ohlc_noise — OBSNOISE-1..4 + MATH-OBS-1
# ---------------------------------------------------------------------------


def test_apply_ohlc_noise_ordering():
    """OBSNOISE-1: H>=max(O,L,C), L<=min(O,H,C) on every masked row."""
    df = _synthetic_ohlcv()
    out = apply_ohlc_noise(df, SIGMA_50BPS, seed=11, noise_start=MID)
    o = out["open"].to_numpy()
    h = out["high"].to_numpy()
    lo = out["low"].to_numpy()
    c = out["close"].to_numpy()
    assert np.all(h >= np.maximum.reduce([o, lo, c]) - 1e-9)
    assert np.all(lo <= np.minimum.reduce([o, h, c]) + 1e-9)


def test_apply_ohlc_noise_sigma0_identity():
    """OBSNOISE-3: sigma=0 -> byte-identical copy."""
    df = _synthetic_ohlcv()
    out = apply_ohlc_noise(df, 0.0, seed=11, noise_start=MID)
    pd.testing.assert_frame_equal(out, df)
    assert out is not df  # a copy, not the same object


def test_apply_ohlc_noise_pre_region_byte_identical():
    """OBSNOISE-2: rows with timestamp < noise_start are unchanged."""
    df = _synthetic_ohlcv()
    out = apply_ohlc_noise(df, SIGMA_50BPS, seed=11, noise_start=MID)
    ns = pd.Timestamp(MID)
    pre = pd.to_datetime(df["timestamp"], utc=True) < ns
    pd.testing.assert_frame_equal(out.loc[pre], df.loc[pre])
    # And the post region actually changed (sanity: noise did something).
    assert not out.loc[~pre, "close"].equals(df.loc[~pre, "close"])


def test_apply_ohlc_noise_volume_untouched():
    """ADR-8: volume passes through unchanged."""
    df = _synthetic_ohlcv()
    out = apply_ohlc_noise(df, SIGMA_50BPS, seed=11, noise_start=MID)
    pd.testing.assert_series_equal(out["volume"], df["volume"])


def test_apply_ohlc_noise_deterministic_same_seed():
    """OBSNOISE-4: same seed -> identical array; different seed -> different."""
    df = _synthetic_ohlcv()
    a = apply_ohlc_noise(df, SIGMA_50BPS, seed=11, noise_start=MID)
    b = apply_ohlc_noise(df, SIGMA_50BPS, seed=11, noise_start=MID)
    pd.testing.assert_frame_equal(a, b)
    c = apply_ohlc_noise(df, SIGMA_50BPS, seed=12, noise_start=MID)
    assert not a["close"].equals(c["close"])


def test_apply_ohlc_noise_no_future_dependence():
    """OBSNOISE-4: changing a LATER bar's input never alters earlier bars' noise.

    The factor for masked row i is drawn positionally from (seed, i), so a future
    change cannot leak back into an earlier bar's perturbed value (no look-ahead).
    """
    df = _synthetic_ohlcv()
    out1 = apply_ohlc_noise(df, SIGMA_50BPS, seed=11, noise_start=MID)
    df2 = df.copy(deep=True)
    last = len(df2) - 1
    for col in OHLC_COLS:  # perturb only the final bar's *input*
        df2.loc[last, col] = float(df2.loc[last, col]) * 1.5
    out2 = apply_ohlc_noise(df2, SIGMA_50BPS, seed=11, noise_start=MID)
    pd.testing.assert_frame_equal(
        out1.iloc[:last].reset_index(drop=True),
        out2.iloc[:last].reset_index(drop=True),
    )


def test_apply_ohlc_noise_positivity_no_nan_large_sigma():
    """MATH-OBS-1 / DATA-CLEAN: large sigma keeps OHLC finite & strictly positive."""
    df = _synthetic_ohlcv()
    out = apply_ohlc_noise(df, sigma=0.05, seed=11, noise_start="2025-09-01")
    block = out[list(OHLC_COLS)].to_numpy()
    assert np.all(np.isfinite(block))
    assert np.all(block > 0.0)


def test_apply_ohlc_noise_does_not_mutate_source():
    df = _synthetic_ohlcv()
    snapshot = df.copy(deep=True)
    _ = apply_ohlc_noise(df, SIGMA_50BPS, seed=11, noise_start=MID)
    pd.testing.assert_frame_equal(df, snapshot)


def test_apply_ohlc_noise_rejects_negative_sigma():
    with pytest.raises(ValueError, match="sigma"):
        apply_ohlc_noise(_synthetic_ohlcv(), -0.001, seed=1, noise_start=MID)


def test_apply_ohlc_noise_rejects_missing_columns():
    df = _synthetic_ohlcv().drop(columns=["high"])
    with pytest.raises(ValueError, match="missing required columns"):
        apply_ohlc_noise(df, SIGMA_50BPS, seed=1, noise_start=MID)


def test_apply_ohlc_noise_rejects_unsorted_timestamps():
    df = _synthetic_ohlcv().iloc[::-1].reset_index(drop=True)
    with pytest.raises(InvariantViolation, match="sorted by timestamp"):
        apply_ohlc_noise(df, SIGMA_50BPS, seed=1, noise_start="2025-09-01")


def test_apply_ohlc_noise_noise_start_past_end_returns_copy():
    df = _synthetic_ohlcv()
    out = apply_ohlc_noise(df, SIGMA_50BPS, seed=1, noise_start="2030-01-01")
    pd.testing.assert_frame_equal(out, df)


# ---------------------------------------------------------------------------
# derive_noise_seed (MATH-OBS-5)
# ---------------------------------------------------------------------------


def test_derive_noise_seed_deterministic_and_distinct():
    s1 = derive_noise_seed(0, 0, 0)
    s2 = derive_noise_seed(0, 0, 0)
    assert s1 == s2 and isinstance(s1, int)
    # Distinct across each axis.
    assert derive_noise_seed(0, 0, 0) != derive_noise_seed(0, 0, 1)
    assert derive_noise_seed(0, 0, 0) != derive_noise_seed(1, 0, 0)
    assert derive_noise_seed(0, 0, 0) != derive_noise_seed(0, 1, 0)


def test_derive_noise_seed_is_seedsequence_not_hash():
    """MATH-OBS-5: the seed equals the SeedSequence-derived integer — NOT a
    process-salted ``hash(tuple)``. This is what makes the noise reproducible
    across processes (the orchestrator runs cells in a ProcessPool)."""
    expected = int(
        np.random.SeedSequence(
            [DEFAULT_NOISE_BASE_SEED, 2, 1, 5]
        ).generate_state(1, dtype=np.uint32)[0]
    )
    assert derive_noise_seed(2, 1, 5) == expected


def test_derive_noise_seed_respects_base_seed():
    assert derive_noise_seed(0, 0, 0, base_seed=1) != derive_noise_seed(
        0, 0, 0, base_seed=2
    )


# ---------------------------------------------------------------------------
# aggregate_fold (ADR-6 / MATH-OBS-4)
# ---------------------------------------------------------------------------


def _spec(label: str = "10bps", sigma: float = SIGMA_10BPS, n: int = 10) -> NoiseSpec:
    return NoiseSpec(sigma=sigma, label=label, n_seeds=n)


def test_aggregate_fold_quantiles_and_ratio():
    pf_seeds = list(np.linspace(1.0, 2.0, 11))  # median 1.5
    mdd_seeds = list(np.linspace(-0.10, -0.02, 11))
    r = aggregate_fold(0, _spec(), pf_nominal=2.0, mdd_nominal=-0.04,
                       pf_seeds=pf_seeds, mdd_seeds=mdd_seeds)
    assert r.pf_q50 == pytest.approx(1.5)
    assert r.pf_ratio == pytest.approx(1.5 / 2.0)
    assert r.anomaly is None
    # absmdd_q95 == q95 of the magnitudes (abs FIRST).
    assert r.absmdd_q95 == pytest.approx(np.quantile(np.abs(mdd_seeds), 0.95))


def test_aggregate_fold_mdd_abs_before_quantile_not_after():
    """MATH-OBS-4 (load-bearing): the gate input is q95(|mdd|), NOT |q95(mdd)|.

    A worst-case fold where the deep drawdowns sit in the lower tail: the signed
    q95 returns the least-negative (rosy) value, so ``|q95(signed)|`` would
    drastically *understate* the risk. The abs-first ordering must report the
    deep tail.
    """
    mdd_seeds = [-0.01, -0.02, -0.03, -0.05, -0.08, -0.12, -0.18, -0.25, -0.30]
    mdd_nominal = -0.02
    r = aggregate_fold(0, _spec(), pf_nominal=1.5, mdd_nominal=mdd_nominal,
                       pf_seeds=[1.5] * 9, mdd_seeds=mdd_seeds)

    abs_first = np.quantile(np.abs(mdd_seeds), 0.95)          # ~0.28 (worst)
    wrong_rosy = abs(np.quantile(mdd_seeds, 0.95))            # ~0.014 (rosy/defeated)
    assert abs_first > 10 * wrong_rosy  # the two orderings genuinely diverge here

    assert r.absmdd_q95 == pytest.approx(abs_first)
    expected_pp = (abs_first - abs(mdd_nominal)) * 100.0
    assert r.mdd_degradation_pp == pytest.approx(expected_pp)
    # The defeated formula would have produced a *negative* (rosy) degradation.
    assert r.mdd_degradation_pp > 0.0
    assert r.mdd_degradation_pp != pytest.approx((wrong_rosy - abs(mdd_nominal)) * 100.0)


def test_aggregate_fold_signed_mdd_quantiles_reported():
    """Signed q05/q50/q95 are surfaced for reporting (in [-1,0])."""
    mdd_seeds = list(np.linspace(-0.20, -0.01, 20))
    r = aggregate_fold(0, _spec(), 1.5, -0.05, [1.5] * 20, mdd_seeds)
    assert r.mdd_q05 <= r.mdd_q50 <= r.mdd_q95 <= 0.0


def test_aggregate_fold_anomaly_pf_nominal_below_one():
    r = aggregate_fold(0, _spec(), pf_nominal=0.8, mdd_nominal=-0.05,
                       pf_seeds=[0.7] * 5, mdd_seeds=[-0.05] * 5)
    assert r.anomaly == "pf_nominal<1.0"


def test_aggregate_fold_anomaly_pf_nominal_cap():
    r = aggregate_fold(0, _spec(), pf_nominal=PF_CAP, mdd_nominal=-0.05,
                       pf_seeds=[PF_CAP] * 5, mdd_seeds=[-0.05] * 5)
    assert r.anomaly == "pf_nominal==CAP"


def test_aggregate_fold_empty_seeds_raises():
    with pytest.raises(ValueError, match="empty seed arrays"):
        aggregate_fold(0, _spec(), 1.5, -0.05, [], [])


# ---------------------------------------------------------------------------
# resolve_obs_noise_gate (ADR-6)
# ---------------------------------------------------------------------------


def _gates(**overrides) -> dict:
    g = {
        "obs_noise_pf_floor_10bps": 0.85,
        "obs_noise_pf_floor_50bps": 0.70,
        "obs_noise_mdd_buffer_pp_10bps": 0.3,
        "obs_noise_mdd_buffer_pp_50bps": 0.7,
        "obs_noise_required_min_folds": 4,
    }
    g.update(overrides)
    return g


def _result(fold: int, label: str, pf_ratio: float, mdd_degr_pp: float,
            anomaly=None) -> FoldNoiseResult:
    """Minimal FoldNoiseResult carrying only the gate-relevant fields."""
    return FoldNoiseResult(
        fold_idx=fold, sigma_label=label,
        pf_nominal=(0.8 if anomaly == "pf_nominal<1.0" else 1.6),
        mdd_nominal=-0.04,
        pf_seeds=(1.5,), mdd_seeds=(-0.04,),
        pf_q05=1.4, pf_q50=1.5, pf_q95=1.6,
        mdd_q05=-0.06, mdd_q50=-0.04, mdd_q95=-0.03,
        absmdd_q95=0.05,
        pf_ratio=pf_ratio, mdd_degradation_pp=mdd_degr_pp, anomaly=anomaly,
    )


def _grid(pf_ratios_10, pf_ratios_50, mdd_10, mdd_50, anomalies=None):
    """Build a folds x {10bps,50bps} result list from per-fold scalars."""
    anomalies = anomalies or [None] * len(pf_ratios_10)
    out = []
    for f, (p10, p50, m10, m50, an) in enumerate(
        zip(pf_ratios_10, pf_ratios_50, mdd_10, mdd_50, anomalies)
    ):
        out.append(_result(f, "10bps", p10, m10, anomaly=an))
        out.append(_result(f, "50bps", p50, m50, anomaly=an))
    return out


def test_resolve_gate_pass():
    results = _grid(
        pf_ratios_10=[0.95, 0.92, 0.90, 0.94],
        pf_ratios_50=[0.80, 0.78, 0.75, 0.79],
        mdd_10=[0.10, 0.20, 0.15, 0.21],
        mdd_50=[0.40, 0.55, 0.50, 0.60],
    )
    v = resolve_obs_noise_gate(results, _gates())
    assert isinstance(v, ObsNoiseVerdict)
    assert v.decision == "PASS"
    assert v.n_folds == 4 and v.n_folds_graded == 4
    assert v.per_sigma["10bps"]["pf_pass"] and v.per_sigma["10bps"]["mdd_pass"]
    assert v.per_sigma["10bps"]["worst_pf_ratio"] == pytest.approx(0.90)
    assert v.per_sigma["10bps"]["worst_fold"] == 2


def test_resolve_gate_fail_on_pf_floor():
    """Worst fold governs: one fold below the 10bps PF floor -> FAIL."""
    results = _grid(
        pf_ratios_10=[0.95, 0.92, 0.60, 0.94],  # fold 2 below 0.85
        pf_ratios_50=[0.80, 0.78, 0.75, 0.79],
        mdd_10=[0.10, 0.20, 0.15, 0.21],
        mdd_50=[0.40, 0.55, 0.50, 0.60],
    )
    v = resolve_obs_noise_gate(results, _gates())
    assert v.decision == "FAIL"
    assert v.per_sigma["10bps"]["pf_pass"] is False
    assert v.per_sigma["10bps"]["worst_fold"] == 2


def test_resolve_gate_fail_on_mdd_buffer():
    """One fold's MDD degradation exceeds the 50bps buffer -> FAIL."""
    results = _grid(
        pf_ratios_10=[0.95, 0.92, 0.90, 0.94],
        pf_ratios_50=[0.80, 0.78, 0.75, 0.79],
        mdd_10=[0.10, 0.20, 0.15, 0.21],
        mdd_50=[0.40, 0.55, 0.50, 1.20],  # fold 3 above 0.7 buffer
    )
    v = resolve_obs_noise_gate(results, _gates())
    assert v.decision == "FAIL"
    assert v.per_sigma["50bps"]["mdd_pass"] is False
    assert v.per_sigma["50bps"]["worst_mdd_fold"] == 3


def test_resolve_gate_unknown_insufficient_folds():
    results = _grid(
        pf_ratios_10=[0.95, 0.92],
        pf_ratios_50=[0.80, 0.78],
        mdd_10=[0.10, 0.20],
        mdd_50=[0.40, 0.55],
    )
    v = resolve_obs_noise_gate(results, _gates())  # required_min_folds=4, only 2
    assert v.decision == "UNKNOWN_INSUFFICIENT_FOLDS"
    assert v.n_folds_graded == 2


def test_resolve_gate_excludes_anomaly_folds():
    """A fold with pf_nominal<1.0 is excluded from grading; its terrible
    pf_ratio must not drag the verdict to FAIL."""
    results = _grid(
        pf_ratios_10=[0.95, 0.92, 0.90, 0.05],  # fold 3 ratio awful but anomalous
        pf_ratios_50=[0.80, 0.78, 0.75, 0.05],
        mdd_10=[0.10, 0.20, 0.15, 9.0],
        mdd_50=[0.40, 0.55, 0.50, 9.0],
        anomalies=[None, None, None, "pf_nominal<1.0"],
    )
    v = resolve_obs_noise_gate(results, _gates(obs_noise_required_min_folds=3))
    assert v.decision == "PASS"
    assert v.n_folds == 4 and v.n_folds_graded == 3
    assert v.per_sigma["10bps"]["worst_fold"] != 3  # the anomaly fold was excluded


def test_resolve_gate_missing_threshold_key_raises():
    results = _grid([0.95] * 4, [0.80] * 4, [0.1] * 4, [0.4] * 4)
    bad = _gates()
    del bad["obs_noise_pf_floor_50bps"]
    with pytest.raises(KeyError, match="obs_noise_pf_floor_50bps"):
        resolve_obs_noise_gate(results, bad)


# ---------------------------------------------------------------------------
# write_noised_parquet
# ---------------------------------------------------------------------------


def test_write_noised_parquet_roundtrip_and_deterministic_sha(tmp_path):
    df = _synthetic_ohlcv()
    src = tmp_path / "src.parquet"
    df.to_parquet(src)

    out1 = tmp_path / "n1.parquet"
    p1, sha1 = write_noised_parquet(src, SIGMA_50BPS, seed=11,
                                    noise_start=MID, out_path=out1)
    p2, sha2 = write_noised_parquet(src, SIGMA_50BPS, seed=11,
                                    noise_start=MID, out_path=tmp_path / "n2.parquet")
    assert p1 == out1
    assert sha1 == sha2  # same (source, sigma, seed, noise_start) -> identical bytes

    reloaded = pd.read_parquet(out1)
    ns = pd.Timestamp(MID)
    post = pd.to_datetime(reloaded["timestamp"], utc=True) >= ns
    assert not reloaded.loc[post, "close"].equals(df.loc[post.values, "close"])
    # Only the handler columns are retained.
    assert list(reloaded.columns) == [c for c in
                                       ("timestamp", "open", "high", "low", "close", "volume")
                                       if c in df.columns]


def test_write_noised_parquet_sigma0_preserves_content(tmp_path):
    """ADR-3: sigma=0 still writes, and the content matches the source."""
    df = _synthetic_ohlcv()
    src = tmp_path / "src.parquet"
    df.to_parquet(src)
    out = tmp_path / "n0.parquet"
    _, _ = write_noised_parquet(src, 0.0, seed=11, noise_start=MID, out_path=out)
    reloaded = pd.read_parquet(out)
    pd.testing.assert_frame_equal(
        reloaded.reset_index(drop=True), df.reset_index(drop=True)
    )


def test_write_noised_parquet_different_seed_different_sha(tmp_path):
    df = _synthetic_ohlcv()
    src = tmp_path / "src.parquet"
    df.to_parquet(src)
    _, sha_a = write_noised_parquet(src, SIGMA_50BPS, seed=1, noise_start=MID,
                                    out_path=tmp_path / "a.parquet")
    _, sha_b = write_noised_parquet(src, SIGMA_50BPS, seed=2, noise_start=MID,
                                    out_path=tmp_path / "b.parquet")
    assert sha_a != sha_b
