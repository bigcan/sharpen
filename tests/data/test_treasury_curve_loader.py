"""Treasury-curve loader DATA-CLEAN + earned manifest (N3; P1-04/P1-05).

The cached ^IRX 3m leg carries 7 negative prints (min -0.105). As the SHARED financing leg
(carry = tenor - financing), a single bad print inflates carry sleeve-wide, and the signal's
`np.isfinite` guard misses it (wrong-but-finite). These tests pin the point-local (causal)
repair, the EARNED manifest status, that exactly-0 (ZIRP) is preserved, and that the cleaned
curve still passes the rates-carry look-ahead tripwire.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sharpen.data import treasury_curve_loader as tcl
from sharpen.features import rates_carry as rc


def _raw_curve_frame(n: int = 300, neg: int = 0, extreme: int = 0, seed: int = 3) -> pd.DataFrame:
    """Synthetic raw curve (percent) with optional bad prints injected into the 3m leg."""
    idx = pd.bdate_range("2018-01-02", periods=n)
    rng = np.random.default_rng(seed)
    base = {"3m": 1.5, "y2": 2.0, "5y": 2.5, "10y": 3.0, "30y": 3.5}
    df = pd.DataFrame({k: v + rng.normal(0.0, 0.05, n) for k, v in base.items()}, index=idx)
    j = df.columns.get_loc("3m")
    for i in range(neg):
        df.iloc[10 + i, j] = -0.105                 # the ^IRX negative-print class
    for i in range(extreme):
        df.iloc[100 + i, j] = 99.0                  # decimal / blow-up error
    return df


def test_clean_curve_repairs_negative_and_extreme():
    clean, report = tcl.clean_curve(_raw_curve_frame(neg=2, extreme=1))
    s3 = clean["3m"].dropna()
    assert s3.min() >= 0.0 and s3.max() <= tcl.CURVE_YIELD_MAX_PCT
    assert report["3m"]["n_negative_repaired"] == 2
    assert report["3m"]["n_extreme_repaired"] == 1
    assert report["3m"]["n_repaired"] == 3
    assert report["10y"]["n_repaired"] == 0          # other legs untouched


def test_zero_yield_is_valid_not_repaired():
    """Exactly 0.0 (ZIRP) is economically valid and must NOT be repaired (only strictly < 0)."""
    df = _raw_curve_frame()
    df.iloc[20, df.columns.get_loc("3m")] = 0.0
    clean, report = tcl.clean_curve(df)
    assert report["3m"]["n_negative_repaired"] == 0
    assert clean["3m"].iloc[20] == 0.0


def test_load_with_manifest_earns_warn(tmp_path):
    """A few repairs over many obs → WARN (status DERIVED from the scan, never hardcoded);
    sidecar manifest written; the returned curve has no surviving negative financing leg."""
    _raw_curve_frame(neg=1).to_parquet(tmp_path / "treasury_curve.parquet")
    curve, man = tcl.load_treasury_curve_with_manifest(cache_dir=tmp_path, research_cache=None)
    assert curve["3m"].min() >= 0.0
    assert man["status"] == "WARN" and man["repaired_legs"] == ["3m"]
    assert man["per_tenor"]["3m"]["n_negative_repaired"] == 1
    assert (tmp_path / "treasury_curve.manifest.json").exists()


def test_load_with_manifest_earns_fail_on_corrupt_leg(tmp_path):
    """A leg with > CURVE_FAIL_REPAIR_FRAC of its obs repaired is FAIL (10/300 ≈ 3.3% > 2%)."""
    _raw_curve_frame(neg=10).to_parquet(tmp_path / "treasury_curve.parquet")
    _, man = tcl.load_treasury_curve_with_manifest(cache_dir=tmp_path, research_cache=None)
    assert man["status"] == "FAIL"
    assert man["worst_repair_frac"] > tcl.CURVE_FAIL_REPAIR_FRAC


def test_clean_curve_pass_status_on_clean_data(tmp_path):
    _raw_curve_frame().to_parquet(tmp_path / "treasury_curve.parquet")
    _, man = tcl.load_treasury_curve_with_manifest(cache_dir=tmp_path, research_cache=None)
    assert man["status"] == "PASS" and man["repaired_legs"] == []


def test_cleaned_curve_passes_rates_carry_causality():
    """The repair uses no future information (point-local), so the cleaned curve must still
    pass the rates-carry look-ahead tripwire — perturbing future yields cannot move past
    conviction."""
    df = _raw_curve_frame(n=400, neg=3)
    clean, _ = tcl.clean_curve(df)
    curve = tcl._from_frame(clean)
    rc.assert_causal(curve, df.index)               # raises AssertionError on a leak


def test_back_compat_load_treasury_curve_returns_cleaned_dict(tmp_path):
    """The back-compat wrapper still returns just the curve dict, cleaned."""
    _raw_curve_frame(neg=2).to_parquet(tmp_path / "treasury_curve.parquet")
    curve = tcl.load_treasury_curve(cache_dir=tmp_path, research_cache=None)
    assert set(curve) == {"3m", "y2", "5y", "10y", "30y"}
    assert curve["3m"].min() >= 0.0                 # negatives repaired
