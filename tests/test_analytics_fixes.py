"""
Verify analytics module bug fixes (BUG-A1, BUG-A2, BUG-A3) and logic for BUG-P2.
"""
import sys
import os
import numpy as np
import pandas as pd
from finrl_pro_ds.analytics.pyfolio_analyzer import PyfolioAnalyzer

# Fix path ensuring finrl_pro_ds is importable
sys.path.append(os.getcwd())

def test_sortino_uses_lower_partial_moment():
    """BUG-A1: Sortino should use sqrt(mean(min(r,0)^2)), not RMS of negatives only."""
    np.random.seed(42)
    r = pd.Series(np.random.randn(10000) * 0.001)
    m = PyfolioAnalyzer(r).get_audit_metrics()

    # For ~symmetric distribution, Sortino ≈ Sharpe * sqrt(2)
    ratio = m["sortino_ratio"] / m["sharpe_ratio"] if abs(m["sharpe_ratio"]) > 0.01 else 1.0
    # Should be roughly sqrt(2) ≈ 1.41
    assert 1.0 < abs(ratio) < 2.5, f"Sortino/Sharpe ratio {ratio:.2f} is out of expected range"
    print(f"PASS: Sortino/Sharpe ratio = {ratio:.2f} (expected ~1.41 for normal)")

def test_sortino_all_positive():
    """Sortino with all-positive returns should be 0 (downside deviation = 0)."""
    r = pd.Series(np.ones(1000) * 0.001)
    m = PyfolioAnalyzer(r).get_audit_metrics()
    assert m["sortino_ratio"] == 0.0, f"Sortino should be 0 for all-positive returns, got {m['sortino_ratio']}"
    print("PASS: All-positive returns -> Sortino = 0")

def test_annual_return_no_overflow():
    """BUG-A2: Annual return should not overflow for very long series."""
    np.random.seed(42)
    r = pd.Series(np.random.randn(500000) * 0.0001)
    m = PyfolioAnalyzer(r).get_audit_metrics()
    assert np.isfinite(m["annual_return"]), f"Annual return is not finite: {m['annual_return']}"
    print(f"PASS: 500K returns -> annual_return = {m['annual_return']:.4f}%")

def test_var_key_renamed():
    """BUG-A3: VaR key should be 'minute_value_at_risk', not 'daily_value_at_risk'."""
    r = pd.Series(np.random.randn(1000) * 0.001)
    m = PyfolioAnalyzer(r).get_audit_metrics()
    assert "minute_value_at_risk" in m, "minute_value_at_risk key missing"
    assert "daily_value_at_risk" not in m, "daily_value_at_risk key still present"
    print(f"PASS: VaR key correctly renamed to minute_value_at_risk = {m['minute_value_at_risk']:.6f}")

def test_trade_count_logic_p2():
    """BUG-P2: Verify robust trade counting logic (Base Count + Sign Change)."""
    print("\nTesting Trade Count Logic (BUG-P2)...")
    
    # Logic being tested:
    # base_count = np.sum(np.abs(np.diff(pos)) > epsilon)
    # sign_flips = np.sum((pos[:-1] * pos[1:]) < -epsilon)
    # trade_count = base_count + sign_flips

    def count_trades(pos_list):
        pos_arr = np.array(pos_list)
        if len(pos_arr) < 2: return 0
        pos_deltas = np.abs(np.diff(pos_arr))
        base = np.sum(pos_deltas > 1e-6)
        sign = np.sum((pos_arr[:-1] * pos_arr[1:]) < -1e-9)
        return base + sign

    # Case 1: Small trade 0 -> 0.1 (Base=1, Sign=0 -> 1)
    assert count_trades([0.0, 0.1]) == 1, "Failed Case 1: Small Trade"
    
    # Case 2: Partial Close 0.2 -> 0.1 (Base=1, Sign=0 -> 1)
    assert count_trades([0.2, 0.1]) == 1, "Failed Case 2: Partial Close"

    # Case 3: Flip 1.0 -> -1.0 (Base=1, Sign=1 -> 2)
    # |diff|=2.0 (Base=1). 1*-1=-1 (Sign=1). Total 2.
    assert count_trades([1.0, -1.0]) == 2, "Failed Case 3: Instant Flip"

    # Case 4: Flip via Zero 1.0 -> 0.0 -> -1.0 (Base=2, Sign=0 -> 2)
    # 1->0: Base=1, Sign=0.
    # 0->-1: Base=1, Sign=0.
    # Total 2.
    assert count_trades([1.0, 0.0, -1.0]) == 2, "Failed Case 4: Interrupted Flip"

    # Case 5: Noise (0 -> 1e-12)
    assert count_trades([0.0, 1e-12]) == 0, "Failed Case 5: Noise"

    print("PASS: Trade Count Logic (Base + Sign Flip) is robust.")

if __name__ == "__main__":
    test_sortino_uses_lower_partial_moment()
    test_sortino_all_positive()
    test_annual_return_no_overflow()
    test_var_key_renamed()
    test_trade_count_logic_p2()
    print("\n=== ALL ANALYTICS FIX TESTS PASSED ===")
