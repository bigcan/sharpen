import json
import pandas as pd
import numpy as np
from pathlib import Path
from finrl_pro.eval.robustness import deflated_sharpe_ratio, probability_backtest_overfitting
from scipy.stats import skew, kurtosis

def run_robustness_checks():
    report_path = Path("reports/matrix_phase3/eval_report.json")
    if not report_path.exists():
        print(f"Error: {report_path} not found.")
        return

    with open(report_path, "r") as f:
        report_data = json.load(f)

    # 1. Gather Universe Statistics for DSR
    sharpe_ratios = []
    fingerprints = []
    
    for entry in report_data:
        metrics = entry.get("evaluated_metrics", {})
        sr = metrics.get("sharpe_ratio")
        fp = entry.get("fingerprint_id")
        if sr is not None and fp:
            sharpe_ratios.append(sr)
            fingerprints.append(fp)

    n_trials = len(sharpe_ratios)
    sr_std = np.std(sharpe_ratios, ddof=1)
    print(f"Universe Statistics (N={n_trials}):")
    print(f"  Mean Sharpe: {np.mean(sharpe_ratios):.4f}")
    print(f"  Std Sharpe:  {sr_std:.4f}")

    # 2. Load Winner Data
    winner_fp = "c60a1cbb-635b-40c3-80b4-d286beaca3ec"
    winner_returns_path = Path(f"reports/{winner_fp}/returns.csv")
    
    if not winner_returns_path.exists():
        print(f"Error: Winner returns file {winner_returns_path} not found.")
        return

    # Assuming returns.csv has a date index and a 'return' column or similar. 
    # Let's check format by reading first few lines? 
    # Usually single column or Date,Return
    try:
        winner_df = pd.read_csv(winner_returns_path, index_col=0, parse_dates=True)
        # Assuming first column is returns if named 'close' or similar, or just 'returns'
        # Let's assume it's the first numeric column
        winner_returns = winner_df.iloc[:, 0].values
    except Exception as e:
        print(f"Error reading winner returns: {e}")
        return

    # 3. Compute DSR
    # Calculate observed stats
    obs_sr = 0.8819302215871491 # From report
    # Recalculate from returns to be sure?
    # ann_factor = 252
    # obs_sr_recalc = (np.mean(winner_returns) / np.std(winner_returns)) * np.sqrt(252)
    # print(f"  Recalculated Winner SR: {obs_sr_recalc:.4f}")
    
    ret_skew = skew(winner_returns)
    ret_kurt = kurtosis(winner_returns) # Fisher kurtosis (normal=0)
    n_obs = len(winner_returns)

    dsr = deflated_sharpe_ratio(
        observed_sr=obs_sr,
        sr_std=sr_std,
        n_trials=n_trials,
        returns_skew=ret_skew,
        returns_kurt=ret_kurt,
        n_returns=n_obs,
        periods_per_year=252
    )

    print("\nDeflated Sharpe Ratio (DSR):")
    print(f"  Observed SR: {obs_sr:.4f}")
    print(f"  Skewness:    {ret_skew:.4f}")
    print(f"  Kurtosis:    {ret_kurt:.4f}")
    print(f"  N Observations: {n_obs}")
    print(f"  DSR Probability: {dsr:.4%}")

    # 4. Compute PBO
    print("\nComputing PBO (Probability of Backtest Overfitting)...")
    # We need to load returns for ALL trials to form the matrix
    # This might be slow if many files, but N=~28 is fine.
    
    all_returns = []
    valid_fps = []
    
    for fp in fingerprints:
        r_path = Path(f"reports/{fp}/returns.csv")
        if r_path.exists():
            try:
                df = pd.read_csv(r_path, index_col=0, parse_dates=True)
                # Align to winner index? Or just assume same length/dates?
                # Better to reindex to winner's index to ensure alignment
                aligned = df.iloc[:, 0].reindex(winner_df.index).fillna(0.0).values
                all_returns.append(aligned)
                valid_fps.append(fp)
            except Exception:
                pass
    
    if not all_returns:
        print("No returns loaded for PBO.")
        return

    matrix = np.column_stack(all_returns) # T x N
    
    pbo, logits = probability_backtest_overfitting(matrix, n_splits=10)
    
    print(f"  Loaded {len(valid_fps)} strategies.")
    print(f"  PBO: {pbo:.4%}")
    
    # Interpretations
    print("\n--- Robustness Conclusions ---")
    if dsr > 0.95:
        print("[PASS] DSR > 95%: High confidence result is not a false positive.")
    elif dsr > 0.50:
        print("[WARN] DSR > 50%: Result is likely real but not statistically significant at 5% level given trial count.")
    else:
        print("[FAIL] DSR < 50%: Result is likely a false discovery due to multiple testing.")
        
    if pbo < 0.20:
        print("[PASS] PBO < 20%: Low risk of overfitting.")
    else:
        print(f"[FAIL] PBO {pbo:.1%}: High risk of overfitting.")

if __name__ == "__main__":
    run_robustness_checks()
