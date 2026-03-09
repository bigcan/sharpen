"""
BC Feasibility Test — Stage 4 Gate Check
=========================================

Quick sklearn test: can simple classifiers predict DP oracle trading decisions
from causal OHLCV features well enough to be profitable?

Two prediction targets × three classifiers = 6 models.
PF simulation validates whether classification accuracy translates to profit.

Usage:
    python scripts/bc_feasibility_test.py
    python scripts/bc_feasibility_test.py --data data/processed/gc_2025_3min_front.parquet
    python scripts/bc_feasibility_test.py --fee 0.50
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Add project root for dp_oracle import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.dp_oracle_1min import dp_oracle


# ═══════════════════════════════════════════════════════════════════════
# Section 1: Oracle Label Generation
# ═══════════════════════════════════════════════════════════════════════

def generate_oracle_labels(prices: np.ndarray, fee_bps: float) -> dict:
    """Run DP oracle and extract direction/switch labels."""
    print("=" * 70)
    print("SECTION 1: Oracle Label Generation")
    print("=" * 70)

    result = dp_oracle(prices, fee_bps=fee_bps)
    positions = result['positions']  # 0=Short, 1=Flat, 2=Long
    T = len(positions)

    # Map to direction: Short→0, Long→1, Flat→carry-forward
    direction = np.zeros(T, dtype=np.int32)
    last_dir = 1  # default Long if starts Flat
    for t in range(T):
        if positions[t] == 0:    # Short
            last_dir = 0
        elif positions[t] == 2:  # Long
            last_dir = 1
        # Flat: carry forward last_dir
        direction[t] = last_dir

    # Switch labels
    switch = np.zeros(T, dtype=np.int32)
    switch[1:] = (direction[1:] != direction[:-1]).astype(np.int32)

    # Stats
    n_long = np.sum(positions == 2)
    n_short = np.sum(positions == 0)
    n_flat = np.sum(positions == 1)
    n_switch = np.sum(switch)
    switch_rate = n_switch / T * 100

    print(f"\n  Oracle PF (PnL):      {result['profit_factor_pnl']:.3f}")
    print(f"  Oracle PF (Returns):  {result['profit_factor_returns']:.3f}")
    print(f"  Positions: Long={n_long:,} ({n_long/T*100:.1f}%), "
          f"Short={n_short:,} ({n_short/T*100:.1f}%), "
          f"Flat={n_flat:,} ({n_flat/T*100:.1f}%)")
    print(f"  Direction (after carry-forward): "
          f"Long={np.sum(direction==1):,} ({np.sum(direction==1)/T*100:.1f}%), "
          f"Short={np.sum(direction==0):,} ({np.sum(direction==0)/T*100:.1f}%)")
    print(f"  Switches: {n_switch:,} (rate: {switch_rate:.2f}%)")

    return {
        'direction': direction,
        'switch': switch,
        'oracle_result': result,
    }


# ═══════════════════════════════════════════════════════════════════════
# Section 2: Feature Engineering (18-dim OHLCV-only)
# ═══════════════════════════════════════════════════════════════════════

def symlog(x: np.ndarray) -> np.ndarray:
    """SymLog transform: sign(x) * log(1 + |x|)."""
    return np.sign(x) * np.log1p(np.abs(x))


def ema_zscore_tanh(values: np.ndarray, span: int = 60) -> np.ndarray:
    """EMA Z-Score with causal shift → tanh soft-clip.
    Replicates feature_engineering.py normalization pipeline."""
    s = pd.Series(values.astype(np.float64))
    ema_mean = s.ewm(span=span, adjust=False).mean().shift(1)
    ema_std = s.ewm(span=span, adjust=False).std().shift(1)

    mu = ema_mean.values.copy()
    sigma = ema_std.values.copy()

    # Backward-fill pre-warmup with first valid value
    first_valid = np.where(~np.isnan(mu))[0]
    if len(first_valid) > 0:
        fv = first_valid[0]
        mu[:fv] = mu[fv]
        sigma[:fv] = sigma[fv]
    else:
        mu[np.isnan(mu)] = 0.0

    sigma = np.where(np.isnan(sigma) | (sigma < 1e-8), 1.0, sigma)
    z = (values.astype(np.float64) - mu) / sigma
    return np.tanh(z * 0.5).astype(np.float32)


def normalize(values: np.ndarray, bounded: bool = False, span: int = 60) -> np.ndarray:
    """Full normalization: SymLog → EMA-Z → tanh. Bypass for bounded features."""
    if bounded:
        return values.astype(np.float32)
    return ema_zscore_tanh(symlog(values), span=span)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build 18-dim OHLCV feature matrix."""
    print("\n" + "=" * 70)
    print("SECTION 2: Feature Engineering (18-dim OHLCV-only)")
    print("=" * 70)

    cl = df['close'].values.astype(np.float64)
    hi = df['high'].values.astype(np.float64)
    lo = df['low'].values.astype(np.float64)
    op = df['open'].values.astype(np.float64)
    vol = df['volume'].values.astype(np.float64)
    vol = np.where(vol > 0, vol, 1.0)

    features = pd.DataFrame(index=df.index)

    # 1. Multi-horizon log returns (6 dims)
    for horizon in [1, 3, 5, 15, 30, 60]:
        lr = np.zeros_like(cl)
        if len(cl) > horizon:
            prev = cl[:-horizon]
            prev_safe = np.where(prev > 0, prev, 1e-9)
            lr[horizon:] = np.log(cl[horizon:] / prev_safe)
        lr = np.clip(lr, -0.1, 0.1)
        features[f'logret_{horizon}'] = normalize(lr)

    # 2. Parkinson volatility 15-bar (1 dim)
    hl_ratio = np.where(lo > 0, hi / lo, 1.0)
    ln_hl_sq = np.log(hl_ratio) ** 2
    parkinson_raw = pd.Series(ln_hl_sq).rolling(window=15, min_periods=1).mean().values
    features['parkinson_vol'] = normalize(np.sqrt(parkinson_raw / (4 * np.log(2))))

    # 3. Volatility regime ratio 15/60 (1 dim) — bounded
    parkinson_60 = pd.Series(ln_hl_sq).rolling(window=60, min_periods=1).mean().values
    ratio_raw = parkinson_raw / (parkinson_60 + 1e-8)
    features['vol_ratio'] = np.tanh(np.log1p(ratio_raw) / 1.5 - 0.46).astype(np.float32)

    # 4. ATR normalized (1 dim)
    atr_raw = pd.Series(hi - lo).rolling(window=15, min_periods=1).mean().values
    mid = (hi + lo) / 2.0
    mid_safe = np.where(mid > 0, mid, 1e-9)
    features['atr_norm'] = normalize(atr_raw / mid_safe)

    # 5. RSI 14 (1 dim) — bounded
    delta = np.diff(cl, prepend=cl[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(span=27, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=27, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-8)
    rsi_raw = 1.0 - 1.0 / (1.0 + rs)
    features['rsi_14'] = (rsi_raw * 2.0 - 1.0).astype(np.float32)  # bounded [-1, 1]

    # 6. Bollinger %B 20 (1 dim) — bounded
    sma20 = pd.Series(cl).rolling(window=20, min_periods=1).mean().values
    std20 = pd.Series(cl).rolling(window=20, min_periods=1).std().values
    std20 = np.where(np.isnan(std20) | (std20 < 1e-9), 1e-9, std20)
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    band_width = upper - lower
    pct_b_raw = (cl - lower) / (band_width + 1e-8)
    features['bbpct_20'] = np.tanh((pct_b_raw - 0.5) * 2.0).astype(np.float32)  # bounded

    # 7. Z-score 20 (1 dim)
    z_score_raw = (cl - sma20) / (std20 + 1e-8)
    features['z_score_20'] = normalize(z_score_raw)

    # 8. Relative volume 15 (1 dim)
    vol_ma15 = pd.Series(vol).rolling(window=15, min_periods=1).mean().values
    vol_ma15_safe = np.where(vol_ma15 > 0, vol_ma15, 1.0)
    features['rvol_15'] = normalize(vol / vol_ma15_safe)

    # 9. CVD proxy 5-bar (1 dim)
    is_bullish = (cl >= op).astype(np.float64)
    signed_vol = (2 * is_bullish - 1) * vol
    features['cvd_proxy_5'] = normalize(
        pd.Series(signed_vol).rolling(window=5, min_periods=1).sum().values
    )

    # 10. VWAP deviation (1 dim)
    tp = (hi + lo + cl) / 3.0
    tp_vol = tp * vol
    vwap_num = pd.Series(tp_vol).rolling(window=20, min_periods=1).sum().values
    vwap_den = pd.Series(vol).rolling(window=20, min_periods=1).sum().values
    vwap_den_safe = np.where(vwap_den > 0, vwap_den, 1e-9)
    vwap = vwap_num / vwap_den_safe
    vwap_dev_bps = (cl - vwap) / (cl + 1e-9) * 10000
    features['vwap_dev'] = normalize(vwap_dev_bps)

    # 11. Session sin/cos (2 dims) — bounded
    if 'timestamp' in df.columns:
        ts = pd.to_datetime(df['timestamp'])
        minutes_in_day = ts.dt.hour * 60 + ts.dt.minute
        phase = 2.0 * np.pi * minutes_in_day.values / 1440.0
        features['session_sin'] = np.sin(phase).astype(np.float32)
        features['session_cos'] = np.cos(phase).astype(np.float32)
    else:
        features['session_sin'] = np.zeros(len(df), dtype=np.float32)
        features['session_cos'] = np.zeros(len(df), dtype=np.float32)

    # 12. Day-of-week sin (1 dim) — bounded (CME Gold)
    if 'timestamp' in df.columns:
        ts = pd.to_datetime(df['timestamp'])
        dow = ts.dt.dayofweek.values.astype(np.float64)
        phase = 2.0 * np.pi * dow / 5.0
        features['dow_sin'] = np.sin(phase).astype(np.float32)
    else:
        features['dow_sin'] = np.zeros(len(df), dtype=np.float32)

    # Check for NaN
    nan_counts = features.isna().sum()
    total_nans = nan_counts.sum()
    print(f"\n  Features: {features.shape[1]} dims, {len(features):,} bars")
    print(f"  NaN count: {total_nans} (after warmup row 60: "
          f"{features.iloc[60:].isna().sum().sum()})")

    # Fill any remaining NaN with 0 (warmup period)
    features = features.fillna(0.0)

    print(f"  Feature ranges (post-norm):")
    for col in features.columns:
        vals = features[col].values
        print(f"    {col:20s}: [{vals.min():+.3f}, {vals.max():+.3f}]  "
              f"mean={vals.mean():+.4f}")

    return features


# ═══════════════════════════════════════════════════════════════════════
# Section 3: Time-Split
# ═══════════════════════════════════════════════════════════════════════

def time_split(df: pd.DataFrame, features: pd.DataFrame,
               direction: np.ndarray, switch: np.ndarray,
               warmup: int = 60) -> dict:
    """Split into train/val/test by date. Drop first `warmup` bars."""
    print("\n" + "=" * 70)
    print("SECTION 3: Time-Split")
    print("=" * 70)

    ts = pd.to_datetime(df['timestamp'])

    train_mask = (ts >= '2025-01-01') & (ts < '2025-11-01')
    val_mask = (ts >= '2025-11-01') & (ts < '2025-12-01')
    test_mask = (ts >= '2025-12-01')

    # Apply warmup: skip first `warmup` bars globally
    warmup_mask = np.zeros(len(df), dtype=bool)
    warmup_mask[:warmup] = True
    train_mask = train_mask & ~warmup_mask

    splits = {}
    for name, mask in [('train', train_mask), ('val', val_mask), ('test', test_mask)]:
        idx = np.where(mask.values)[0]
        splits[name] = {
            'X': features.iloc[idx].values,
            'y_dir': direction[idx],
            'y_switch': switch[idx],
            'idx': idx,
        }
        n = len(idx)
        dir_balance = np.mean(splits[name]['y_dir'])
        switch_rate = np.mean(splits[name]['y_switch']) * 100
        print(f"  {name:5s}: {n:>7,} bars  |  "
              f"dir_balance={dir_balance:.3f} (1=all Long)  |  "
              f"switch_rate={switch_rate:.2f}%")

    print(f"\n  WARNING: Normalization computed on full dataset (minor leakage, "
          f"acceptable for feasibility)")


    return splits


# ═══════════════════════════════════════════════════════════════════════
# Section 4: Classifiers
# ═══════════════════════════════════════════════════════════════════════

def train_classifiers(splits: dict) -> dict:
    """Train 6 classifiers: 3 for direction, 3 for switch."""
    print("\n" + "=" * 70)
    print("SECTION 4: Classifiers (sklearn)")
    print("=" * 70)

    classifiers = {
        'LogisticRegression': LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs'),
        'RandomForest': RandomForestClassifier(
            n_estimators=100, max_depth=10, n_jobs=-1, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
    }

    results = {}

    # --- Direction prediction ---
    print("\n  Direction Prediction (target: oracle direction 0=Short, 1=Long)")
    print("  " + "-" * 60)

    X_train = splits['train']['X']
    y_train = splits['train']['y_dir']

    for clf_name, clf in classifiers.items():
        t0 = time.time()
        clf.fit(X_train, y_train)
        elapsed = time.time() - t0

        for split_name in ['val', 'test']:
            X = splits[split_name]['X']
            y_true = splits[split_name]['y_dir']
            y_pred = clf.predict(X)
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='macro')

            key = f'dir_{clf_name}_{split_name}'
            results[key] = {
                'y_pred': y_pred,
                'y_true': y_true,
                'acc': acc,
                'f1': f1,
                'clf_name': clf_name,
                'target': 'direction',
                'split': split_name,
            }

        val_acc = results[f'dir_{clf_name}_val']['acc']
        test_acc = results[f'dir_{clf_name}_test']['acc']
        print(f"    {clf_name:25s}  val_acc={val_acc:.4f}  test_acc={test_acc:.4f}  "
              f"({elapsed:.1f}s)")

    # --- Switch prediction ---
    print("\n  Switch Prediction (target: switch 0/1, with current_direction feature)")
    print("  " + "-" * 60)

    # Add current direction as extra feature for switch prediction
    X_train_sw = np.column_stack([X_train, splits['train']['y_dir']])
    y_train_sw = splits['train']['y_switch']

    classifiers_sw = {
        'LogisticRegression': LogisticRegression(
            C=1.0, max_iter=1000, solver='lbfgs', class_weight='balanced'),
        'RandomForest': RandomForestClassifier(
            n_estimators=100, max_depth=10, n_jobs=-1, random_state=42,
            class_weight='balanced'),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
    }

    for clf_name, clf in classifiers_sw.items():
        t0 = time.time()
        clf.fit(X_train_sw, y_train_sw)
        elapsed = time.time() - t0

        for split_name in ['val', 'test']:
            X = splits[split_name]['X']
            y_dir = splits[split_name]['y_dir']
            X_sw = np.column_stack([X, y_dir])
            y_true = splits[split_name]['y_switch']
            y_pred = clf.predict(X_sw)
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='macro')

            key = f'switch_{clf_name}_{split_name}'
            results[key] = {
                'y_pred': y_pred,
                'y_true': y_true,
                'acc': acc,
                'f1': f1,
                'clf_name': clf_name,
                'target': 'switch',
                'split': split_name,
            }

        val_acc = results[f'switch_{clf_name}_val']['acc']
        test_acc = results[f'switch_{clf_name}_test']['acc']
        val_f1 = results[f'switch_{clf_name}_val']['f1']
        test_f1 = results[f'switch_{clf_name}_test']['f1']
        print(f"    {clf_name:25s}  val_acc={val_acc:.4f}  test_acc={test_acc:.4f}  "
              f"val_F1={val_f1:.4f}  test_F1={test_f1:.4f}  ({elapsed:.1f}s)")

    return results


# ═══════════════════════════════════════════════════════════════════════
# Section 5: PF Simulation
# ═══════════════════════════════════════════════════════════════════════

def simulate_pf(directions: np.ndarray, prices: np.ndarray,
                fee_bps: float = 0.35) -> dict:
    """
    Simulate PF from predicted direction sequence.

    Position follows predicted direction each bar.
    PnL per bar = direction_sign * (price[t+1] - price[t])
    Switch cost = 2 * fee_bps * price when direction changes.
    Terminal exit fee charged on last bar.
    """
    T = len(directions)
    fee_frac = fee_bps / 10000.0

    # Map 0=Short→-1, 1=Long→+1
    pos_sign = directions.astype(np.float64) * 2 - 1  # 0→-1, 1→+1

    pnl = np.zeros(T)
    for t in range(T - 1):
        dp = prices[t + 1] - prices[t]
        pnl[t] = pos_sign[t] * dp

        # Switch cost
        if t > 0 and directions[t] != directions[t - 1]:
            pnl[t] -= 2 * fee_frac * prices[t]

    # First bar: entry cost
    pnl[0] -= fee_frac * prices[0]

    # Terminal exit fee (closing the final position)
    if T >= 2:
        pnl[T - 2] -= fee_frac * prices[T - 1]

    pos_pnl = pnl[pnl > 0].sum()
    neg_pnl = abs(pnl[pnl < 0].sum())
    pf = pos_pnl / neg_pnl if neg_pnl > 1e-9 else float('inf')

    total_return = np.sum(pnl) / prices[0] * 100
    n_switches = np.sum(np.diff(directions) != 0)

    return {
        'pf': pf,
        'total_return_pct': total_return,
        'n_switches': int(n_switches),
        'pos_pnl': pos_pnl,
        'neg_pnl': neg_pnl,
    }


def run_pf_simulations(splits: dict, clf_results: dict,
                       prices: np.ndarray, fee_bps: float) -> dict:
    """Run PF simulation for oracle, random, and all classifiers."""
    print("\n" + "=" * 70)
    print("SECTION 5: PF Simulation")
    print("=" * 70)

    pf_results = {}

    for split_name in ['val', 'test']:
        idx = splits[split_name]['idx']
        split_prices = prices[idx]
        oracle_dir = splits[split_name]['y_dir']

        print(f"\n  --- {split_name.upper()} split ({len(idx):,} bars) ---")

        # Oracle reconstruction
        r = simulate_pf(oracle_dir, split_prices, fee_bps)
        pf_results[f'oracle_{split_name}'] = r
        print(f"    Oracle:        PF={r['pf']:.3f}  "
              f"Return={r['total_return_pct']:+.1f}%  "
              f"Switches={r['n_switches']:,}")

        # Random baseline
        rng = np.random.RandomState(42)
        random_dir = rng.randint(0, 2, size=len(idx))
        r = simulate_pf(random_dir, split_prices, fee_bps)
        pf_results[f'random_{split_name}'] = r
        print(f"    Random:        PF={r['pf']:.3f}  "
              f"Return={r['total_return_pct']:+.1f}%  "
              f"Switches={r['n_switches']:,}")

        # All direction classifiers
        for key, res in clf_results.items():
            if res['target'] != 'direction' or res['split'] != split_name:
                continue
            y_pred = res['y_pred']
            r = simulate_pf(y_pred, split_prices, fee_bps)
            pf_results[f'{key}_pf'] = r
            print(f"    {res['clf_name']:18s} PF={r['pf']:.3f}  "
                  f"Return={r['total_return_pct']:+.1f}%  "
                  f"Switches={r['n_switches']:,}  "
                  f"Acc={res['acc']:.4f}")

    return pf_results


# ═══════════════════════════════════════════════════════════════════════
# Section 6: Summary & Verdict
# ═══════════════════════════════════════════════════════════════════════

def print_verdict(clf_results: dict, pf_results: dict):
    """Print summary table with GO / MARGINAL / NO-GO verdict."""
    print("\n" + "=" * 70)
    print("SECTION 6: Summary & Verdict")
    print("=" * 70)

    print("\n  Direction Classifiers -- Test Set:")
    print(f"  {'Classifier':<25s} {'Acc':>7s} {'F1':>7s} {'PF':>7s} {'Return':>8s} {'Verdict':>10s}")
    print("  " + "-" * 70)

    best_acc = 0
    best_pf = 0

    for key, res in clf_results.items():
        if res['target'] != 'direction' or res['split'] != 'test':
            continue

        acc = res['acc']
        f1 = res['f1']
        pf_key = f'{key}_pf'
        pf_data = pf_results.get(pf_key, {})
        pf = pf_data.get('pf', 0)
        ret = pf_data.get('total_return_pct', 0)

        best_acc = max(best_acc, acc)
        best_pf = max(best_pf, pf)

        if acc > 0.55 and pf > 1.10:
            verdict = "GO"
        elif acc > 0.52 or pf > 1.00:
            verdict = "MARGINAL"
        else:
            verdict = "NO-GO"

        print(f"  {res['clf_name']:<25s} {acc:>7.4f} {f1:>7.4f} {pf:>7.3f} "
              f"{ret:>+7.1f}% {verdict:>10s}")

    # Oracle and random baselines
    oracle_pf = pf_results.get('oracle_test', {}).get('pf', 0)
    random_pf = pf_results.get('random_test', {}).get('pf', 0)
    print(f"\n  Oracle baseline PF:  {oracle_pf:.3f}")
    print(f"  Random baseline PF:  {random_pf:.3f}")

    # Switch classifiers summary
    print("\n  Switch Classifiers -- Test Set:")
    print(f"  {'Classifier':<25s} {'Acc':>7s} {'F1':>7s}")
    print("  " + "-" * 40)
    for key, res in clf_results.items():
        if res['target'] != 'switch' or res['split'] != 'test':
            continue
        print(f"  {res['clf_name']:<25s} {res['acc']:>7.4f} {res['f1']:>7.4f}")

    # Overall verdict
    print("\n" + "=" * 70)
    if best_acc > 0.55 and best_pf > 1.10:
        verdict = "GO"
        msg = ("BC is feasible. Best classifier exceeds both thresholds "
               f"(acc={best_acc:.4f} > 0.55, PF={best_pf:.3f} > 1.10). "
               "Proceed to Stage 4 full BC pipeline.")
    elif best_acc > 0.52 or best_pf > 1.00:
        verdict = "MARGINAL"
        msg = (f"BC shows weak signal (acc={best_acc:.4f}, PF={best_pf:.3f}). "
               "Consider: (1) more features (LOB), (2) neural BC, "
               "(3) different target formulation.")
    else:
        verdict = "NO-GO"
        msg = (f"BC not feasible with OHLCV features (acc={best_acc:.4f}, "
               f"PF={best_pf:.3f}). Oracle decisions may require LOB features "
               "or non-linear temporal patterns beyond sklearn scope.")

    print(f"  VERDICT: {verdict}")
    print(f"  {msg}")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="BC Feasibility Test — Stage 4 Gate Check")
    parser.add_argument("--data", default="data/processed/gc_2025_3min_front.parquet",
                        help="Path to 3-min OHLCV parquet")
    parser.add_argument("--fee", type=float, default=0.35,
                        help="One-way fee in bps (default: 0.35 for Gold CME)")
    parser.add_argument("--resolution", default=None,
                        help="Resample to resolution (e.g. 5min, 15min, 30min). None=use as-is.")
    parser.add_argument("--price_col", default="close", choices=["close", "mid_price"],
                        help="Price column for oracle & PF simulation (default: close)")
    args = parser.parse_args()

    t_start = time.time()

    # Load data
    print("=" * 70)
    print("BC FEASIBILITY TEST -- Stage 4 Gate Check")
    print("=" * 70)
    print(f"\n  Data: {args.data}")
    print(f"  Fee:  {args.fee} bps one-way ({args.fee * 2} bps RT)")
    print(f"  Price column: {args.price_col}")

    df = pd.read_parquet(args.data)

    # Resample if requested
    if args.resolution:
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        df = df.resample(args.resolution).agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum',
        }).dropna().reset_index()
        # Always compute mid_price for fallback
        df['mid_price'] = (df['high'] + df['low']) / 2.0
        print(f"  Resampled to {args.resolution}: {len(df):,} bars")
    else:
        print(f"  Bars: {len(df):,}")

    # Ensure mid_price column exists for backward compat / xcheck
    if 'mid_price' not in df.columns:
        df['mid_price'] = (df['high'] + df['low']) / 2.0

    prices = df[args.price_col].values.astype(np.float64)

    # Section 1: Oracle labels
    labels = generate_oracle_labels(prices, fee_bps=args.fee)

    # Section 2: Features
    features = build_features(df)

    # Section 3: Time-split
    splits = time_split(df, features, labels['direction'], labels['switch'])

    # Section 4: Classifiers
    clf_results = train_classifiers(splits)

    # Section 5: PF simulation
    pf_results = run_pf_simulations(splits, clf_results, prices, args.fee)

    # Section 6: Verdict
    print_verdict(clf_results, pf_results)

    # PF-XCHECK: cross-validate with the other price column
    print("\n" + "=" * 70)
    print("PF-XCHECK — Cross-validation: close vs mid_price")
    print("=" * 70)

    other_col = 'mid_price' if args.price_col == 'close' else 'close'
    other_prices = df[other_col].values.astype(np.float64)

    # Run oracle on both price series (use max 50K bars for speed)
    xcheck_limit = min(50000, len(prices))
    primary_result = dp_oracle(prices[:xcheck_limit], fee_bps=args.fee)
    other_result = dp_oracle(other_prices[:xcheck_limit], fee_bps=args.fee)

    pf_primary = primary_result['profit_factor_pnl']
    pf_other = other_result['profit_factor_pnl']
    avg_pf = (pf_primary + pf_other) / 2.0
    divergence_pct = abs(pf_primary - pf_other) / avg_pf * 100 if avg_pf > 1e-9 else 0

    print(f"\n  PF-XCHECK (PnL-based, first {xcheck_limit:,} bars):")
    print(f"    {args.price_col:10s} PF = {pf_primary:.3f}")
    print(f"    {other_col:10s} PF = {pf_other:.3f}")
    print(f"    Divergence = {divergence_pct:.1f}%")

    if divergence_pct > 30:
        print(f"\n  *** WARNING: PF divergence {divergence_pct:.1f}% > 30% ***")
        print("  *** Data may have corrupted high/low values inflating mid_price. ***")
        print("  *** Run scripts/clean_ohlcv.py before trusting results. ***")
    else:
        print("    OK — divergence within 30% tolerance")

    elapsed = time.time() - t_start
    print(f"\n  Total runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
