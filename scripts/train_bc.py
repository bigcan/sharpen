"""
BC Training Pipeline — Gold 15-min Walk-Forward
================================================

Behavioral cloning from DP oracle labels on 15-min Gold.
3-window expanding walk-forward validation.

Primary model: LogisticRegression (proven GO in feasibility test).
Secondary: GBT, MLP (for comparison).

Answers: "Does the BC model produce consistent PF > 1.15 across all OOS windows?"

Usage:
    python scripts/train_bc.py
    python scripts/train_bc.py --data data/processed/gc_2025_3min_front.parquet --fee 6.5
    python scripts/train_bc.py --epochs 200 --patience 20
"""

import argparse
import copy
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.bc_feasibility_test import build_features, generate_oracle_labels, simulate_pf


# ═══════════════════════════════════════════════════════════════════════
# Section 1: TradingMLP (secondary model)
# ═══════════════════════════════════════════════════════════════════════

class TradingMLP(nn.Module):
    """Simple 2-layer MLP for binary direction prediction.

    Input:  (B, 18) — 18 OHLCV features, no private state
    Output: (B, 2)  — logits for {Short=0, Long=1}
    ~19K params.
    """

    def __init__(self, input_dim: int = 18, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ═══════════════════════════════════════════════════════════════════════
# Section 2: Data Loading + Resampling
# ═══════════════════════════════════════════════════════════════════════

def load_and_resample(data_path: str) -> pd.DataFrame:
    """Load 3-min parquet, resample to 15-min."""
    print("=" * 70)
    print("BC TRAINING PIPELINE — Gold 15-min Walk-Forward")
    print("=" * 70)

    df = pd.read_parquet(data_path)
    print(f"\n  Source: {data_path}")
    print(f"  Raw bars: {len(df):,}")

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')

    df = df.resample('15min').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna().reset_index()

    df['mid_price'] = (df['high'] + df['low']) / 2.0
    print(f"  Resampled to 15-min: {len(df):,} bars")
    return df


# ═══════════════════════════════════════════════════════════════════════
# Section 3: Walk-Forward Windows
# ═══════════════════════════════════════════════════════════════════════

WINDOWS = [
    {'name': 'W1', 'train_start': '2025-01-01', 'train_end': '2025-07-01',
     'test_start': '2025-07-01', 'test_end': '2025-09-01'},
    {'name': 'W2', 'train_start': '2025-01-01', 'train_end': '2025-09-01',
     'test_start': '2025-09-01', 'test_end': '2025-11-01'},
    {'name': 'W3', 'train_start': '2025-01-01', 'train_end': '2025-11-01',
     'test_start': '2025-11-01', 'test_end': '2026-01-01'},
]

WARMUP_BARS = 60


def make_splits(df: pd.DataFrame, features: np.ndarray, direction: np.ndarray,
                mid_prices: np.ndarray, window: dict, val_frac: float = 0.2):
    """Create train/val/test splits for one walk-forward window."""
    ts = pd.to_datetime(df['timestamp'])

    train_mask = (ts >= window['train_start']) & (ts < window['train_end'])
    test_mask = (ts >= window['test_start']) & (ts < window['test_end'])

    train_idx = np.where(train_mask.values)[0]
    test_idx = np.where(test_mask.values)[0]

    # Drop warmup from train
    train_idx = train_idx[train_idx >= WARMUP_BARS]

    # Split last val_frac of train for validation
    n_val = int(len(train_idx) * val_frac)
    val_idx = train_idx[-n_val:]
    train_idx = train_idx[:-n_val]

    splits = {}
    for name, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        splits[name] = {
            'X': features[idx],
            'y': direction[idx],
            'prices': mid_prices[idx],
            'idx': idx,
        }

    return splits


# ═══════════════════════════════════════════════════════════════════════
# Section 4: MLP Training Loop
# ═══════════════════════════════════════════════════════════════════════

def train_mlp(X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 100, patience: int = 15,
              batch_size: int = 512, lr: float = 1e-3,
              device: str = 'cpu') -> nn.Module:
    """Train TradingMLP with early stopping on val loss."""
    input_dim = X_train.shape[1]
    model = TradingMLP(input_dim=input_dim).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.long, device=device)
    X_v = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_v = torch.tensor(y_val, dtype=torch.long, device=device)

    best_val_loss = float('inf')
    best_state = None
    wait = 0
    n_samples = len(X_t)

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_samples, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            idx = perm[i:i + batch_size]
            logits = model(X_t[idx])
            loss = criterion(logits, y_t[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_v)
            val_loss = criterion(val_logits, y_v).item()
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == y_v).float().mean().item()

        train_loss = epoch_loss / n_batches

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}  train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"    Early stopping at epoch {epoch} "
                      f"(best val_loss={best_val_loss:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════════════
# Section 5: sklearn Classifiers
# ═══════════════════════════════════════════════════════════════════════

def train_sklearn(X_train: np.ndarray, y_train: np.ndarray):
    """Train LogReg and GBT. Return dict of fitted classifiers."""
    classifiers = {
        'LogReg': LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs'),
        'GBT': GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
    }
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
    return classifiers


# ═══════════════════════════════════════════════════════════════════════
# Section 6: Window Evaluation
# ═══════════════════════════════════════════════════════════════════════

def evaluate_window(splits: dict, mlp: nn.Module, sklearn_clfs: dict,
                    fee_bps: float, device: str = 'cpu') -> dict:
    """Evaluate all models on the test split of one window."""
    X_test = splits['test']['X']
    y_test = splits['test']['y']
    prices = splits['test']['prices']

    results = {}

    # MLP predictions
    mlp.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32, device=device)
        mlp_preds = mlp(X_t).argmax(dim=1).cpu().numpy()

    mlp_acc = accuracy_score(y_test, mlp_preds)
    mlp_pf = simulate_pf(mlp_preds, prices, fee_bps)
    results['MLP'] = {'acc': mlp_acc, **mlp_pf}

    # sklearn classifiers
    for name, clf in sklearn_clfs.items():
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        pf = simulate_pf(preds, prices, fee_bps)
        results[name] = {'acc': acc, **pf}

    # Oracle
    oracle_pf = simulate_pf(y_test, prices, fee_bps)
    results['Oracle'] = {'acc': 1.0, **oracle_pf}

    # Random
    rng = np.random.RandomState(42)
    random_dir = rng.randint(0, 2, size=len(y_test))
    random_pf = simulate_pf(random_dir, prices, fee_bps)
    results['Random'] = {'acc': accuracy_score(y_test, random_dir), **random_pf}

    return results


# ═══════════════════════════════════════════════════════════════════════
# Section 7: Bootstrap Confidence Intervals
# ═══════════════════════════════════════════════════════════════════════

def bootstrap_pf(directions: np.ndarray, prices: np.ndarray, fee_bps: float,
                 n_bootstrap: int = 10000, ci: float = 0.95,
                 block_size: int = 50) -> dict:
    """Block bootstrap PF confidence interval.

    Uses block bootstrap (not i.i.d.) to preserve autocorrelation
    in financial time series. Block size ~50 bars = ~12.5 hours at 15-min.
    """
    rng = np.random.RandomState(42)
    T = len(directions)
    n_blocks = (T + block_size - 1) // block_size

    pf_samples = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        # Sample blocks with replacement
        block_starts = rng.randint(0, T - block_size + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, min(s + block_size, T)) for s in block_starts])
        idx = idx[:T]  # trim to original length

        r = simulate_pf(directions[idx], prices[idx], fee_bps)
        pf_samples[b] = r['pf']

    alpha = (1 - ci) / 2
    lo = np.percentile(pf_samples, alpha * 100)
    hi = np.percentile(pf_samples, (1 - alpha) * 100)
    median = np.median(pf_samples)

    return {
        'median': median,
        'ci_lo': lo,
        'ci_hi': hi,
        'ci_level': ci,
        'n_bootstrap': n_bootstrap,
        'samples': pf_samples,
    }


# ═══════════════════════════════════════════════════════════════════════
# Section 8: Summary + Verdict
# ═══════════════════════════════════════════════════════════════════════

CANDIDATE_MODELS = ['LogReg', 'GBT', 'MLP']
PF_GATE = 1.15


def print_summary(all_results: dict, windows: list, bootstrap_results: dict):
    """Print walk-forward summary table, evaluate ALL models, pick winner."""
    print("\n" + "=" * 70)
    print("WALK-FORWARD SUMMARY")
    print("=" * 70)

    # Header
    header = (f"  {'Window':<6s} {'Test Period':<20s} "
              + "".join(f" {m:>8s}" for m in ['LogReg', 'GBT', 'MLP', 'Oracle'])
              + f" {'LR Acc':>7s}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    # Collect PFs per model
    model_pfs = {m: [] for m in CANDIDATE_MODELS}

    for w in windows:
        wname = w['name']
        r = all_results[wname]
        test_period = f"{w['test_start'][:7]}..{w['test_end'][:7]}"

        row = f"  {wname:<6s} {test_period:<20s} "
        for m in ['LogReg', 'GBT', 'MLP', 'Oracle']:
            pf = r[m]['pf']
            row += f" {pf:>8.3f}"
        row += f" {r['LogReg']['acc']:>6.1%}"
        print(row)

        for m in CANDIDATE_MODELS:
            model_pfs[m].append(r[m]['pf'])

    # Evaluate each candidate against the gate
    print("\n  --- Walk-Forward Gate (PF > {:.2f} in ALL windows) ---".format(PF_GATE))
    print(f"  {'Model':<10s} {'W1':>8s} {'W2':>8s} {'W3':>8s} {'Min':>8s} {'Mean':>8s} {'Pass?':>7s}")
    print("  " + "-" * 55)

    winners = []
    for m in CANDIDATE_MODELS:
        pfs = model_pfs[m]
        min_pf = min(pfs)
        mean_pf = np.mean(pfs)
        passes = all(pf > PF_GATE for pf in pfs)
        pass_str = "PASS" if passes else "FAIL"
        if passes:
            winners.append((m, min_pf, mean_pf))

        row = f"  {m:<10s}"
        for pf in pfs:
            marker = " " if pf > PF_GATE else "*"
            row += f" {pf:>7.3f}{marker}"
        row += f" {min_pf:>8.3f} {mean_pf:>8.3f} {pass_str:>7s}"
        print(row)

    # Bootstrap CIs for winners
    if bootstrap_results:
        print("\n  --- Bootstrap 95% CI (block bootstrap, 10K samples) ---")
        for model_name, bs_windows in bootstrap_results.items():
            ci_strs = []
            all_above_1 = True
            for wname, bs in bs_windows.items():
                ci_strs.append(f"{wname}: [{bs['ci_lo']:.3f}, {bs['ci_hi']:.3f}]")
                if bs['ci_lo'] < 1.0:
                    all_above_1 = False
            sig_str = "SIGNIFICANT" if all_above_1 else "NOT SIGNIFICANT"
            print(f"  {model_name:<10s} {' | '.join(ci_strs)}  -> {sig_str}")

    # Final verdict
    print("\n" + "=" * 70)

    if winners:
        # Pick winner: highest min PF
        winners.sort(key=lambda x: x[1], reverse=True)
        best_name, best_min, best_mean = winners[0]

        # Check statistical significance
        sig = True
        if best_name in bootstrap_results:
            for bs in bootstrap_results[best_name].values():
                if bs['ci_lo'] < 1.0:
                    sig = False

        verdict = "GO"
        msg = (f"Winner: {best_name} — passes walk-forward gate in all {len(windows)} windows "
               f"(min PF={best_min:.3f}, mean PF={best_mean:.3f}).")
        if sig:
            msg += " Bootstrap CI lower bounds > 1.0 — statistically significant."
        else:
            msg += " WARNING: Bootstrap CI includes PF < 1.0 in some windows."
    else:
        verdict = "NO-GO"
        best_name = None
        msg = (f"No model passes PF > {PF_GATE:.2f} in all {len(windows)} windows. "
               "BC does not pass walk-forward gate.")

    print(f"  VERDICT: {verdict}")
    print(f"  {msg}")
    print("=" * 70)

    return verdict, best_name


# ═══════════════════════════════════════════════════════════════════════
# Section 9: Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="BC Training Pipeline — Gold 15-min Walk-Forward")
    parser.add_argument("--data", default="data/processed/gc_2025_3min_front.parquet",
                        help="Path to 3-min OHLCV parquet")
    parser.add_argument("--fee", type=float, default=6.5,
                        help="One-way fee in bps (default: 6.5)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Max MLP training epochs (default: 100)")
    parser.add_argument("--patience", type=int, default=15,
                        help="MLP early stopping patience (default: 15)")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="MLP batch size (default: 512)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="MLP learning rate (default: 1e-3)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--no_mlp", action="store_true",
                        help="Skip MLP training (sklearn-only mode)")
    args = parser.parse_args()

    t_start = time.time()

    # Seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")

    # Load + resample
    df = load_and_resample(args.data)
    mid_prices = df['mid_price'].values.astype(np.float64)

    # Oracle labels (computed once on full series)
    labels = generate_oracle_labels(mid_prices, fee_bps=args.fee)
    direction = labels['direction']

    # Features (computed once on full series)
    features_df = build_features(df)
    features = features_df.values.astype(np.float32)
    print(f"\n  WARNING: Normalization computed on full dataset "
          f"(minor leakage, acceptable for research)")

    # Walk-forward loop
    all_results = {}
    # Store per-window sklearn classifiers and predictions for bootstrap
    window_preds = {}  # {wname: {model_name: (preds, prices)}}

    for w in WINDOWS:
        print("\n" + "=" * 70)
        print(f"WINDOW {w['name']}: Train {w['train_start']}->{w['train_end']}, "
              f"Test {w['test_start']}->{w['test_end']}")
        print("=" * 70)

        splits = make_splits(df, features, direction, mid_prices, w)
        n_train = len(splits['train']['idx'])
        n_val = len(splits['val']['idx'])
        n_test = len(splits['test']['idx'])
        print(f"  Bars: train={n_train:,}  val={n_val:,}  test={n_test:,}")
        print(f"  Train dir balance: {np.mean(splits['train']['y']):.3f} "
              f"(1=all Long)")

        # sklearn classifiers
        print("\n  --- sklearn classifiers ---")
        sklearn_clfs = train_sklearn(splits['train']['X'], splits['train']['y'])
        for name, clf in sklearn_clfs.items():
            val_acc = accuracy_score(splits['val']['y'], clf.predict(splits['val']['X']))
            print(f"    {name}: val_acc={val_acc:.4f}")

        # MLP training (optional)
        if not args.no_mlp:
            print("\n  --- MLP training ---")
            mlp = train_mlp(
                splits['train']['X'], splits['train']['y'],
                splits['val']['X'], splits['val']['y'],
                epochs=args.epochs, patience=args.patience,
                batch_size=args.batch_size, lr=args.lr, device=device,
            )
        else:
            # Dummy MLP that predicts majority class
            mlp = TradingMLP(input_dim=features.shape[1]).to(device)

        # Evaluate
        print("\n  --- Test evaluation ---")
        results = evaluate_window(splits, mlp, sklearn_clfs, args.fee, device)
        all_results[w['name']] = results

        for model_name, r in results.items():
            print(f"    {model_name:<8s}  acc={r['acc']:.4f}  "
                  f"PF={r['pf']:.3f}  return={r['total_return_pct']:+.1f}%  "
                  f"switches={r['n_switches']}")

        # Store predictions for bootstrap
        test_prices = splits['test']['prices']
        window_preds[w['name']] = {}
        for name, clf in sklearn_clfs.items():
            window_preds[w['name']][name] = (clf.predict(splits['test']['X']), test_prices)

    # Bootstrap CIs for models that pass the gate
    print("\n" + "=" * 70)
    print("BOOTSTRAP CONFIDENCE INTERVALS")
    print("=" * 70)

    bootstrap_results = {}
    for m in CANDIDATE_MODELS:
        pfs = [all_results[w['name']][m]['pf'] for w in WINDOWS]
        if all(pf > PF_GATE for pf in pfs):
            print(f"\n  {m} passes gate — computing bootstrap CIs...")
            bootstrap_results[m] = {}
            for w in WINDOWS:
                wname = w['name']
                if m in window_preds.get(wname, {}):
                    preds, prices = window_preds[wname][m]
                    bs = bootstrap_pf(preds, prices, args.fee)
                    bootstrap_results[m][wname] = bs
                    print(f"    {wname}: PF={all_results[wname][m]['pf']:.3f}  "
                          f"95% CI=[{bs['ci_lo']:.3f}, {bs['ci_hi']:.3f}]")
        else:
            fail_windows = [w['name'] for w, pf in zip(WINDOWS, pfs) if pf <= PF_GATE]
            print(f"\n  {m} fails gate in {fail_windows} — skipping bootstrap")

    # Summary + verdict
    verdict, winner = print_summary(all_results, WINDOWS, bootstrap_results)

    # Save winning model
    ckpt_dir = Path("checkpoints/bc_gold_15min")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if winner and winner in ('LogReg', 'GBT'):
        # Retrain winner on full Jan-Oct data (W3 training set) for deployment
        print(f"\n  Retraining {winner} on full Jan-Oct for deployment checkpoint...")
        w3 = WINDOWS[-1]
        final_splits = make_splits(df, features, direction, mid_prices, w3)
        # Merge train + val for final model
        X_final = np.concatenate([final_splits['train']['X'], final_splits['val']['X']])
        y_final = np.concatenate([final_splits['train']['y'], final_splits['val']['y']])

        if winner == 'LogReg':
            final_clf = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
        else:
            final_clf = GradientBoostingClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        final_clf.fit(X_final, y_final)

        ckpt_path = ckpt_dir / f"best_{winner.lower()}.joblib"
        joblib.dump({
            'model': final_clf,
            'model_name': winner,
            'input_dim': features.shape[1],
            'feature_names': list(features_df.columns),
            'fee_bps': args.fee,
            'train_period': f"{w3['train_start']} -> {w3['train_end']}",
            'verdict': verdict,
            'walk_forward_pfs': {w['name']: all_results[w['name']][winner]['pf']
                                 for w in WINDOWS},
            'bootstrap_cis': {wn: {'lo': bs['ci_lo'], 'hi': bs['ci_hi']}
                              for wn, bs in bootstrap_results.get(winner, {}).items()},
        }, ckpt_path)
        print(f"  Checkpoint saved: {ckpt_path}")

        # Also print the model coefficients for LogReg (interpretability)
        if winner == 'LogReg':
            print(f"\n  LogReg coefficients (Long class):")
            coefs = final_clf.coef_[0]
            feat_names = list(features_df.columns)
            sorted_idx = np.argsort(np.abs(coefs))[::-1]
            for i in sorted_idx:
                print(f"    {feat_names[i]:20s}: {coefs[i]:+.4f}")

    elapsed = time.time() - t_start
    print(f"\n  Total runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
