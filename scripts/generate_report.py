import argparse
import json
import os
import re
import sqlite3
import statistics as _st

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(d, *keys, default=None, fmt=None, scale=1):
    """Walk a chain of fallback keys. Return formatted value or default."""
    for k in keys:
        v = d.get(k)
        if v is not None:
            v = v * scale if isinstance(v, (int, float)) and scale != 1 else v
            if fmt and isinstance(v, (int, float)):
                return fmt.format(v)
            return v
    return default


def _pct(d, *keys, default="N/A"):
    """Return a percentage-formatted value from a 0-1 ratio."""
    v = _get(d, *keys)
    if v is None:
        return default
    return f"{v * 100:.2f}%"


def _round(d, *keys, n=2, default="N/A"):
    """Return a rounded float or default marker."""
    v = _get(d, *keys)
    if v is None:
        return default
    return round(v, n)


def _classify_run(tags, config):
    """Classify run as Pilot or Production using SKILL heuristics."""
    tags_lower = [t.lower() for t in (tags or [])]

    # Priority 1: explicit tags
    if any(t in tags_lower for t in ("dev", "localtest", "pilot", "tier1-validation")):
        return "Pilot"
    if "production" in tags_lower:
        return "Production"

    # Priority 2: total_timesteps
    steps = (config.get("training", {}).get("total_timesteps", 0) or 0)
    if steps <= 500_000:
        return "Pilot"
    if steps >= 1_000_000:
        return "Production"

    # Priority 3: num_envs
    envs = (config.get("env", {}).get("num_envs", 0) or 0)
    if envs <= 16:
        return "Pilot"

    return "Production"


def _deviation(val, test):
    """Sharpe deviation % (absolute)."""
    try:
        val_f = float(val)
        test_f = float(test)
        if val_f == 0:
            return "N/A"
        return f"{abs(val_f - test_f) / abs(val_f) * 100:.1f}%"
    except (ValueError, TypeError, ZeroDivisionError):
        return "N/A"


def _overfitting_diagnosis(val_s, test_s, dev_str):
    """Return emoji + diagnosis string for overfitting assessment."""
    try:
        val_f = float(val_s)
        test_f = float(test_s)
        dev_s = dev_str.replace("%", "").strip()
        dev_f = float(dev_s) if dev_s != "N/A" else None
    except (ValueError, TypeError):
        return "⚠️ Insufficient Data", "Unable to assess — metrics missing."

    if val_f > 0 and test_f > 0:
        if dev_f is not None and dev_f < 30:
            return "✅ Healthy Generalization", "Both positive with <30% deviation."
        elif dev_f is not None and dev_f < 50:
            return "⚠️ Moderate Degradation", "Positive on both but deviation warrants investigation."
        else:
            return "🚨 Significant Overfitting", "Large deviation suggests poor generalization."
    elif val_f > 0 and test_f <= 0:
        return "🚨 Catastrophic Overfitting", "Positive val collapses to negative on test."
    elif val_f <= 0 and test_f > 0:
        return "⚠️ Inverted (Noise)", "Negative val, positive test — likely noise from small sample."
    else:
        return "❌ Both Negative", "Strategy failed on both val and test."


_WF_FOLD_RE = re.compile(r"^win(\d+)/wf_fold_(\d{2})/profit_factor$")


def _wf_folds(summary):
    """Extract per-fold metrics from a WF-consolidated run.

    Returns a list of fold dicts sorted by (window, fold), or [] if not WF.
    """
    hits = []
    for k in summary:
        m = _WF_FOLD_RE.match(k)
        if m:
            hits.append((int(m.group(1)), int(m.group(2))))
    if not hits:
        return []
    folds = []
    for win_i, fold_i in sorted(hits):
        fp = f"win{win_i}/wf_fold_{fold_i:02d}"
        tp = f"win{win_i}/train"
        folds.append({
            "win": win_i,
            "fold": fold_i,
            "pf": summary.get(f"{fp}/profit_factor"),
            "dd": summary.get(f"{fp}/max_drawdown"),
            "ret": summary.get(f"{fp}/total_return"),
            "sharpe_d": summary.get(f"{fp}/sharpe_daily"),
            "sortino": summary.get(f"{fp}/sortino"),
            "trades": summary.get(f"{fp}/trade_count"),
            "win_rate": summary.get(f"{fp}/win_rate"),
            "stability": summary.get(f"{fp}/stability"),
            "exposure": summary.get(f"{fp}/market_exposure"),
            "sps": summary.get(f"{tp}/sps"),
            "steps": summary.get(f"{tp}/total_steps"),
            "agent_type": summary.get(f"{tp}/agent_type"),
            "taker_fee": summary.get(f"{tp}/taker_fee"),
            "checkpoint": summary.get(f"{tp}/checkpoint"),
        })
    return folds


def _wf_aggregate(folds):
    pfs = [f["pf"] for f in folds if isinstance(f["pf"], (int, float))]
    dds = [f["dd"] for f in folds if isinstance(f["dd"], (int, float))]
    rets = [f["ret"] for f in folds if isinstance(f["ret"], (int, float))]
    shs = [f["sharpe_d"] for f in folds if isinstance(f["sharpe_d"], (int, float))]
    trades = [f["trades"] for f in folds if isinstance(f["trades"], (int, float))]

    def cv(xs):
        if len(xs) < 2:
            return None
        m = _st.mean(xs)
        if m == 0:
            return None
        return _st.stdev(xs) / abs(m)

    return {
        "n_folds": len(folds),
        "n_profitable": sum(1 for p in pfs if p > 1.0),
        "n_pf_ge_11": sum(1 for p in pfs if p >= 1.1),
        "median_pf": _st.median(pfs) if pfs else None,
        "min_pf": min(pfs) if pfs else None,
        "max_pf": max(pfs) if pfs else None,
        "cv_pf": cv(pfs),
        "mean_return": _st.mean(rets) if rets else None,
        "total_return": sum(rets) if rets else None,
        "worst_dd": min(dds) if dds else None,
        "median_sharpe_d": _st.median(shs) if shs else None,
        "total_trades": sum(trades) if trades else None,
    }


def _status_icon(status):
    if status == "finished":
        return "✅ Finished"
    elif status == "crashed":
        return "💥 Crashed"
    elif status == "failed":
        return "❌ Failed"
    elif status == "running":
        return "🔄 Running"
    return status


def _get_comparison_runs(current_run_id, n=5):
    """Query metrics.db for recent runs to build comparison table."""
    db_path = os.path.join(os.getcwd(), "results", "metrics.db")
    if not os.path.exists(db_path):
        return []

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()
        # Get column names first to handle flexible schemas
        cursor.execute("PRAGMA table_info(runs)")
        cols = {row[1] for row in cursor.fetchall()}

        needed = {"id", "name", "created_at", "status"}
        if not needed.issubset(cols):
            conn.close()
            return []

        cursor.execute(
            "SELECT id, name, created_at, status, "
            "  val_sharpe, test_sharpe, test_return, test_trade_count "
            "FROM runs "
            "WHERE id != ? "
            "ORDER BY created_at DESC LIMIT ?",
            (current_run_id, n),
        )
        rows = cursor.fetchall()
        conn.close()
        return rows
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Walk-Forward Generator (WandB consolidation format, S488+)
# ---------------------------------------------------------------------------

def _generate_wf_report(run_id, output_path, data, config, tags, name, status,
                        classification, folds):
    """Render a fold-aware report for WF-consolidated runs."""
    agg = _wf_aggregate(folds)
    agent_type = (folds[0].get("agent_type") or "unknown").upper()
    steps_per_fold = folds[0].get("steps") or 0
    total_steps = sum((f.get("steps") or 0) for f in folds)
    taker_fee = folds[0].get("taker_fee")
    taker_fee_str = f"{taker_fee*10000:.2f}" if isinstance(taker_fee, (int, float)) else "N/A"

    tags_str = ", ".join(f"`{t}`" for t in tags) if tags else "None"

    # Gate verdict — SG-1 XAUUSD ensemble plan style, reused as a generic WF heuristic.
    g1_pass = (agg["n_pf_ge_11"] or 0) >= max(1, int(0.83 * (agg["n_folds"] or 1)))  # ≥83% folds PF≥1.1
    g_profit = agg["n_profitable"] == agg["n_folds"]
    ftmo_ok = isinstance(agg["worst_dd"], (int, float)) and agg["worst_dd"] > -0.05
    cv_ok = isinstance(agg["cv_pf"], (int, float)) and agg["cv_pf"] <= 0.35

    # Per-fold table
    fold_rows = []
    for f in folds:
        pf = f"{f['pf']:.3f}" if isinstance(f["pf"], (int, float)) else "N/A"
        dd = f"{f['dd']*100:+.2f}%" if isinstance(f["dd"], (int, float)) else "N/A"
        ret = f"{f['ret']*100:+.2f}%" if isinstance(f["ret"], (int, float)) else "N/A"
        sh = f"{f['sharpe_d']:.2f}" if isinstance(f["sharpe_d"], (int, float)) else "N/A"
        trd = int(f["trades"]) if isinstance(f["trades"], (int, float)) else "N/A"
        wr = f"{f['win_rate']:.1f}%" if isinstance(f["win_rate"], (int, float)) else "N/A"
        fold_rows.append(f"| {f['fold']} | {pf} | {dd} | {ret} | {sh} | {trd} | {wr} |")
    fold_table = "\n".join(fold_rows)

    # Aggregate formatters
    def _fmt_pct(x):
        return f"{x*100:+.2f}%" if isinstance(x, (int, float)) else "N/A"
    def _fmt_num(x, n=3):
        return f"{x:.{n}f}" if isinstance(x, (int, float)) else "N/A"
    def _fmt_cv(x):
        return f"{x*100:.2f}%" if isinstance(x, (int, float)) else "N/A"

    data_cfg = config.get("data", {}) or {}
    ticker = data_cfg.get("ticker", "N/A")
    wf_cfg = config.get("walk_forward", {}) or config.get("wf", {}) or {}
    train_window = wf_cfg.get("train_window", data_cfg.get("train_window", "N/A"))
    val_window = wf_cfg.get("val_window", data_cfg.get("val_window", "N/A"))
    test_window = wf_cfg.get("test_window", data_cfg.get("test_window", "N/A"))

    report = f"""# DeepScalper Walk-Forward Report: {name}

**Run ID:** `{run_id}`
**Name:** {name}
**Date:** {data.get("created_at", "N/A")}
**Status:** {_status_icon(status)}
**Classification:** {classification} (WF, {agg['n_folds']} folds)
**WandB URL:** [Link]({data.get("url", "#")})
**Tags:** {tags_str}

---

## 1. Executive Summary

*   **Agent:** {agent_type}
*   **Outcome:** {_status_icon(status)}
*   **WF Folds:** {agg['n_folds']} completed, {agg['n_profitable']} profitable, {agg['n_pf_ge_11']} at PF≥1.1
*   **Median PF:** {_fmt_num(agg['median_pf'])}  (min {_fmt_num(agg['min_pf'])} / max {_fmt_num(agg['max_pf'])})
*   **Worst Fold MaxDD:** {_fmt_pct(agg['worst_dd'])}
*   **Median Daily Sharpe:** {_fmt_num(agg['median_sharpe_d'], n=2)}

---

## 2. Walk-Forward Fold Results

| Fold | PF | MaxDD | Return | Sharpe_d | Trades | WinRate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{fold_table}

### 2a. Aggregate Statistics

| Metric | Value |
| :--- | :--- |
| **Folds Completed** | {agg['n_folds']} |
| **Folds Profitable (PF>1.0)** | {agg['n_profitable']} / {agg['n_folds']} |
| **Folds PF ≥ 1.1** | {agg['n_pf_ge_11']} / {agg['n_folds']} |
| **Median Fold PF** | {_fmt_num(agg['median_pf'])} |
| **Min / Max Fold PF** | {_fmt_num(agg['min_pf'])} / {_fmt_num(agg['max_pf'])} |
| **CV(PF) across folds** | {_fmt_cv(agg['cv_pf'])} |
| **Mean Fold Return** | {_fmt_pct(agg['mean_return'])} |
| **Sum Fold Returns** | {_fmt_pct(agg['total_return'])} |
| **Worst Fold MaxDD** | {_fmt_pct(agg['worst_dd'])} |
| **Median Fold Daily Sharpe** | {_fmt_num(agg['median_sharpe_d'], n=2)} |
| **Total Trades** | {int(agg['total_trades']) if isinstance(agg['total_trades'], (int, float)) else 'N/A'} |

---

## 3. Operational Telemetry (Training)

| Metric | Value |
| :--- | :--- |
| **Agent** | {agent_type} |
| **Steps / Fold** | {steps_per_fold:,} |
| **Total Steps (all folds)** | {total_steps:,} |
| **Mean SPS** | {_fmt_num(_st.mean([f['sps'] for f in folds if isinstance(f['sps'], (int, float))]) if any(isinstance(f['sps'], (int, float)) for f in folds) else None, n=1)} |
| **Taker Fee** | {taker_fee_str} bps |

---

## 4. Configuration Highlights

*   **Agent:** {agent_type}
*   **Data:** {ticker}
*   **WF Windows:** train={train_window}, val={val_window}, test={test_window}
*   **Fold Count:** {agg['n_folds']}

---

## 5. Gate Assessment

| Gate | Threshold | Result | Verdict |
| :--- | :--- | :--- | :--- |
| **All Folds Profitable** | PF>1.0 for all | {agg['n_profitable']}/{agg['n_folds']} | {'✅' if g_profit else '❌'} |
| **Coverage PF≥1.1** | ≥83% of folds | {agg['n_pf_ge_11']}/{agg['n_folds']} | {'✅' if g1_pass else '❌'} |
| **FTMO Daily DD** | worst fold > -5% | {_fmt_pct(agg['worst_dd'])} | {'✅' if ftmo_ok else '❌'} |
| **CV(PF)** | ≤ 35% | {_fmt_cv(agg['cv_pf'])} | {'✅' if cv_ok else '❌'} |

---

## 6. Automated Decision
"""
    passes = sum([g_profit, g1_pass, ftmo_ok, cv_ok])
    if passes == 4:
        report += "\n✅ **WF PASS**: All generic gates clear. Proceed to ensemble eval / stress sidecar.\n"
    elif passes >= 2:
        report += f"\n⚠️ **PARTIAL**: {passes}/4 gates passed. Investigate failing gates before promotion.\n"
    else:
        report += f"\n❌ **WF FAIL**: {passes}/4 gates passed. Do not promote.\n"

    # Optional: comparison runs section (reuse existing helper)
    comp_rows = _get_comparison_runs(run_id, n=5)
    if comp_rows:
        report += "\n---\n\n## 7. Comparison to Previous Runs\n\n"
        report += "| Run | Date | Status |\n| :--- | :--- | :--- |\n"
        for row in comp_rows:
            rid, _, rdate, rstatus = row[0], row[1], row[2], row[3]
            short_date = str(rdate)[:10] if rdate else "N/A"
            report += f"| `{rid}` | {short_date} | {rstatus} |\n"

    report += "\n---\n\n*Report auto-generated by `generate_report.py` (WF mode) — enrich with behavioral diagnosis and recommendations as needed.*\n"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Report generated: {output_path}")


# ---------------------------------------------------------------------------
# Main Generator
# ---------------------------------------------------------------------------

def generate_report(run_id, output_path):
    # ── 1. Load Data ──────────────────────────────────────────────────────
    json_path = os.path.join(os.getcwd(), "results", f"run_data_{run_id}.json")
    if not os.path.exists(json_path):
        print(f"Error: Data file not found at {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    summary = data.get("summary", {})
    config = data.get("config", {})
    tags = data.get("tags", [])
    name = data.get("name", "Unknown Run")
    status = data.get("status", "unknown")

    # ── 2. Classify ───────────────────────────────────────────────────────
    classification = _classify_run(tags, config)

    # ── 2a. WF short-circuit ──────────────────────────────────────────────
    # WF-consolidated runs (S488+) nest metrics under win<i>/wf_fold_<NN>/*,
    # so the standard backtest_val/backtest_test keys don't exist. Render a
    # fold-aware report instead and return early.
    folds = _wf_folds(summary)
    if folds:
        _generate_wf_report(
            run_id=run_id,
            output_path=output_path,
            data=data,
            config=config,
            tags=tags,
            name=name,
            status=status,
            classification=classification,
            folds=folds,
        )
        return

    # ── 3. Extract Metrics ────────────────────────────────────────────────
    # Financial — Validation
    val_sharpe     = _round(summary, "backtest_val/sharpe", "backtest_validation/sharpe", "backtest/validation_sharpe")
    val_sortino    = _round(summary, "backtest_val/sortino", "backtest_validation/sortino")
    val_calmar     = _round(summary, "backtest_val/calmar", "backtest_validation/calmar")
    val_omega      = _round(summary, "backtest_val/omega", "backtest_validation/omega", n=4)
    val_return     = _pct(summary, "backtest_val/total_return", "backtest_validation/total_return")
    val_maxdd      = _pct(summary, "backtest_val/max_drawdown", "backtest_validation/max_drawdown")
    val_trades     = _get(summary, "backtest_val/trade_count", "backtest_validation/trade_count", default="N/A")
    val_winrate    = _round(summary, "backtest_val/win_rate", "backtest_validation/win_rate")
    val_pf         = _round(summary, "backtest_val/profit_factor", "backtest_validation/profit_factor", n=4)
    val_stability  = _round(summary, "backtest_val/stability", "backtest_validation/stability", n=3)
    val_var        = _pct(summary, "backtest_val/minute_var", "backtest_validation/minute_var")
    val_exposure   = _pct(summary, "backtest_val/market_exposure", "backtest_validation/market_exposure")

    # Financial — Test
    test_sharpe    = _round(summary, "backtest_test/sharpe", "backtest/test_sharpe")
    test_sortino   = _round(summary, "backtest_test/sortino")
    test_calmar    = _round(summary, "backtest_test/calmar")
    test_omega     = _round(summary, "backtest_test/omega", n=4)
    test_return    = _pct(summary, "backtest_test/total_return")
    test_maxdd     = _pct(summary, "backtest_test/max_drawdown")
    test_trades    = _get(summary, "backtest_test/trade_count", default="N/A")
    test_winrate   = _round(summary, "backtest_test/win_rate")
    test_pf        = _round(summary, "backtest_test/profit_factor", n=4)
    test_stability = _round(summary, "backtest_test/stability", n=3)
    test_var       = _pct(summary, "backtest_test/minute_var")
    test_exposure  = _pct(summary, "backtest_test/market_exposure")

    # Deviation
    dev_sharpe     = _deviation(val_sharpe, test_sharpe)

    # Operational telemetry
    sps            = _round(summary, "train/sps", n=0, default="N/A")
    total_steps    = _get(summary, "train/global_step", "step", default=0)
    reward_mean    = _round(summary, "train/reward_mean", n=2, default="N/A")
    final_loss     = _round(summary, "agent/loss_total", n=4, default="N/A")
    policy_loss    = _round(summary, "agent/policy_loss", n=6, default="N/A")
    value_loss     = _round(summary, "agent/value_loss", n=2, default="N/A")
    entropy        = _round(summary, "agent/entropy", n=3, default="N/A")
    approx_kl      = _round(summary, "agent/approx_kl", n=4, default="N/A")
    clip_frac      = _pct(summary, "agent/clip_fraction")
    lr             = _get(summary, "agent/learning_rate", fmt="{:.2e}", default="N/A")

    # Agent type
    agent_type     = _get(summary, "train/agent_type", default="unknown").upper()

    # Config
    num_envs       = config.get("env", {}).get("num_envs", "N/A")
    total_timesteps = (
        config.get("training", {}).get("total_timesteps")
        or config.get("agents", {}).get("total_timesteps")
        or 0
    )
    try:
        total_timesteps = int(total_timesteps)
    except (TypeError, ValueError):
        total_timesteps = 0
    maker_fee      = config.get("env", {}).get("maker_fee_bps", config.get("env", {}).get("maker_fee", "N/A"))
    taker_fee      = config.get("env", {}).get("taker_fee_bps", config.get("env", {}).get("taker_fee", "N/A"))

    # Data range
    data_cfg       = config.get("data", {})
    train_start    = data_cfg.get("train_start_date", "N/A")
    train_end      = data_cfg.get("train_end_date", "N/A")
    val_start      = data_cfg.get("val_start_date", "N/A")
    val_end        = data_cfg.get("val_end_date", "N/A")
    test_start     = data_cfg.get("test_start_date", "N/A")
    test_end       = data_cfg.get("test_end_date", "N/A")
    ticker         = data_cfg.get("ticker", "N/A")

    # Tags string
    tags_str       = ", ".join(f"`{t}`" for t in tags) if tags else "None"

    # ── 4. Overfitting Diagnosis ──────────────────────────────────────────
    diag_label, diag_text = _overfitting_diagnosis(val_sharpe, test_sharpe, dev_sharpe)

    # ── 5. Build Report ───────────────────────────────────────────────────
    report = f"""# DeepScalper {classification} Report: {name}

**Run ID:** `{run_id}`
**Name:** {name}
**Date:** {data.get("created_at", "N/A")}
**Status:** {_status_icon(status)}
**Classification:** {classification}
**WandB URL:** [Link]({data.get("url", "#")})
**Tags:** {tags_str}

---

## 1. Executive Summary

*   **Classification:** {classification} ({total_timesteps:,} steps, {num_envs} envs)
*   **Agent:** {agent_type}
*   **Outcome:** {_status_icon(status)}
*   **Performance:** Test Sharpe **{test_sharpe}** (Target: > 1.0)

---

## 2. Financial Performance (Backtest)

### 2a. Return & Risk Ratios
| Metric | Validation | Test | Deviation |
| :--- | :--- | :--- | :--- |
| **Sharpe Ratio** | {val_sharpe} | {test_sharpe} | {dev_sharpe} |
| **Sortino Ratio** | {val_sortino} | {test_sortino} | - |
| **Calmar Ratio** | {val_calmar} | {test_calmar} | - |
| **Omega Ratio** | {val_omega} | {test_omega} | - |
| **Total Return** | {val_return} | {test_return} | - |
| **Max Drawdown** | {val_maxdd} | {test_maxdd} | - |

### 2b. Trade Quality
| Metric | Validation | Test |
| :--- | :--- | :--- |
| **Win Rate** | {val_winrate} | {test_winrate} |
| **Profit Factor** | {val_pf} | {test_pf} |
| **Trade Count** | {val_trades} | {test_trades} |

### 2c. Risk Metrics
| Metric | Validation | Test |
| :--- | :--- | :--- |
| **Stability (R²)** | {val_stability} | {test_stability} |
| **Minute VaR (5%)** | {val_var} | {test_var} |
| **Market Exposure** | {val_exposure} | {test_exposure} |

---

## 3. Operational Telemetry (Training)

| Metric | Value | Status |
| :--- | :--- | :--- |
| **Throughput (SPS)** | {sps} steps/s | {"✅" if isinstance(sps, (int, float)) and sps > 58 else "⚠️"} |
| **Total Steps** | {total_steps:,} | {"✅ Complete" if total_steps and total_steps > 0 else "⚠️ Zero steps"} |
| **Final Loss** | {final_loss} | - |
| **Policy Loss** | {policy_loss} | - |
| **Value Loss** | {value_loss} | - |
| **Entropy** | {entropy} | - |
| **Approx KL** | {approx_kl} | - |
| **Clip Fraction** | {clip_frac} | - |
| **Learning Rate** | {lr} | - |
| **Reward Mean** | {reward_mean} | {"🚨" if isinstance(reward_mean, (int, float)) and reward_mean < 0 else "✅"} |

---

## 4. Configuration Highlights

*   **Agent:** {agent_type}
*   **Environment:** num_envs={num_envs}
*   **Fees:** maker={maker_fee} bps, taker={taker_fee} bps
*   **Training:** {total_timesteps:,} steps
*   **Data:** {ticker}, Train: {train_start} → {train_end}, Val: {val_start} → {val_end}, Test: {test_start} → {test_end}

---

## 5. Overfitting Assessment

| Val Sharpe | Test Sharpe | Deviation | Diagnosis |
| :--- | :--- | :--- | :--- |
| {val_sharpe} | {test_sharpe} | {dev_sharpe} | **{diag_label}** |

{diag_text}

---

## 6. Automated Decision
"""
    if isinstance(test_sharpe, (int, float)) and isinstance(val_sharpe, (int, float)):
        if test_sharpe >= 1.0 and val_sharpe >= 1.0:
            report += "\n✅ **GOAL ACHIEVED**: Run meets production criteria (Val ≥ 1.0, Test ≥ 1.0).\n"
        elif test_sharpe >= 0 and val_sharpe >= 0:
            report += "\n⚠️ **PROMISING**: Both Sharpe positive but below 1.0 target. Scale training or tune HPO.\n"
        else:
            report += "\n❌ **GOAL MISSED**: Performance below target thresholds.\n"
    else:
        report += "\n⚠️ **UNABLE TO ASSESS**: Sharpe metrics not available.\n"

    # ── 6. HPO Summary (if available) ─────────────────────────────────────
    artifact_dir = os.path.join(os.getcwd(), "results", run_id)
    hpo_db_path = os.path.join(artifact_dir, "hpo.db")
    if os.path.exists(hpo_db_path):
        try:
            conn = sqlite3.connect(f"file:{hpo_db_path}?mode=ro", uri=True)
            cursor = conn.cursor()

            cursor.execute("SELECT count(*) FROM trials")
            n_trials = cursor.fetchone()[0]

            cursor.execute("SELECT count(*) FROM trials WHERE state='COMPLETE'")
            n_complete = cursor.fetchone()[0]

            cursor.execute("SELECT max(value) FROM trial_values")
            best_val = cursor.fetchone()[0]

            conn.close()

            report += f"""
---

## 7. HPO Summary

*   **Trials:** {n_complete}/{n_trials} completed
*   **Best Objective Value:** {best_val if best_val is not None else 'N/A'}
*   *Detailed trial data available in `results/{run_id}/hpo.db`*
"""
        except Exception as e:
            report += f"\n---\n\n## 7. HPO Summary\n\n*⚠️ Error reading HPO DB: {e}*\n"

    # ── 7. Remote Log Tail (if available) ─────────────────────────────────
    log_path = os.path.join(artifact_dir, "run.log")
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - 2048))
                tail = f.read()

            report += f"""
---

## 8. Remote Log (Tail)

```text
{tail}
```
"""
        except Exception as e:
            report += f"\n---\n\n## 8. Remote Log\n\n*⚠️ Error reading log: {e}*\n"

    # ── 8. Comparison to Previous Runs ────────────────────────────────────
    comp_rows = _get_comparison_runs(run_id, n=5)
    if comp_rows:
        report += "\n---\n\n## 9. Comparison to Previous Runs\n\n"
        report += "| Run | Date | Status | Val Sharpe | Test Sharpe | Test Return | Trades |\n"
        report += "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
        for row in comp_rows:
            rid, _rname, rdate, rstatus = row[0], row[1], row[2], row[3]
            r_vs = row[4] if row[4] is not None else "N/A"
            r_ts = row[5] if row[5] is not None else "N/A"
            r_tr = row[6] if row[6] is not None else "N/A"
            r_tc = row[7] if row[7] is not None else "N/A"
            short_date = str(rdate)[:10] if rdate else "N/A"
            report += f"| `{rid}` | {short_date} | {rstatus} | {r_vs} | {r_ts} | {r_tr} | {r_tc} |\n"

    report += "\n---\n\n*Report auto-generated by `generate_report.py` — enrich with behavioral diagnosis and recommendations as needed.*\n"

    # ── 9. Save Report ────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Report generated: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a standardized DeepScalper report from WandB JSON data.",
    )
    parser.add_argument("--run_id", required=True, help="WandB 8-char run ID")
    parser.add_argument("--output", required=True, help="Output markdown path")
    args = parser.parse_args()

    generate_report(args.run_id, args.output)
