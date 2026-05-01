"""AlphaSeek Phase 6 Realistic-Fee Audit (Stage A.2).

Replays the 12 Phase 6 HPO `full/best` checkpoints on their test segments
under a sweep of realistic fee/slippage configs:

    contest            : taker 0.007 bps, half-spread 0 bps     (Phase 6 baseline sanity check)
    spot_vip0_retail   : taker 10.0  bps, half-spread 0.5 bps   (Binance SPOT retail)
    spot_vip0_bnb      : taker  7.5  bps, half-spread 0.5 bps   (spot + BNB discount)
    spot_vip3          : taker  4.5  bps, half-spread 0.5 bps   (spot VIP-3)
    perp_vip0_retail   : taker  4.0  bps, half-spread 0.3 bps   (Binance PERP retail)
    perp_vip0_bnb      : taker  3.6  bps, half-spread 0.3 bps   (perp + BNB discount)
    perp_vip3          : taker  2.0  bps, half-spread 0.3 bps   (perp VIP-3)

Outputs:
    results/alphaseek_fee_audit.json        machine-readable full sweep
    docs/alphaseek_fee_audit_report.md      human-readable summary + verdict

Usage:
    python scripts/alphaseek_fee_audit.py                  # full sweep (12×7 = 84 evals)
    python scripts/alphaseek_fee_audit.py --smoke          # 1 checkpoint × contest only
    python scripts/alphaseek_fee_audit.py --windows 3      # only window 3 (12 evals)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch as th

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from finrl_pro_ds.alphaseek.agent_wrapper import AlphaSeekAgent  # noqa: E402
from finrl_pro_ds.alphaseek.lob_trade_simulator import (  # noqa: E402
    EvalLOBTradeSimulator,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fee_audit")

CKPT_DIR = PROJECT_ROOT / "results" / "alphaseek_hpo" / "checkpoints"
LOB_PARQUET = PROJECT_ROOT / "data" / "lob_parquet" / "btcusdt_lob_1s.parquet"
OUT_JSON = PROJECT_ROOT / "results" / "alphaseek_fee_audit.json"
OUT_REPORT = PROJECT_ROOT / "docs" / "alphaseek_fee_audit_report.md"

AGENTS = ["D3QN", "DoubleDQN", "TwinD3QN"]
WINDOWS = [0, 1, 2, 3]

# Phase 6 walk-forward schedule (from alphaseek_hpo_runner.build_window_schedule
# with train_segments=8, val_segments=1, test_segments=1, slide_by=1, n_segments=13)
WF_TEST_SEGS = {0: [9], 1: [10], 2: [11], 3: [12]}
WF_TRAIN_SEGS = {
    0: [0, 1, 2, 3, 4, 5, 6, 7],
    1: [1, 2, 3, 4, 5, 6, 7, 8],
    2: [2, 3, 4, 5, 6, 7, 8, 9],
    3: [3, 4, 5, 6, 7, 8, 9, 10],
}

FEE_CONFIGS: dict[str, dict[str, float]] = {
    "contest":          {"taker_bps": 0.007, "half_spread_bps": 0.0, "label": "Phase 6 baseline (contest)"},
    "spot_vip0_retail": {"taker_bps": 10.0,  "half_spread_bps": 0.5, "label": "Binance SPOT retail"},
    "spot_vip0_bnb":    {"taker_bps": 7.5,   "half_spread_bps": 0.5, "label": "SPOT + BNB 25% discount"},
    "spot_vip3":        {"taker_bps": 4.5,   "half_spread_bps": 0.5, "label": "SPOT VIP-3"},
    "perp_vip0_retail": {"taker_bps": 4.0,   "half_spread_bps": 0.3, "label": "Binance PERP retail"},
    "perp_vip0_bnb":    {"taker_bps": 3.6,   "half_spread_bps": 0.3, "label": "PERP + BNB 10% discount"},
    "perp_vip3":        {"taker_bps": 2.0,   "half_spread_bps": 0.3, "label": "PERP VIP-3"},
}

# Verdict gate thresholds (per plan)
VERDICT_GO_PF = 1.2          # median PF >= 1.2 AND n_profitable >= 8/12
VERDICT_GO_PROFITABLE = 8
VERDICT_AMBIG_PF_LOW = 1.0   # median PF in [1.0, 1.2)


class FeeAwareEvalSim(EvalLOBTradeSimulator):
    """EvalLOBTradeSimulator that deducts taker_fee + half-spread from reward.

    reward = (new_asset - old_asset) - fee_cost - half_spread_cost
      where
        fee_cost         = |action_int * mid_price| * taker_bps * 1e-4
        half_spread_cost = |action_int * mid_price| * half_spread_bps * 1e-4

    Taker fees fire on EVERY non-zero action_int (open, close, reverse).
    Half-spread is the cost of crossing the book at taker speed (also per fill).
    """

    def __init__(
        self,
        *args,
        taker_bps: float = 0.0,
        half_spread_bps: float = 0.0,
        **kwargs,
    ):
        # Set slippage=0.0 — fees dominate at realistic levels. The base
        # class's `slippage` param kept at 0 so we only apply our fee model.
        kwargs.setdefault("slippage", 0.0)
        super().__init__(*args, **kwargs)
        self.taker_bps = float(taker_bps)
        self.half_spread_bps = float(half_spread_bps)
        self._total_fees = 0.0
        self._total_fills = 0

    def _step(self, action, _if_random=True):
        # Reproduce base _step but add fee/half-spread deduction at reward time.
        self.step_i += self.step_gap
        step_is = self.step_is + self.step_i

        action = action.squeeze(1).to(self.device)
        action_int = action - 1  # (0, 1, 2) -> (-1, 0, +1)
        del action

        old_cash = self.cash
        old_asset = self.asset
        old_position = self.position

        mid_price = self.price_ary[step_is, 2]

        truncated = self.step_i >= (self.max_step * self.step_gap)
        if truncated:
            action_int = -old_position
        else:
            new_position = (old_position + action_int).clip(
                -self.max_position, self.max_position,
            )
            action_int = new_position - old_position
            done_mask = (new_position * old_position).lt(0) & old_position.ne(0)
            if done_mask.sum() > 0:
                action_int[done_mask] = -old_position[done_mask]

        self.holding = self.holding + 1
        mask_max_holding = self.holding.gt(self.max_holding)
        if mask_max_holding.sum() > 0:
            action_int[mask_max_holding] = -old_position[mask_max_holding]
        self.holding[old_position == 0] = 0

        # Stop-loss (same as base)
        direction_mask1 = old_position.gt(0)
        if direction_mask1.sum() > 0:
            _best = th.max(
                th.stack([self.best_price[direction_mask1], mid_price[direction_mask1]]),
                dim=0,
            )[0]
            self.best_price[direction_mask1] = _best

        direction_mask2 = old_position.lt(0)
        if direction_mask2.sum() > 0:
            _best = th.min(
                th.stack([self.best_price[direction_mask2], mid_price[direction_mask2]]),
                dim=0,
            )[0]
            self.best_price[direction_mask2] = _best

        sl_mask1 = th.logical_and(
            direction_mask1, (self.best_price - mid_price).gt(self.stop_loss_thresh),
        )
        sl_mask2 = th.logical_and(
            direction_mask2, (mid_price - self.best_price).gt(self.stop_loss_thresh),
        )
        sl_mask = th.logical_or(sl_mask1, sl_mask2)
        if sl_mask.sum() > 0:
            action_int[sl_mask] = -old_position[sl_mask]

        new_position = old_position + action_int

        entry_mask = old_position.eq(0)
        if entry_mask.sum() > 0:
            self.best_price[entry_mask] = mid_price[entry_mask]

        direction = action_int.gt(0)
        cost = action_int * mid_price
        new_cash = old_cash - cost * th.where(
            direction, 1 + self.slippage, 1 - self.slippage,
        )

        # Fee + half-spread deduction on the notional of the fill.
        fill_notional = th.abs(action_int.float() * mid_price)
        fee_cost = fill_notional * (self.taker_bps * 1e-4)
        half_spread_cost = fill_notional * (self.half_spread_bps * 1e-4)
        total_cost = fee_cost + half_spread_cost

        new_cash = new_cash - total_cost
        new_asset = new_cash + new_position * mid_price

        reward = new_asset - old_asset

        self.cash = new_cash
        self.asset = new_asset
        self.position = new_position
        self.action_int = action_int

        # Track aggregate fees
        self._total_fees += float(total_cost.sum().item())
        self._total_fills += int((action_int != 0).sum().item())

        state = self.get_state(step_is)
        info_dict = {}
        if truncated:
            terminal = th.ones_like(self.position, dtype=th.bool)
            state = self.reset()
        else:
            terminal = th.zeros_like(self.position, dtype=th.bool)

        return state, reward, terminal, info_dict


def load_hpo_params(window: int, agent: str) -> dict:
    """Load step_gap + stop_loss_thresh from best_<agent>.json."""
    with open(CKPT_DIR / f"w{window}" / f"best_{agent}.json") as f:
        best = json.load(f)
    return best["params"]


def eval_one_checkpoint(
    window: int,
    agent: str,
    fee_key: str,
    fee_cfg: dict[str, float],
    device: str = "cpu",
) -> dict:
    """Run one deterministic eval and return metrics.

    Returns dict with keys: total_return, sharpe, sortino, profit_factor,
    max_drawdown, win_rate, hold_rate, n_trades, n_steps, total_fees,
    total_fills.
    """
    params = load_hpo_params(window, agent)
    step_gap = int(params["step_gap"])
    stop_loss_thresh = float(params["stop_loss_thresh"])

    ckpt_dir = CKPT_DIR / f"w{window}" / agent / "full" / "best"
    if not (ckpt_dir / "act.pth").exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_dir}")

    # Build eval simulator on the test segment for this window
    test_segs = WF_TEST_SEGS[window]
    sim = FeeAwareEvalSim(
        lob_parquet_path=str(LOB_PARQUET),
        num_sims=64,
        slippage=0.0,
        taker_bps=fee_cfg["taker_bps"],
        half_spread_bps=fee_cfg["half_spread_bps"],
        max_position=1,
        step_gap=step_gap,
        num_ignore_step=60,
        seq_len=3600,
        stop_loss_thresh=stop_loss_thresh,
        norm_span=120,
        momentum_window=5,
        vol_window=30,
        device=th.device(device),
        gpu_id=-1 if device == "cpu" else 0,
        segment_filter=test_segs,
    )

    # Load agent (full nn.Module — arch inferred from checkpoint)
    agent_obj = AlphaSeekAgent(
        agent_type=agent, net_dims=(256, 256), state_dim=sim.state_dim,
        action_dim=sim.action_dim, device=device,
    )
    agent_obj.load(str(ckpt_dir))

    # Deterministic rollout
    state = sim.reset()
    if not isinstance(state, th.Tensor):
        state = th.tensor(state, dtype=th.float32, device=sim.device)

    all_rewards: list[float] = []
    all_actions: list[float] = []
    step = 0
    max_step = sim.max_step

    with th.no_grad():
        for _ in range(max_step):
            q_vals = agent_obj.act(state)
            if q_vals.dim() == 3:
                q_vals = q_vals.squeeze(1)
            action = q_vals.argmax(dim=1, keepdim=True)
            state, reward, terminal, _ = sim.step(action)
            all_rewards.append(float(reward.mean().item()))
            all_actions.append(float(action.float().mean().item()))
            step += 1
            if terminal.any():
                break

    rewards = np.array(all_rewards, dtype=np.float64)
    actions = np.array(all_actions, dtype=np.float64)

    total_return = float(rewards.sum())

    if len(rewards) > 1 and rewards.std() > 1e-12:
        sharpe = float(rewards.mean() / rewards.std())
    else:
        sharpe = 0.0

    neg = rewards[rewards < 0]
    if len(neg) > 1 and neg.std() > 1e-12:
        sortino = float(rewards.mean() / neg.std())
    else:
        sortino = 0.0 if rewards.mean() <= 0 else float("inf")

    pos_sum = float(rewards[rewards > 0].sum()) if (rewards > 0).any() else 0.0
    neg_sum = float(abs(rewards[rewards < 0].sum())) if (rewards < 0).any() else 0.0
    profit_factor = pos_sum / max(neg_sum, 1e-12)

    cum = np.cumsum(rewards)
    peak = np.maximum.accumulate(cum)
    max_dd = float((peak - cum).max()) if len(cum) > 0 else 0.0

    non_hold = rewards != 0
    win_rate = (
        float((rewards[non_hold] > 0).sum() / max(non_hold.sum(), 1))
        if non_hold.any() else 0.0
    )
    hold_rate = float((np.abs(actions - 1.0) < 0.3).mean())
    n_trades = int(non_hold.sum())

    return {
        "window": window,
        "agent": agent,
        "fee_key": fee_key,
        "fee_cfg": fee_cfg,
        "step_gap": step_gap,
        "stop_loss_thresh": stop_loss_thresh,
        "total_return": total_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "profit_factor": profit_factor,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "hold_rate": hold_rate,
        "n_trades": n_trades,
        "n_steps": len(rewards),
        "total_fees_usd": sim._total_fees,
        "total_fills": sim._total_fills,
    }


def render_report(results: list[dict], out_path: Path) -> None:
    """Write a human-readable markdown report."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Organise: fee_key -> list of results
    by_fee: dict[str, list[dict]] = {k: [] for k in FEE_CONFIGS}
    for r in results:
        by_fee[r["fee_key"]].append(r)

    lines: list[str] = []
    lines.append("# AlphaSeek Phase 6 — Realistic-Fee Audit Report")
    lines.append("")
    lines.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    lines.append("**Stage:** A.2 (per 2026-04-18 replan plan)")
    lines.append(f"**Checkpoints:** 12 × full/best (3 agents × 4 WF windows)")
    lines.append(f"**Fee configs:** {len(FEE_CONFIGS)} (contest + 6 realistic)")
    lines.append("")
    lines.append("## Fee sweep summary (median across 12 checkpoints)")
    lines.append("")
    lines.append(
        "| fee_key | taker_bps | half_spread_bps | median PF | min PF | max PF | "
        "median return | n_profitable | total fees ($) |",
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    )
    for fee_key, rs in by_fee.items():
        if not rs:
            continue
        pfs = [r["profit_factor"] for r in rs]
        rets = [r["total_return"] for r in rs]
        fees = [r["total_fees_usd"] for r in rs]
        n_prof = sum(1 for r in rs if r["profit_factor"] > 1.0 and r["total_return"] > 0)
        cfg = FEE_CONFIGS[fee_key]
        lines.append(
            f"| `{fee_key}` | {cfg['taker_bps']:.2f} | {cfg['half_spread_bps']:.2f} | "
            f"{np.median(pfs):.3f} | {min(pfs):.3f} | {max(pfs):.3f} | "
            f"{np.median(rets):.1f} | {n_prof}/{len(rs)} | "
            f"{np.median(fees):.2f} |",
        )

    # Per-checkpoint table, grouped by fee config
    lines.append("")
    lines.append("## Per-checkpoint detail")
    lines.append("")
    for fee_key, rs in by_fee.items():
        if not rs:
            continue
        cfg = FEE_CONFIGS[fee_key]
        lines.append(
            f"### `{fee_key}` — {cfg['label']} "
            f"(taker={cfg['taker_bps']} bps, half-spread={cfg['half_spread_bps']} bps)",
        )
        lines.append("")
        lines.append("| window | agent | PF | return | sharpe | max DD | n_trades | win % | fees ($) |")
        lines.append("|---:|:---|---:|---:|---:|---:|---:|---:|---:|")
        for r in sorted(rs, key=lambda x: (x["window"], x["agent"])):
            lines.append(
                f"| w{r['window']} | {r['agent']} | {r['profit_factor']:.3f} | "
                f"{r['total_return']:.1f} | {r['sharpe']:.4f} | {r['max_drawdown']:.1f} | "
                f"{r['n_trades']} | {r['win_rate']*100:.1f} | {r['total_fees_usd']:.2f} |",
            )
        lines.append("")

    # Verdict per plan A.3
    lines.append("## Verdict (per Stage A.3 gate)")
    lines.append("")
    for key in ["spot_vip0_retail", "perp_vip0_retail"]:
        rs = by_fee.get(key, [])
        if not rs:
            continue
        pfs = [r["profit_factor"] for r in rs]
        med_pf = float(np.median(pfs))
        n_prof = sum(1 for r in rs if r["profit_factor"] > 1.0 and r["total_return"] > 0)
        if med_pf >= VERDICT_GO_PF and n_prof >= VERDICT_GO_PROFITABLE:
            verdict = "GO"
        elif med_pf >= VERDICT_AMBIG_PF_LOW:
            verdict = "AMBIGUOUS"
        else:
            verdict = "NO-GO"
        lines.append(
            f"- **{key}**: median PF = {med_pf:.3f}, n_profitable = {n_prof}/{len(rs)} → "
            f"**{verdict}**",
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- **GO** (median PF ≥ 1.2 AND ≥ 8/12 profitable): proceed to Stage B v2 rebuild with this fee baseline.",
    )
    lines.append(
        "- **AMBIGUOUS** (median PF ∈ [1.0, 1.2)): proceed to Stage B but upgrade to ADR-4 "
        "Option 3 reward shaping (DD-proximity + turnover penalty) from stage 1.",
    )
    lines.append(
        "- **NO-GO** (median PF < 1.0 on majority): halt; pivot to maker-only frame, "
        "retrain on Binance PERP LOB, or retire workstream.",
    )
    lines.append("")
    lines.append(
        "## Baseline reproducibility: `contest` row should approximately match Phase 6 `run_data_k28l6ef8.json` test metrics.",
    )
    lines.append(
        "Small differences expected from: (a) deterministic `num_sims=64` vs Phase 6 `num_sims_eval=64` with possibly different seeds, "
        "(b) random start indices (EvalLOBTradeSimulator uses fixed offset — same across runs), "
        "(c) torch.compile wrapping stripped on save.",
    )
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote report: %s", out_path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="Single checkpoint × contest fee config only (~5 min)")
    ap.add_argument("--windows", nargs="*", type=int, default=None,
                    help="Restrict to specific window ids")
    ap.add_argument("--agents", nargs="*", default=None,
                    help="Restrict to specific agent names")
    ap.add_argument("--fees", nargs="*", default=None,
                    help="Restrict to specific fee_keys")
    ap.add_argument("--device", default="cpu",
                    help="'cpu' or 'cuda:0' (CPU sufficient for small eval)")
    args = ap.parse_args()

    windows = args.windows or WINDOWS
    agents = args.agents or AGENTS
    fee_keys = args.fees or list(FEE_CONFIGS.keys())

    if args.smoke:
        windows, agents, fee_keys = [3], ["D3QN"], ["contest"]

    total = len(windows) * len(agents) * len(fee_keys)
    logger.info(
        "Fee audit sweep: %d windows × %d agents × %d fee configs = %d evals",
        len(windows), len(agents), len(fee_keys), total,
    )

    results: list[dict] = []
    t_start = time.time()

    for wi, w in enumerate(windows):
        for ai, a in enumerate(agents):
            for fi, fk in enumerate(fee_keys):
                idx = wi * len(agents) * len(fee_keys) + ai * len(fee_keys) + fi + 1
                t0 = time.time()
                try:
                    r = eval_one_checkpoint(w, a, fk, FEE_CONFIGS[fk], device=args.device)
                    dt = time.time() - t0
                    logger.info(
                        "[%d/%d] w%d %s %s -> PF=%.3f return=%.1f "
                        "trades=%d fees=$%.2f (%.1fs)",
                        idx, total, w, a, fk,
                        r["profit_factor"], r["total_return"], r["n_trades"],
                        r["total_fees_usd"], dt,
                    )
                    results.append(r)
                except Exception as e:
                    logger.error("[%d/%d] w%d %s %s FAILED: %s", idx, total, w, a, fk, e)

    elapsed = time.time() - t_start
    logger.info("Sweep complete in %.1fs (%d results)", elapsed, len(results))

    # Dump JSON
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "fee_configs": FEE_CONFIGS,
            "windows": windows,
            "agents": agents,
            "fee_keys": fee_keys,
            "results": results,
            "elapsed_seconds": elapsed,
        }, f, indent=2, default=float)
    logger.info("Wrote JSON: %s", OUT_JSON)

    # Markdown report
    render_report(results, OUT_REPORT)

    return 0


if __name__ == "__main__":
    sys.exit(main())
