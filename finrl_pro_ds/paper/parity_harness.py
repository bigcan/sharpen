"""Parity harness: the rung-1 forward path vs the ``evaluate_linear_core`` oracle.

Step 3 of the paper-executor build (S553-cont-47; spec
``.agent/artifacts/paper_executor_spec.md``, ADR-7). The whole value of the paper
soak is sim↔live fidelity. The harness:

  1. **sim oracle** — drives the frozen linear core through the env ONCE
     (``allocator_factory.linear_core_trajectory``) to get the byte-identical weight
     trajectory + equity curve the executor must reproduce.
  2. **forward replay** — steps a fresh :class:`PaperState` book through the same
     daily bars (operator decision 2026-06-13: shadow ``evaluate_linear_core``
     VERBATIM — daily vol-rescale of the monthly conviction; the env-processed weight
     ``W[k]`` is the day-k target, ``W[-1]`` the live forward target), booking the
     small daily deltas via a :class:`FillEngine`.
  3. **compare** — the four pre-registered ``paper_soak.parity`` metrics:
     ``weight_l1_drift``, ``daily_return_te_bps``, ``missed_rebalances``,
     ``cost_drift_ratio``.

Because the replay's target weights come from the SAME ``_linear_core_drive`` path as
the oracle, and the book reproduces the env accounting, rung-1 parity is ≈0 **by
construction**.

**SCOPE (Tier-2 audit 2026-06-14: P3-01 / P10-01 / P8-03 / P11-04 / P1-09).** That 0 is
**load-bearing for ACCOUNTING only** — it proves ``PaperState.step_bar`` reproduces
``MultiAssetAllocatorEnv.step`` on identical weights (a real independent-reimplementation
check: ``daily_return_te_bps`` and the equity curve genuinely diverge if the book is
wrong). It is **tautological on the WEIGHT/COST axis**: ``weight_l1_drift`` is 0 because
the replay consumes the oracle's *own* weights (``min_trade_pct=0``), and
``cost_drift_ratio`` is 1.0 because ``SimFillEngine`` is a line-for-line transcription of
the env cost model.

**:meth:`ParityHarness.run_independent_recompute` (step-4) makes ``weight_l1_drift``
load-bearing on the WEIGHT axis** — it does NOT consume the oracle's weights; instead it
reconstructs the held-conviction series INDEPENDENTLY on a growing live-fetched window (the
live scheduler's view) and drives the same env with it. A correct forward assembly
reproduces the oracle (drift ≈ 0); a calendar / truncation / look-ahead bug (e.g. the
P2-01 in-progress-tail trap) diverges and is caught. The COST axis (``cost_drift_ratio``)
stays sim-tautological at 1.0 until rung-2's real IB fills. Promote to capital only after
running this variant under a Tier-2 re-audit (NEXT roadmap #2).

The default :meth:`run` path validates ACCOUNTING only; its look-ahead guarantee comes from
``tests/paper/test_paper_causality.py`` + ``test_forward_prefix_consistency``, NOT from a
green ``run``-path ``weight_l1_drift``. Do NOT read a green rung-1 ``run`` ``weight_l1_drift``
/ ``cost_drift`` as forward-path validation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from finrl_pro_ds.envs.allocator_factory import drive_with_conviction, linear_core_trajectory
from finrl_pro_ds.paper.fill_engine import FillEngine, SimFillEngine
from finrl_pro_ds.paper.paper_state import LiveTrajectory, PaperState, generate_orders

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParityReport:
    """The four ``paper_soak.parity`` metrics + their support.

    ``weight_l1_drift`` is ``Σ_i|w_live − w_sim|`` per step; we report the MAX (the
    gate is "at each rebalance") and the mean. ``daily_return_te_bps`` is
    ``|r_live − r_sim|`` per day in bps (mean + max). ``cost_drift_ratio`` is realized
    one-way cost / modeled cost (== 1.0 in rung-1 sim by construction; the rung-2 IB
    engine makes it informative). ``missed_rebalances`` counts scheduled month-end
    bars the live path failed to process (0 in a full replay).
    """

    weight_l1_drift_max: float
    weight_l1_drift_mean: float
    daily_return_te_bps_mean: float
    daily_return_te_bps_max: float
    missed_rebalances: int
    cost_drift_ratio: float
    n_steps: int

    def as_dict(self) -> dict:
        return {
            "weight_l1_drift_max": self.weight_l1_drift_max,
            "weight_l1_drift_mean": self.weight_l1_drift_mean,
            "daily_return_te_bps_mean": self.daily_return_te_bps_mean,
            "daily_return_te_bps_max": self.daily_return_te_bps_max,
            "missed_rebalances": self.missed_rebalances,
            "cost_drift_ratio": self.cost_drift_ratio,
            "n_steps": self.n_steps,
        }


class ParityHarness:
    """Runs the sim oracle + the rung-1 forward replay and compares them."""

    def __init__(self, config: Mapping) -> None:
        self.config = config
        env_cfg = dict(config.get("env", {}))
        # taker_fee → taker_fee_pct rename tolerated (mirrors allocator_factory._ENV_KEY_MAP).
        self.taker_fee_pct = float(env_cfg.get("taker_fee_pct", env_cfg.get("taker_fee", 0.0002)))
        self.slippage_base_bps = float(env_cfg.get("slippage_base_bps", 1.0))
        self.slippage_impact_bps = float(env_cfg.get("slippage_impact_bps", 5.0))
        self.initial_capital = float(env_cfg.get("initial_capital", 100_000.0))
        # The full intended month-end calendar, stashed by run() for compare()'s
        # missed_rebalances (None until run() has executed — see _true_month_end_ts).
        self._expected_rebalance_ts: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    def sim_oracle(self, arrays: Mapping) -> dict:
        """The byte-identical sim ground truth (env-driven frozen linear core)."""
        return linear_core_trajectory(arrays, self.config)

    def default_fill_engine(self) -> SimFillEngine:
        return SimFillEngine(
            taker_fee_pct=self.taker_fee_pct,
            slippage_base_bps=self.slippage_base_bps,
            slippage_impact_bps=self.slippage_impact_bps,
        )

    # ------------------------------------------------------------------ #
    def run(self, arrays: Mapping, *, fill_engine: FillEngine | None = None) -> tuple[LiveTrajectory, dict]:
        """Drive the oracle once and replay the book forward over the same window.

        Returns ``(live_trajectory, sim_oracle_dict)`` ready for :meth:`compare`.
        """
        sim = self.sim_oracle(arrays)
        live = self._replay(arrays, sim["weights"], fill_engine=fill_engine)
        # Stash the FULL intended month-end calendar (P3-02) for compare()'s
        # missed_rebalances — derived from the complete arrays span, NOT from what was
        # processed, so a (step-4) scheduler that skips a scheduled month-end is detectable.
        # In the rung-1 full replay every month-end is processed ⇒ genuinely 0 (not vacuous).
        self._expected_rebalance_ts = self._true_month_end_ts(
            np.asarray(arrays["timestamps"], dtype=np.int64))
        return live, sim

    # ------------------------------------------------------------------ #
    def run_independent_recompute(
        self,
        arrays: Mapping,
        *,
        fill_engine: FillEngine | None = None,
        conviction_fn: Callable[[Mapping, int], np.ndarray] | None = None,
    ) -> tuple[LiveTrajectory, dict]:
        """Step-4 forward-path parity variant — makes ``weight_l1_drift`` LOAD-BEARING.

        Unlike :meth:`run` (which feeds the replay the oracle's OWN weights, so
        ``weight_l1_drift`` is 0 by construction — accounting-only; Tier-2 audit
        P3-01/P10-01/P8-03/P11-04), this re-derives the live weight series
        **independently**: it walks the harness's own true-month-end calendar
        (:meth:`_true_month_end_ts`) and at each rebalance recomputes the held
        conviction on the GROWING window ``arrays[:t+1]`` (the live scheduler's view),
        assembles the daily held-conviction series, and drives the SAME env once with it
        via :func:`~finrl_pro_ds.envs.allocator_factory.drive_with_conviction`. The env
        re-applies only the deterministic daily vol-scale (already validated), so the only
        independently-reconstructed part is the monthly conviction — exactly where the
        forward calendar/truncation risk lives.

        A CORRECT forward assembly reproduces the oracle weights (``weight_l1_drift`` ≈ 0,
        legitimately); a calendar / truncation / look-ahead bug in the growing-window
        assembly (e.g. the P2-01 in-progress-tail trap, or peeking a future bar) moves the
        weights and ``weight_l1_drift`` > 0 CATCHES it. This is the config the Tier-2
        re-audit (NEXT roadmap #2) requires before rung-2 capital; the rung-1 ``run`` path
        validates ACCOUNTING only.

        ``conviction_fn(arrays, t) -> (N,)`` is the per-rebalance recompute (default
        :meth:`_safe_monthend_conviction`, the SAFE confirmed-month-end reader). Tests
        inject a deliberately-broken reader to prove the escape (drift > 0) exists.

        Returns ``(live_trajectory, sim_oracle_dict)`` ready for :meth:`compare` — where
        ``compare`` now measures the genuine live-vs-oracle weight divergence.
        """
        sim = self.sim_oracle(arrays)
        conv_live = self._assemble_forward_conviction(arrays, conviction_fn=conviction_fn)
        drive = drive_with_conviction(arrays, self.config, conv_live)
        live = self._replay(arrays, drive["weights"], fill_engine=fill_engine)
        self._expected_rebalance_ts = self._true_month_end_ts(
            np.asarray(arrays["timestamps"], dtype=np.int64))
        return live, sim

    def _assemble_forward_conviction(
        self, arrays: Mapping, *,
        conviction_fn: Callable[[Mapping, int], np.ndarray] | None = None,
    ) -> np.ndarray:
        """Reconstruct the daily held-conviction series a LIVE scheduler would carry, by
        recomputing on a GROWING window at each true month-end — INDEPENDENT of the
        oracle's batch ``monthly_rebal_conviction``. Returns ``(T, N)`` to feed
        :func:`drive_with_conviction`.

        Between rebalances the most recent month-end conviction is held (the env then
        vol-scales it daily). When ``conviction_fn`` is the default SAFE reader this
        reproduces the batch ``monthly_rebal_conviction`` exactly; a broken reader (or a
        wrong calendar) diverges and is caught downstream by ``weight_l1_drift``.
        """
        ts = np.asarray(arrays["timestamps"], dtype=np.int64)
        conv_raw = np.asarray(arrays["conviction_ary"], dtype=np.float64)
        T, N = conv_raw.shape
        rebal_ts = {int(t) for t in self._true_month_end_ts(ts)}
        fn = conviction_fn or self._safe_monthend_conviction
        out = np.zeros((T, N), dtype=np.float64)
        held = np.zeros(N, dtype=np.float64)
        for t in range(T):
            if int(ts[t]) in rebal_ts:
                held = np.asarray(fn(arrays, t), dtype=np.float64).ravel()
            out[t] = held
        return out

    @staticmethod
    def _safe_monthend_conviction(arrays: Mapping, t: int) -> np.ndarray:
        """The conviction a live scheduler decides at true month-end bar ``t`` from the
        GROWING window ``[0..t]`` only — SAFE: the recompute runs on the truncated slice
        and takes the value at the CONFIRMED month-end bar ``t``, never an in-progress
        tail (cf. the :func:`monthly_rebal_conviction` P2-01 caveat). Reproduces the batch
        ``conv_monthly`` exactly when the forward assembly is correct."""
        conv_window = np.asarray(arrays["conviction_ary"], dtype=np.float64)[:t + 1]
        return conv_window[-1]

    def _asset_meta(self, arrays: Mapping, n_assets: int) -> tuple[list[str], dict[str, str]]:
        assets = list(arrays.get("assets") or self.config.get("universe", {}).get("assets")
                      or [f"asset_{i}" for i in range(n_assets)])
        if len(assets) != n_assets:
            assets = [f"asset_{i}" for i in range(n_assets)]
        asset_class = dict(self.config.get("universe", {}).get("asset_class")
                           or {a: "all" for a in assets})
        return assets, asset_class

    def _replay(self, arrays: Mapping, target_weights: np.ndarray,
                *, fill_engine: FillEngine | None) -> LiveTrajectory:
        price = np.asarray(arrays["price_ary"], dtype=np.float64)
        volume = np.asarray(arrays["volume_ary"], dtype=np.float64)     # DOLLAR volume (F1)
        carry = np.asarray(arrays["carry_ary"], dtype=np.float64)
        ts = np.asarray(arrays["timestamps"], dtype=np.int64)
        T, N = price.shape
        W = np.asarray(target_weights, dtype=np.float64)
        # The asset axis MUST match (a real shape bug); the TIME axis may be SHORT when the
        # oracle terminated early on the env circuit-break (PV < 0.1×capital) — replay the
        # covered prefix and flag coverage_incomplete instead of crashing (P10-03; the old
        # hard `assert W.shape == (T-1, N)` raised an unhandled AssertionError on a DD-flatten).
        if W.ndim != 2 or W.shape[1] != N:
            raise ValueError(f"target_weights {W.shape} incompatible with {N} assets")
        n_replay = min(T - 1, W.shape[0])
        coverage_incomplete = W.shape[0] < (T - 1)
        if coverage_incomplete:
            logger.warning(
                "parity replay: target_weights has %d rows < %d bars (early-terminated / "
                "circuit-broken oracle) — replaying the covered %d-step prefix; parity is "
                "valid only over it (coverage_incomplete)", W.shape[0], T - 1, n_replay)

        engine = fill_engine or self.default_fill_engine()
        assets, asset_class = self._asset_meta(arrays, N)
        book = PaperState(n_assets=N, initial_capital=self.initial_capital, assets=assets)

        weights, equity = [], [self.initial_capital]
        rets, turns, cumfees, gross, net, stamps = [], [], [], [], [], []
        for k in range(n_replay):
            prev_price, price_now = price[k], price[k + 1]
            pv_before = book.pv_before(prev_price)
            # W[k] is already the env's post-dust, post-cap realized position, so book
            # the RAW delta (no re-deadband): delta == the env's applied weight change.
            delta = generate_orders(W[k], book.positions, min_trade_pct=0.0)
            fill = engine.fill(
                delta_weights=delta, ref_prices=price_now,
                pv_before=pv_before, dollar_volume=volume[k],   # env vol_idx = max((k+1)-1,0) = k
            )
            info = book.step_bar(
                delta_weights=delta, fill=fill, prev_price=prev_price, price_now=price_now,
                carry_rates=carry[k + 1], pv_before=pv_before, as_of_ts=int(ts[k + 1]),
            )
            weights.append(book.positions.copy())
            equity.append(info["portfolio_value"])
            rets.append(info["step_return"])
            turns.append(info["turnover"])
            cumfees.append(book.cumulative_fees)
            gross.append(info["gross_exposure"])
            net.append(info["net_exposure"])
            stamps.append(int(ts[k + 1]))

        # Attribute over the COVERED prefix only (W[:n_replay] / price[:n_replay+1]) so a
        # circuit-broken short replay keeps class_pnl / spy_returns aligned to step_returns.
        class_pnl, spy_returns = self._attribution(
            W[:n_replay], price[:n_replay + 1], assets, asset_class)
        return LiveTrajectory(
            weights=np.asarray(weights, dtype=np.float64),
            equity_curve=np.asarray(equity, dtype=np.float64),
            step_returns=np.asarray(rets, dtype=np.float64),
            turnovers=np.asarray(turns, dtype=np.float64),
            cumulative_fees=np.asarray(cumfees, dtype=np.float64),
            gross_exposure=np.asarray(gross, dtype=np.float64),
            net_exposure=np.asarray(net, dtype=np.float64),
            timestamps=np.asarray(stamps, dtype=np.int64),
            class_pnl=class_pnl,
            assets=assets,
            asset_class=asset_class,
            initial_capital=self.initial_capital,
            spy_returns=spy_returns,
            coverage_incomplete=coverage_incomplete,
        )

    @staticmethod
    def _asset_returns(price: np.ndarray) -> np.ndarray:
        """(T-1, N) per-asset daily returns, zero where either price is invalid (the
        env zeroes the weight there, so the attribution contribution is zero too)."""
        prev, cur = price[:-1], price[1:]
        valid = (prev > 1e-10) & (cur > 1e-10)
        rets = np.zeros_like(prev)
        np.divide(cur, prev, out=rets, where=valid)
        rets = np.where(valid, rets - 1.0, 0.0)
        return np.nan_to_num(rets, nan=0.0, posinf=0.0, neginf=0.0)

    def _attribution(self, W: np.ndarray, price: np.ndarray, assets: list[str],
                     asset_class: dict[str, str]) -> tuple[dict[str, float], np.ndarray | None]:
        """Cumulative gross-return attribution per class (``Σ_k Σ_{i∈c} W[k,i]·ret[k+1,i]``)
        and SPY daily returns for the corr-to-SPY drift gate (None if SPY absent)."""
        rets = self._asset_returns(price)              # (T-1, N)
        contrib = W * rets                             # (T-1, N) gross attribution
        class_pnl: dict[str, float] = {}
        for c in sorted(set(asset_class.get(a, "all") for a in assets)):
            idx = [i for i, a in enumerate(assets) if asset_class.get(a, "all") == c]
            class_pnl[c] = float(contrib[:, idx].sum()) if idx else 0.0
        spy_returns = rets[:, assets.index("SPY")] if "SPY" in assets else None
        return class_pnl, spy_returns

    # ------------------------------------------------------------------ #
    def compare(self, live: LiveTrajectory, sim: Mapping) -> ParityReport:
        """The four ``paper_soak.parity`` metrics, live forward path vs sim oracle."""
        w_live, w_sim = live.weights, np.asarray(sim["weights"], dtype=np.float64)
        r_live, r_sim = live.step_returns, np.asarray(sim["step_returns"], dtype=np.float64)
        n = min(len(w_live), len(w_sim))
        if n == 0:
            return ParityReport(0.0, 0.0, 0.0, 0.0, 0, 1.0, 0)

        l1 = np.abs(w_live[:n] - w_sim[:n]).sum(axis=1)
        te_bps = np.abs(r_live[:n] - r_sim[:n]) * 1e4

        sim_cost = float(np.asarray(sim["cumulative_fees"])[-1]) if len(sim["cumulative_fees"]) else 0.0
        live_cost = float(live.cumulative_fees[-1]) if live.n_steps else 0.0
        if sim_cost > 1e-12:
            cost_drift_ratio = live_cost / sim_cost
        else:
            cost_drift_ratio = 1.0 if live_cost <= 1e-12 else float("inf")

        # Restrict the stashed full-span month-end calendar to the live soak window so
        # warmup / index-0 edge bars cannot create a spurious miss; fall back to the
        # (self-derived, rung-1-vacuous) calendar only if run() was not used.
        expected = self._expected_rebalance_ts
        if expected is not None and live.n_steps:
            lo, hi = int(live.timestamps.min()), int(live.timestamps.max())
            expected = expected[(expected >= lo) & (expected <= hi)]
        missed = self._missed_rebalances(live.timestamps, expected_ts=expected)

        return ParityReport(
            weight_l1_drift_max=float(l1.max()),
            weight_l1_drift_mean=float(l1.mean()),
            daily_return_te_bps_mean=float(te_bps.mean()),
            daily_return_te_bps_max=float(te_bps.max()),
            missed_rebalances=missed,
            cost_drift_ratio=cost_drift_ratio,
            n_steps=n,
        )

    @staticmethod
    def _true_month_end_ts(timestamps: np.ndarray) -> np.ndarray:
        """The true monthly-rebalance calendar of a timestamp span: the LAST actual
        trading day of each (year, month) present. This is the *intended* calendar a
        live scheduler must hit — independent of which bars were actually processed,
        so a dropped month-end is detectable (cf. the self-derived calendar that is a
        subset of `processed` by construction and can never report a miss)."""
        ts = np.asarray(timestamps, dtype=np.int64)
        if ts.size == 0:
            return np.empty(0, dtype=np.int64)
        months = pd.to_datetime(ts, unit="s").to_period("M")
        return pd.Series(ts).groupby(months.values).max().to_numpy().astype(np.int64)

    @staticmethod
    def _missed_rebalances(processed_ts: np.ndarray, expected_ts: np.ndarray | None = None) -> int:
        """Scheduled month-end bars the live path did not process.

        Rung-1 replay processes every bar, so the expected month-ends (derived from the
        processed span) are all present ⇒ 0. The step-4 scheduler will pass an explicit
        ``expected_ts`` calendar; a skipped month-end then shows up here.
        """
        processed = set(int(t) for t in processed_ts)
        if expected_ts is None:
            if len(processed_ts) == 0:
                return 0
            dates = pd.to_datetime(np.asarray(processed_ts, dtype=np.int64), unit="s")
            months = dates.to_period("M")
            expected = pd.Series(np.asarray(processed_ts, dtype=np.int64)).groupby(months.values).max().to_numpy()
        else:
            expected = np.asarray(expected_ts, dtype=np.int64)
        return int(sum(1 for t in expected if int(t) not in processed))
