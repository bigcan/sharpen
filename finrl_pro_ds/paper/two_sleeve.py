"""Two-sleeve (momentum + rates-carry) paper executor — the fund-of-funds rung-1 path.

Wires the validated **rates-carry** sleeve (Treasury-curve carry+roll over
{SHY,IEF,TLT,LQD}; net SR 0.467, corr-to-momentum 0.014, S553-cont-45/53) into the
rung-1 paper executor alongside the existing momentum core, as a **fund-of-funds**
(operator decision 2026-06-18: full env-render combine, not additive conviction):

  1. drive EACH sleeve through the env independently (``linear_core_trajectory`` — the
     env applies its per-asset vol-targeting + gross cap on that sleeve's OWN universe);
  2. risk-parity-combine the two post-vol-scale weight trajectories with a
     **trailing-causal inverse-vol** scalar recomputed monthly and held between
     (``allocator_factory.risk_parity_alphas`` / ``combine_sleeve_weights``);
  3. book the combined weights over the 19-asset union through :class:`PaperState`
     (the pinned env-accounting transcription) — the combined sim oracle.

Parity model (mirrors the single-sleeve :class:`ParityHarness`):
  - :meth:`run` — the BATCH historical book + accounting tautology. The combined book
    has NO single env drive (two independent vol-scalings can't be one drive), so the
    env-accounting authority for the combined positions IS ``PaperState``; its fidelity
    to the env is proven PER SLEEVE (``test_parity_vs_linear_core`` for momentum; the
    rates-sleeve accounting-parity test) + the combine math unit tests. ``run`` therefore
    validates the WIRING (drive→combine→replay→gates) and is parity-0 by construction.
  - :meth:`run_independent_recompute` — the LOAD-BEARING forward-path check (step-4). It
    re-derives EACH sleeve's held conviction independently on a growing window and the
    risk-parity α from the forward sleeve returns, recombines, and replays; a calendar /
    truncation / look-ahead bug in EITHER sleeve's forward assembly moves the combined
    weights and is caught by ``weight_l1_drift``. The Tier-2 re-audit (NEXT roadmap #2)
    runs THIS on the 2-sleeve book before any paper-capital promotion.

The returned ``sim``/``live`` are plug-compatible with :meth:`ParityHarness.compare` and
:func:`evaluate_paper_soak_gates` (the union replay computes per-CLASS attribution incl.
SHY→rates out of the box); per-SLEEVE attribution is attached separately.
"""
from __future__ import annotations

import logging
from typing import Callable, Mapping

import numpy as np

from finrl_pro_ds.envs.allocator_factory import (
    combine_sleeve_weights,
    drive_with_conviction,
    linear_core_trajectory,
    risk_parity_alphas,
)
from finrl_pro_ds.paper.fill_engine import FillEngine
from finrl_pro_ds.paper.paper_state import LiveTrajectory
from finrl_pro_ds.paper.parity_harness import ParityHarness

logger = logging.getLogger(__name__)

_SLEEVES = ("momentum", "rates_carry")


class TwoSleeveExecutor:
    """Drives the momentum + rates-carry fund-of-funds and produces the combined rung-1
    sim oracle + forward replay. Composes a :class:`ParityHarness` for the (reused)
    union replay / compare / month-end-calendar machinery."""

    def __init__(self, config: Mapping) -> None:
        self.config = config
        self.harness = ParityHarness(config)
        rp = dict(config.get("risk_parity", {}))
        self.rp_window = int(rp.get("trailing_window", 252))
        self.rp_min_periods = int(rp.get("min_periods", 63))
        self.rp_monthly_meta = bool(rp.get("monthly_meta", True))
        # None = convex inverse-vol (no portfolio-vol overlay; the executor's declined-
        # overlay default). A float levers the combined book toward that annualized vol.
        tv = rp.get("target_portfolio_vol", None)
        self.rp_target_vol = float(tv) if tv is not None else None
        self.max_gross = float(dict(config.get("env", {})).get("max_gross_exposure", 3.0))

    # ------------------------------------------------------------------ #
    def _alphas(self, sleeve_returns: Mapping[str, np.ndarray], ts_decision: np.ndarray) -> dict:
        return risk_parity_alphas(
            sleeve_returns, ts_decision, window=self.rp_window,
            min_periods=self.rp_min_periods, monthly_meta=self.rp_monthly_meta,
            target_portfolio_vol=self.rp_target_vol)

    def _combine(self, sleeve_weights, sleeve_assets, sleeve_returns, union_assets, ts_decision):
        alphas = self._alphas(sleeve_returns, ts_decision)
        combined = combine_sleeve_weights(
            sleeve_weights, sleeve_assets, alphas, union_assets,
            max_gross_exposure=self.max_gross)
        return combined, alphas

    @staticmethod
    def _sleeve_assets(bundle: Mapping) -> dict[str, list[str]]:
        return {s: list(bundle[s]["assets"]) for s in _SLEEVES}

    @staticmethod
    def _decision_ts(bundle: Mapping) -> np.ndarray:
        """Per-step DECISION stamps (one per of the T-1 steps) — the union calendar's
        bars 0..T-2 (weights w[k] are decided at bar k, applied k→k+1)."""
        return np.asarray(bundle["union"]["timestamps"], dtype=np.int64)[:-1]

    # ------------------------------------------------------------------ #
    def sim_oracle(self, bundle: Mapping) -> tuple[dict, dict]:
        """Batch combined oracle: drive each sleeve, risk-parity-combine, replay over the
        union. Returns ``(oracle_dict, detail)`` where ``oracle_dict`` is compare-ready
        (``weights/step_returns/equity_curve/cumulative_fees/turnovers``) and ``detail``
        carries the per-sleeve trajectories, the α paths, and the combined weights."""
        traj = {s: linear_core_trajectory(bundle[s], self.config) for s in _SLEEVES}
        sleeve_w = {s: traj[s]["weights"] for s in _SLEEVES}
        sleeve_r = {s: traj[s]["step_returns"] for s in _SLEEVES}
        sleeve_assets = self._sleeve_assets(bundle)
        union_assets = list(bundle["union"]["assets"])
        ts_dec = self._decision_ts(bundle)

        combined_w, alphas = self._combine(sleeve_w, sleeve_assets, sleeve_r, union_assets, ts_dec)
        live = self.harness._replay(bundle["union"], combined_w, fill_engine=None)
        oracle = _as_oracle(live)
        detail = {"sleeve_traj": traj, "alphas": alphas, "combined_w": combined_w,
                  "trajectory": live,
                  "sleeve_pnl": self._sleeve_attribution(
                      sleeve_w, sleeve_assets, alphas, union_assets, bundle["union"]["price_ary"])}
        return oracle, detail

    def run(self, bundle: Mapping, *, fill_engine: FillEngine | None = None) -> tuple[LiveTrajectory, dict]:
        """Batch rung-1 book: ``(live_trajectory, sim_oracle_dict)``. Accounting-tautology
        (parity ≈ 0 by construction); use :meth:`run_independent_recompute` for the
        load-bearing forward-path check. ``fill_engine`` overrides the default sim engine."""
        oracle, detail = self.sim_oracle(bundle)
        # Replay the SAME combined weights forward through a fresh book — rung-1 batch path.
        live = self.harness._replay(bundle["union"], detail["combined_w"], fill_engine=fill_engine)
        self.harness._expected_rebalance_ts = self.harness._true_month_end_ts(
            np.asarray(bundle["union"]["timestamps"], dtype=np.int64))
        live.sleeve_pnl = detail["sleeve_pnl"]  # attach per-sleeve attribution for the digest
        return live, oracle

    def run_independent_recompute(
        self,
        bundle: Mapping,
        *,
        fill_engine: FillEngine | None = None,
        conviction_fns: Mapping[str, Callable[[Mapping, int], np.ndarray]] | None = None,
    ) -> tuple[LiveTrajectory, dict]:
        """Step-4 forward-path parity — makes ``weight_l1_drift`` LOAD-BEARING on the
        2-sleeve book. Re-derives EACH sleeve's held conviction INDEPENDENTLY on a growing
        window (``ParityHarness._assemble_forward_conviction``), drives the same env, and
        recomputes the risk-parity α from the forward sleeve returns before combining. A
        correct forward assembly reproduces the batch oracle (drift ≈ 0); a calendar /
        look-ahead bug in either sleeve diverges and is caught.

        ``conviction_fns`` maps a sleeve name → a per-rebalance recompute (default: the SAFE
        confirmed-month-end reader for both). Tests inject a deliberately-broken reader for
        one sleeve to prove the escape exists.
        """
        oracle, _ = self.sim_oracle(bundle)
        fns = dict(conviction_fns or {})
        sleeve_assets = self._sleeve_assets(bundle)
        union_assets = list(bundle["union"]["assets"])
        ts_dec = self._decision_ts(bundle)

        sleeve_w_fwd, sleeve_r_fwd = {}, {}
        for s in _SLEEVES:
            conv_live = self.harness._assemble_forward_conviction(
                bundle[s], conviction_fn=fns.get(s))
            fwd = drive_with_conviction(bundle[s], self.config, conv_live)
            sleeve_w_fwd[s] = fwd["weights"]
            sleeve_r_fwd[s] = fwd["step_returns"]

        combined_w, _ = self._combine(sleeve_w_fwd, sleeve_assets, sleeve_r_fwd, union_assets, ts_dec)
        live = self.harness._replay(bundle["union"], combined_w, fill_engine=fill_engine)
        self.harness._expected_rebalance_ts = self.harness._true_month_end_ts(
            np.asarray(bundle["union"]["timestamps"], dtype=np.int64))
        return live, oracle

    def compare(self, live: LiveTrajectory, sim: Mapping):
        """Delegates to the underlying :meth:`ParityHarness.compare` (the four
        ``paper_soak.parity`` metrics, combined live vs combined oracle)."""
        return self.harness.compare(live, sim)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _sleeve_attribution(
        sleeve_weights: Mapping[str, np.ndarray], sleeve_assets: Mapping[str, list[str]],
        alphas: Mapping[str, np.ndarray], union_assets: list[str], union_price: np.ndarray,
    ) -> dict[str, float]:
        """Cumulative gross-return contribution of EACH sleeve to the combined book:
        ``Σ_k α_s(k) · Σ_{a∈sleeve_s} w_s[k,a] · ret_union[k+1,a]``. Unambiguous even when
        sleeves overlap on a bond ETF (each sleeve's own weight is attributed to it). The
        per-CLASS drift gate is computed separately by the union replay (SHY→rates)."""
        price = np.asarray(union_price, dtype=np.float64)
        prev, cur = price[:-1], price[1:]
        valid = (prev > 1e-10) & (cur > 1e-10)
        ret = np.zeros_like(prev)
        np.divide(cur, prev, out=ret, where=valid)
        ret = np.where(valid, ret - 1.0, 0.0)            # (K, U) union asset returns
        idx = {a: i for i, a in enumerate(union_assets)}
        out: dict[str, float] = {}
        for s, w in sleeve_weights.items():
            cols = [idx[a] for a in sleeve_assets[s]]
            a_s = np.asarray(alphas[s], dtype=np.float64)[:, None]
            contrib = (a_s * np.asarray(w, dtype=np.float64)) * ret[:, cols]
            out[s] = float(contrib.sum())
        return out


def _as_oracle(live: LiveTrajectory) -> dict:
    """Expose a combined :class:`LiveTrajectory` as a compare-ready oracle dict (the keys
    :meth:`ParityHarness.compare` reads)."""
    return {
        "weights": live.weights,
        "step_returns": live.step_returns,
        "equity_curve": live.equity_curve,
        "cumulative_fees": live.cumulative_fees,
        "turnovers": live.turnovers,
    }
