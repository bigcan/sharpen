"""N-sleeve fund-of-funds paper executor (the cross-account combine).

The top of the multi-sleeve generalization (S553-cont-68; spec
``.agent/artifacts/multi_sleeve_paper_executor_architecture.md``). Combines an
:class:`~sharpen.paper.sleeves.AllocatorBookSleeve` (the ETF/IB account — momentum
⊕ rates, weight-combined internally, UNCHANGED) with zero or more
:class:`~sharpen.paper.sleeves.ReturnStreamSleeve` (the Deribit VRP account) at the
**return / capital-allocation** level — the only correct combine across separate
brokerage accounts (MS-ADR-1).

Combine math (MS-ADR-3, MS-ADR-4 — the Math-gated core):
  1. Align all sleeve streams on the common business-day calendar (intersection — the
     overlap window; pre-coverage bars are dropped, never zero-filled).
  2. ``α = risk_parity_alphas({sleeve: r_sleeve}, common_ts)`` — convex inverse-vol,
     monthly-held, causal (REUSED verbatim; α changes only at month-ends).
  3. **Cross-account compounding (buy-and-hold within the α meta-period, NOT daily
     constant-mix):** at each meta-rebalance (α-change / month-end) the capital is
     re-split ``E_s ← α_s · ΣE``; between, each sub-account compounds at its own return
     ``E_s ← E_s·(1+r_s)``; the combined equity is ``Σ_s E_s`` and ``R = ΔE/E``. This is
     the physically-accurate model for separate accounts (you cannot transfer capital
     between Deribit and IB daily). At a rebalance bar it reduces to ``R = Σ_s α_s·r_s``;
     between bars the weights drift (buy-and-hold). With a single sleeve (VRP disabled,
     α≡1) it reduces to ``R = r_etf`` and ``E = C0·Π(1+r_etf)`` — byte-identical to the
     ETF account's own equity curve (MS-ADR-6).

Parity (mirrors :class:`~sharpen.paper.two_sleeve.TwoSleeveExecutor`):
  - :meth:`run` — batch oracle (consumes each sleeve's ``sim_oracle``); parity ≈ 0 by
    construction; validates the WIRING.
  - :meth:`run_independent_recompute` — the load-bearing forward path (each sleeve's
    ``forward_recompute`` on a growing window); a calendar / look-ahead bug in ANY sleeve
    moves the combined ``R`` (caught by ``daily_return_te_bps``) and, for the ETF book,
    the combined weights (``weight_l1_drift``). VRP forward bugs also perturb α (hence the
    weights) through ``σ_vrp``.

Invariants: LEAK-2 (causal α + causal forward sleeves); MARGIN-CFG (gross from the ETF
weight vector); gates-not-hardcoded (thresholds in yaml, evaluated downstream); no raw
``print``.
"""
from __future__ import annotations

import logging
from typing import Mapping

import numpy as np

from sharpen.envs.allocator_factory import combiner_alphas
from sharpen.paper.paper_state import LiveTrajectory
from sharpen.paper.sleeves import (
    AllocatorBookSleeve,
    ReturnStreamSleeve,
    Sleeve,
    SleeveContext,
)
from sharpen.paper.parity_harness import ParityHarness
from sharpen.paper.two_sleeve import _as_oracle

logger = logging.getLogger(__name__)

_ALPHA_CHANGE_EPS = 1e-12   # α is monthly-held (exact within a month); a change ⇒ a meta-rebalance


def _align_index(ts: np.ndarray, common: np.ndarray) -> np.ndarray:
    """Indices into ``ts`` for each (sorted, ⊆ ts) element of ``common``; raises on a miss."""
    ts = np.asarray(ts, dtype=np.int64)
    common = np.asarray(common, dtype=np.int64)
    idx = np.searchsorted(ts, common)
    if idx.size and (idx.max() >= ts.size or not np.array_equal(ts[idx], common)):
        raise ValueError("sleeve calendar alignment failed (common not a subset of timestamps)")
    return idx


class PortfolioExecutor:
    """Drives N sleeves (one allocator book + ≥0 return-stream sleeves) and produces the
    combined rung-1 sim oracle + forward replay. Composes a :class:`ParityHarness` for the
    reused compare / month-end-calendar machinery."""

    def __init__(self, config: Mapping) -> None:
        self.config = config
        self.harness = ParityHarness(config)
        rp = dict(config.get("risk_parity", {}))
        self.rp_window = int(rp.get("trailing_window", 252))
        self.rp_min_periods = int(rp.get("min_periods", 63))
        self.rp_monthly_meta = bool(rp.get("monthly_meta", True))
        tv = rp.get("target_portfolio_vol", None)
        self.rp_target_vol = float(tv) if tv is not None else None
        # C1.2: optional dynamic-combiner block. Absent ⇒ inverse-vol (back-compat, MS-ADR-6).
        self.sleeve_combiner = dict(config.get("sleeve_combiner", {}))
        self.initial_capital = float(dict(config.get("env", {})).get("initial_capital", 100_000.0))
        uni = dict(config.get("universe", {}))
        self.union_assets = list(uni.get("assets", []))
        self.asset_class = dict(uni.get("asset_class", {a: "all" for a in self.union_assets}))
        self.weight_caps = self._weight_caps(config)
        self.sleeves: list[Sleeve] = self._build_sleeves(config)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_sleeves(config: Mapping) -> list[Sleeve]:
        """One :class:`AllocatorBookSleeve` (the ETF account) + a :class:`VRPSleeve` iff
        ``sleeves.vrp.enabled`` (MS-ADR-6 — VRP stays flag-off until the 2-sleeve soak
        passes its ≥3-month gate)."""
        sleeves: list[Sleeve] = [AllocatorBookSleeve(config)]
        vrp_cfg = dict(config.get("sleeves", {}).get("vrp", {}))
        if vrp_cfg.get("enabled", False):
            from sharpen.paper.sleeves import VRPSleeve
            sleeves.append(VRPSleeve(config))
        return sleeves

    def _streams(self, data_bundle: Mapping, *, oracle: bool, broken: Mapping | None):
        """Run every sleeve (oracle or forward), the allocator sleeve FIRST so its calendar
        becomes the common business-day target the return-stream sleeves resample onto."""
        broken = dict(broken or {})
        alloc = next(s for s in self.sleeves if isinstance(s, AllocatorBookSleeve))
        etf_ctx = SleeveContext(data=data_bundle["etf"])
        etf_stream = (alloc.sim_oracle(etf_ctx) if oracle
                      else alloc.forward_recompute(etf_ctx, broken_assembler=broken.get(alloc.name)))
        cal = etf_stream.timestamps
        streams = {alloc.name: etf_stream}
        for s in self.sleeves:
            if not isinstance(s, ReturnStreamSleeve):
                continue
            ctx = SleeveContext(data=data_bundle.get(s.name, {}), union_timestamps=cal)
            streams[s.name] = (s.sim_oracle(ctx) if oracle
                               else s.forward_recompute(ctx, broken_assembler=broken.get(s.name)))
        return streams

    # ------------------------------------------------------------------ #
    def run(self, data_bundle: Mapping) -> tuple[LiveTrajectory, dict]:
        """Batch combined book: ``(live, oracle_dict)``. Parity ≈ 0 by construction."""
        streams = self._streams(data_bundle, oracle=True, broken=None)
        live = self._combine_streams(streams)
        return live, _as_oracle(live)

    def run_independent_recompute(
        self, data_bundle: Mapping, *, broken: Mapping | None = None,
    ) -> tuple[LiveTrajectory, dict]:
        """Load-bearing forward path: ``(live_fwd, oracle_dict)`` for :meth:`compare`. A
        calendar / look-ahead bug in any sleeve (inject via ``broken={sleeve: assembler}``)
        moves the combined return / weights and is caught."""
        oracle_live = self._combine_streams(self._streams(data_bundle, oracle=True, broken=None))
        live = self._combine_streams(self._streams(data_bundle, oracle=False, broken=broken))
        return live, _as_oracle(oracle_live)

    def compare(self, live: LiveTrajectory, sim: Mapping):
        """The four ``paper_soak.parity`` metrics on the combined book (delegates to
        :meth:`ParityHarness.compare`; ``weight_l1_drift`` from the ETF weight vector,
        ``daily_return_te_bps`` from the combined return)."""
        self.harness._expected_rebalance_ts = self.harness._true_month_end_ts(
            np.asarray(live.timestamps, dtype=np.int64))
        return self.harness.compare(live, sim)

    # ------------------------------------------------------------------ #
    def _alphas(self, returns: Mapping[str, np.ndarray], ts: np.ndarray) -> dict:
        return combiner_alphas(
            returns, ts, window=self.rp_window, min_periods=self.rp_min_periods,
            monthly_meta=self.rp_monthly_meta, target_portfolio_vol=self.rp_target_vol,
            sleeve_combiner=self.sleeve_combiner)

    @staticmethod
    def _weight_caps(config: Mapping) -> dict[str, float]:
        """Per-sleeve α caps (``sleeves.<name>.max_weight``) for return-stream sleeves —
        the VRP cap that stops trailing-vol risk-parity over-allocating to the tail-risk
        sleeve. Keyed by sleeve name (the config key == the sleeve's ``.name``, e.g. ``vrp``)."""
        caps: dict[str, float] = {}
        for key, spec in dict(config.get("sleeves", {})).items():
            spec = spec or {}
            if "max_weight" in spec and str(spec.get("type", "allocator")) == "return_stream":
                caps[key] = float(spec["max_weight"])
        return caps

    def _apply_weight_caps(self, alphas: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Clip capped sleeves to their ``max_weight`` and redistribute the excess to the
        uncapped sleeves proportional to their weight, per bar — preserving ``Σα = 1``. A
        no-op when no caps are configured (so VRP-off / uncapped books are unchanged) and
        monthly-held by construction (α only changes at month-ends, so the clip does too).
        The inner loop converges in ≤ N passes when redistribution pushes another sleeve to
        its own cap (cascading caps)."""
        if not self.weight_caps:
            return dict(alphas)
        names = list(alphas)
        cap = np.array([self.weight_caps.get(n, np.inf) for n in names], dtype=np.float64)
        A = np.vstack([np.asarray(alphas[n], dtype=np.float64) for n in names])   # (N, K)
        n_sleeves, K = A.shape
        out = A.copy()
        for k in range(K):
            col = out[:, k].copy()
            for _ in range(n_sleeves):
                over = col > cap + 1e-15
                if not over.any():
                    break
                excess = float((col[over] - cap[over]).sum())
                col[over] = cap[over]
                absorb = ~over
                wsum = float(col[absorb].sum())
                if wsum <= 1e-12 or excess <= 1e-15:
                    break
                col[absorb] += excess * (col[absorb] / wsum)
            out[:, k] = col
        return {n: out[i] for i, n in enumerate(names)}

    def _compound(self, returns: Mapping[str, np.ndarray], alphas: Mapping[str, np.ndarray],
                  ) -> tuple[np.ndarray, np.ndarray]:
        """Cross-account buy-and-hold-within-meta-period compounding (MS-ADR-3).

        ``E_s`` is each sub-account's equity; capital is re-split by ``α`` ONLY when ``α``
        changes (a month-end meta-rebalance), and each sub-account compounds at its own
        return between. Returns ``(R (K,), equity (K+1,))`` with ``equity[0] = C0``.
        """
        names = list(returns)
        K = len(next(iter(returns.values())))
        C0 = self.initial_capital
        E_s = {n: float(alphas[n][0]) * C0 for n in names}     # initial split (bar 0)
        prev_alpha = {n: float(alphas[n][0]) for n in names}
        equity = np.empty(K + 1, dtype=np.float64)
        equity[0] = C0
        R = np.empty(K, dtype=np.float64)
        for k in range(K):
            if k > 0 and any(abs(float(alphas[n][k]) - prev_alpha[n]) > _ALPHA_CHANGE_EPS for n in names):
                tot = float(sum(E_s.values()))
                E_s = {n: float(alphas[n][k]) * tot for n in names}    # meta-rebalance re-split
                prev_alpha = {n: float(alphas[n][k]) for n in names}
            for n in names:
                E_s[n] *= (1.0 + float(returns[n][k]))             # buy-and-hold compound
            tot = float(sum(E_s.values()))
            R[k] = tot / equity[k] - 1.0 if equity[k] > 1e-12 else 0.0
            equity[k + 1] = tot
        return R, equity

    def _combine_streams(self, streams: Mapping) -> LiveTrajectory:
        names = list(streams)
        alloc_name = next(n for n in names if streams[n].is_allocator)
        rs_names = [n for n in names if n != alloc_name]

        # Fast path / back-compat: only the allocator account ⇒ return its trajectory
        # VERBATIM (byte-identical to TwoSleeveExecutor, MS-ADR-6).
        if not rs_names:
            return streams[alloc_name].risk_extra["trajectory"]

        # 1. common calendar = intersection of all sleeve calendars (the overlap window).
        common = np.asarray(streams[names[0]].timestamps, dtype=np.int64)
        for n in names[1:]:
            common = np.intersect1d(common, np.asarray(streams[n].timestamps, dtype=np.int64))
        if common.size == 0:
            raise ValueError("no overlapping calendar across sleeves")

        # 2. align each sleeve's return stream to the common calendar.
        idx = {n: _align_index(streams[n].timestamps, common) for n in names}
        r = {n: np.asarray(streams[n].step_returns, dtype=np.float64)[idx[n]] for n in names}

        # 3. risk-parity α on the common return streams (convex, monthly-held, causal),
        #    then apply per-sleeve weight caps (the VRP cap: trailing vol understates the
        #    short-vol TAIL, so naive inverse-vol over-allocates to it — operator decision
        #    2026-06-22). Capping preserves Σα=1 (the excess redistributes to uncapped sleeves).
        alphas = self._apply_weight_caps(self._alphas(r, common))

        # 4. cross-account compounding → combined equity + return.
        R, equity = self._compound(r, alphas)

        # 5. combined weights = α_etf · ETF union weights (over the common window). VRP has
        #    no columns; its leverage is bounded by its own tail gates (surfaced separately).
        etf = streams[alloc_name]
        w_etf = np.asarray(etf.union_weights, dtype=np.float64)[idx[alloc_name]]
        a_alloc = np.asarray(alphas[alloc_name], dtype=np.float64)[:, None]
        weights = a_alloc * w_etf
        gross = np.abs(weights).sum(axis=1)
        net = weights.sum(axis=1)

        # 6. cost / turnover series from the ETF account (the cost_drift gate is ETF-level;
        #    VRP costs are inside r_vrp). Turnover scaled by the ETF capital share.
        etf_fees = np.asarray(etf.risk_extra["cumulative_fees"], dtype=np.float64)[idx[alloc_name]]
        etf_turn = np.asarray(etf.risk_extra["turnovers"], dtype=np.float64)[idx[alloc_name]]
        turnovers = a_alloc.ravel() * etf_turn

        # 7. attribution: each sleeve's combined contribution C_s = Σ α_s·r_s; the ETF
        #    contribution is distributed across its classes/sub-sleeves by their own shares.
        C = {n: float(np.sum(np.asarray(alphas[n]) * r[n])) for n in names}
        class_pnl = self._scaled_attribution(dict(etf.class_pnl), C[alloc_name])
        sleeve_pnl = self._scaled_attribution(dict(etf.risk_extra.get("sleeve_pnl") or {}), C[alloc_name])
        for n in rs_names:
            label = next(iter(streams[n].class_pnl), n)
            class_pnl[label] = C[n]
            sleeve_pnl[n] = C[n]

        spy = etf.spy_returns
        spy_common = np.asarray(spy, dtype=np.float64)[idx[alloc_name]] if spy is not None else None

        live = LiveTrajectory(
            weights=weights, equity_curve=equity, step_returns=R, turnovers=turnovers,
            cumulative_fees=etf_fees, gross_exposure=gross, net_exposure=net,
            timestamps=common, class_pnl=class_pnl, assets=list(self.union_assets),
            asset_class=dict(self.asset_class), initial_capital=self.initial_capital,
            spy_returns=spy_common, sleeve_pnl=sleeve_pnl, coverage_incomplete=False)
        # Per-sleeve aligned return streams + α paths — consumed by the step-5
        # diversification gate (corr(vrp, etf)) and the verdict digest (attached
        # dynamically, mirroring how TwoSleeveExecutor attaches sleeve_pnl).
        live.sleeve_returns = {n: r[n] for n in names}     # type: ignore[attr-defined]
        live.sleeve_alphas = {n: np.asarray(alphas[n], dtype=np.float64) for n in names}  # type: ignore[attr-defined]
        live.sleeve_risk = {n: dict(streams[n].risk_extra) for n in rs_names}  # type: ignore[attr-defined]
        live.allocator_sleeve = alloc_name                 # type: ignore[attr-defined]
        return live

    @staticmethod
    def _scaled_attribution(parts: dict[str, float], target_total: float) -> dict[str, float]:
        """Rescale a sub-account's attribution dict so it sums to the account's combined
        contribution ``target_total`` while preserving relative shares (so the combined
        drift gate sees comparable per-class / per-sleeve magnitudes). Identity if the
        parts sum to ~0 (longs/shorts cancel)."""
        total = float(sum(parts.values()))
        if abs(total) <= 1e-12:
            return dict(parts)
        scale = target_total / total
        return {k: v * scale for k, v in parts.items()}
