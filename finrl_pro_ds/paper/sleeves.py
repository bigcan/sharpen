"""Portfolio sleeve interface for the N-sleeve paper executor (fund-of-funds).

The generalization of the hardcoded 2-sleeve allocator into an N-sleeve FoF
(S553-cont-68; spec ``.agent/artifacts/multi_sleeve_paper_executor_architecture.md``).
A *sleeve* is a top-level capital-allocation unit producing a daily NET-return stream
on the portfolio's COMMON business-day calendar, plus a forward-recompute for the
load-bearing parity check. Two kinds:

  - :class:`AllocatorBookSleeve` — weights over the ETF union (an IB account): the
    existing momentum+rates :class:`~finrl_pro_ds.paper.two_sleeve.TwoSleeveExecutor`,
    adapted. WEIGHT-combined internally (shared bonds net); exposes ``union_weights``
    for the weight-parity / gross gates.
  - :class:`ReturnStreamSleeve` — a self-contained return stream from its own sim (the
    BTC options-VRP sleeve, a Deribit account; concrete :class:`VRPSleeve` in step 3).
    No union weights; its sim is its own parity oracle.

:class:`~finrl_pro_ds.paper.portfolio_executor.PortfolioExecutor` combines these at the
RETURN level (cross-account; MS-ADR-1) via ``risk_parity_alphas`` — because separate
brokerage accounts (Deribit vs IB) cannot net positions or share margin. Within the ETF
account the combine stays at the WEIGHT level (unchanged), so a VRP-disabled portfolio
is byte-identical to ``TwoSleeveExecutor`` (MS-ADR-6).

Invariants: LEAK-2 (causal forward assembly — the ``forward_recompute`` tripwire);
epoch-SECONDS timestamps on every stream (the VRP panel is epoch-ms and MUST be
converted by its sleeve — MS-ADR-3); no raw ``print`` (logging only).
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from typing import Callable, Mapping

import numpy as np

from finrl_pro_ds.crypto.options_vrp_sim import SimConfig, returns_from_pnl, simulate_asset
from finrl_pro_ds.paper.paper_state import LiveTrajectory
from finrl_pro_ds.paper.two_sleeve import TwoSleeveExecutor

_VRP_ANN = 365.0  # crypto 24/7 — the native VRP return calendar (tail-stat annualization)

logger = logging.getLogger(__name__)


@dataclass
class SleeveStream:
    """A sleeve's daily NET-return stream on the portfolio's common business-day calendar.

    ``step_returns[k]`` is the return of $1 *fully allocated* to this sleeve over the
    decision bar ``timestamps[k]`` (the portfolio executor scales each sleeve by its
    risk-parity α). ``union_weights`` is populated only for an :class:`AllocatorBookSleeve`
    (the signed env-booked weights over the ETF union, used for the weight-parity / gross
    gates); it is ``None`` for a pure return-stream sleeve. ``risk_extra`` carries
    sleeve-specific surfaces (e.g. the VRP short-vol tail: CVaR/vega/sortino).
    """

    name: str
    timestamps: np.ndarray            # (K,) int64 epoch-SECONDS, business-day decision stamps
    step_returns: np.ndarray          # (K,) daily NET return of $1 fully allocated to this sleeve
    union_weights: np.ndarray | None = None   # (K, U) signed union weights; None for return-stream
    class_pnl: dict[str, float] = field(default_factory=dict)
    risk_extra: dict = field(default_factory=dict)
    spy_returns: np.ndarray | None = None      # (K,) SPY daily returns aligned to timestamps, or None

    def __post_init__(self) -> None:
        self.timestamps = np.asarray(self.timestamps, dtype=np.int64).ravel()
        self.step_returns = np.asarray(self.step_returns, dtype=np.float64).ravel()
        if self.timestamps.shape != self.step_returns.shape:
            raise ValueError(
                f"sleeve {self.name!r}: timestamps {self.timestamps.shape} != "
                f"step_returns {self.step_returns.shape}")
        if not np.isfinite(self.step_returns).all():
            raise ValueError(f"sleeve {self.name!r}: non-finite step_returns (sanitize before constructing)")
        if self.union_weights is not None:
            self.union_weights = np.asarray(self.union_weights, dtype=np.float64)
            if self.union_weights.ndim != 2 or self.union_weights.shape[0] != self.step_returns.shape[0]:
                raise ValueError(
                    f"sleeve {self.name!r}: union_weights {self.union_weights.shape} "
                    f"incompatible with K={self.step_returns.shape[0]}")
        if self.spy_returns is not None:
            self.spy_returns = np.asarray(self.spy_returns, dtype=np.float64).ravel()
            if self.spy_returns.shape != self.step_returns.shape:
                raise ValueError(
                    f"sleeve {self.name!r}: spy_returns {self.spy_returns.shape} != "
                    f"step_returns {self.step_returns.shape}")

    @property
    def n_steps(self) -> int:
        return int(self.step_returns.shape[0])

    @property
    def is_allocator(self) -> bool:
        """True iff this stream carries union weights (an AllocatorBookSleeve)."""
        return self.union_weights is not None


@dataclass
class SleeveContext:
    """Inputs a sleeve needs to produce its stream, plus the portfolio's common calendar.

    ``data`` is sleeve-specific (the two-sleeve ``bundle`` for an AllocatorBookSleeve; the
    options panel for a VRPSleeve). ``union_timestamps`` is the portfolio's common
    business-day decision calendar (epoch-SECONDS) that a return-stream sleeve resamples
    its native (e.g. 24/7 crypto) returns onto (MS-ADR-3); ``None`` lets a sleeve emit on
    its own calendar (the executor then intersects).
    """

    data: Mapping
    union_timestamps: np.ndarray | None = None


# A broken-assembler injection: maps a sub-key (e.g. an allocator sleeve name) → a
# deliberately-wrong forward reader, used ONLY by tests to prove the look-ahead escape
# exists (drives weight_l1_drift / daily_return_te_bps above the gate). None = the safe path.
BrokenAssembler = Mapping[str, Callable] | Callable | None


class Sleeve(ABC):
    """A top-level portfolio sleeve: a daily return stream + a load-bearing forward path."""

    name: str

    @abstractmethod
    def sim_oracle(self, ctx: SleeveContext) -> SleeveStream:
        """Batch ground-truth stream over the full common window. Byte-identical to the
        sleeve's own validated oracle (AllocatorBookSleeve → ``TwoSleeveExecutor.run``;
        VRPSleeve → the falsification sim on the cached panel)."""
        raise NotImplementedError

    @abstractmethod
    def forward_recompute(self, ctx: SleeveContext, *, broken_assembler: BrokenAssembler = None) -> SleeveStream:
        """Re-derive the stream from an INCREMENTALLY assembled view on a growing window
        (the live scheduler's data path). On the SAFE path reproduces :meth:`sim_oracle`
        (per-step drift ≈ 0); ``broken_assembler`` (test-only) injects a look-ahead /
        calendar bug to prove the escape exists. Invariant: LEAK-2 (only data ``<= t``)."""
        raise NotImplementedError


class ReturnStreamSleeve(Sleeve, ABC):
    """Marker base for a sleeve that IS a self-contained return stream from its own sim
    (a separate account; no weights-over-union ⇒ ``union_weights`` always ``None``). The
    sim is the per-sleeve parity oracle. Concrete subclass: :class:`VRPSleeve` (step 3),
    which resamples its native 24/7 crypto returns onto the portfolio's business-day
    calendar (MS-ADR-3) before emitting the stream."""


class AllocatorBookSleeve(Sleeve):
    """Adapter wrapping :class:`~finrl_pro_ds.paper.two_sleeve.TwoSleeveExecutor` as one
    top-level sleeve (the ETF/IB account). The internal weight-combine of its allocator
    sub-sleeves (momentum, rates, …) is UNCHANGED — this only re-expresses the executor's
    combined :class:`LiveTrajectory` as a :class:`SleeveStream` for the portfolio combine.
    """

    def __init__(self, config: Mapping, *, name: str = "etf_book") -> None:
        self.name = name
        self.config = config
        self.executor = TwoSleeveExecutor(config)

    def sim_oracle(self, ctx: SleeveContext) -> SleeveStream:
        live, _ = self.executor.run(ctx.data)
        return self._to_stream(live)

    def forward_recompute(self, ctx: SleeveContext, *, broken_assembler: BrokenAssembler = None) -> SleeveStream:
        # broken_assembler maps an allocator sub-sleeve name → a broken conviction reader
        # (the existing TwoSleeveExecutor look-ahead escape); None = the safe forward path.
        conviction_fns = broken_assembler if isinstance(broken_assembler, Mapping) else None
        live, _ = self.executor.run_independent_recompute(ctx.data, conviction_fns=conviction_fns)
        return self._to_stream(live)

    def _to_stream(self, live: LiveTrajectory) -> SleeveStream:
        return SleeveStream(
            name=self.name,
            timestamps=live.timestamps,
            step_returns=live.step_returns,
            union_weights=live.weights,                 # full-scale env-booked union weights
            class_pnl=dict(live.class_pnl),
            # The full ETF-account trajectory is carried so the PortfolioExecutor can return
            # it VERBATIM when VRP is disabled (byte-identical to TwoSleeveExecutor, MS-ADR-6)
            # and source the cost/turnover series when combining.
            risk_extra={"sleeve_pnl": live.sleeve_pnl, "trajectory": live,
                        "cumulative_fees": live.cumulative_fees, "turnovers": live.turnovers},
            spy_returns=live.spy_returns,
        )


# --------------------------------------------------------------------------- #
# Calendar resample (MS-ADR-3) — compound a native (24/7) return stream onto the
# portfolio's business-day calendar; weekend/holiday P&L marks at the next business day.
# --------------------------------------------------------------------------- #
def resample_returns_to_calendar(
    native_ts_s: np.ndarray, native_returns: np.ndarray, target_ts_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compound a sleeve's NATIVE daily returns onto a target (business-day) calendar.

    Target bar ``k`` accumulates every native bar in the half-open interval
    ``(target[k-1], target[k]]`` (the first emitted bar accumulates from the start of
    native coverage). Both timestamp arrays are epoch-SECONDS, strictly increasing. Only
    target bars within native coverage (``native_ts[0] <= target[k] <= native_ts[-1]``)
    are emitted — the portfolio executor then intersects sleeve calendars on the overlap
    (MS-ADR-3 / Open Item 5), so the pre-coverage period is dropped, never zero-filled
    (a zero-fill would deflate the sleeve's risk-parity vol).

    Compounding telescopes: ``Π_k (1+R_k) == Π_n (1+native[n])`` over the covered native
    span (a unit-test invariant). NaN native returns are treated as 0 (degenerate bars).

    Returns ``(covered_target_ts, resampled_returns)`` both of length ``K_covered``.
    """
    native_ts = np.asarray(native_ts_s, dtype=np.int64).ravel()
    nret = np.nan_to_num(np.asarray(native_returns, dtype=np.float64).ravel(), nan=0.0)
    target = np.asarray(target_ts_s, dtype=np.int64).ravel()
    if native_ts.size == 0 or target.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    lo_t, hi_t = int(native_ts[0]), int(native_ts[-1])
    covered = target[(target >= lo_t) & (target <= hi_t)]
    if covered.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    # Pcum[j] = Π(1+native[:j]); product over native indices [a, b) = Pcum[b]/Pcum[a].
    pcum = np.concatenate([[1.0], np.cumprod(1.0 + nret)])
    left = np.empty(covered.size, dtype=np.int64)
    left[0] = lo_t - 1                      # first bucket includes from the start of coverage
    left[1:] = covered[:-1]                 # subsequent buckets start at the prior business mark
    idx_lo = np.searchsorted(native_ts, left, side="right")
    idx_hi = np.searchsorted(native_ts, covered, side="right")
    ret = pcum[idx_hi] / pcum[idx_lo] - 1.0
    return covered.astype(np.int64), ret.astype(np.float64)


def _vrp_tail_stats(daily_ret: np.ndarray, eq_curve: np.ndarray,
                    vega_pct: np.ndarray | None) -> dict:
    """Compact short-vol tail surface on the NATIVE (24/7) VRP returns — the metrics the
    combined verdict surfaces against ``options_vol_harvest.gates.yaml`` ``tail`` kills
    (CVaR95 floor, net-vega cap, worst-window DD). Mirrors the falsification ``_risk_block``
    essentials (ANN=365)."""
    d = np.asarray(daily_ret, dtype=np.float64)
    d = d[np.isfinite(d)]
    out: dict = {"n_obs": int(d.size)}
    if d.size >= 3:
        mu, sd = float(d.mean()), float(d.std(ddof=1))
        out["skew"] = float(((d - mu) ** 3).mean() / sd ** 3) if sd > 0 else 0.0
        downside = d[d < 0]
        out["sortino"] = (float(mu / downside.std(ddof=1) * np.sqrt(_VRP_ANN))
                          if downside.size > 2 and downside.std(ddof=1) > 0 else float("nan"))
        v95 = float(np.quantile(d, 0.05))
        out["cvar95_pct_daily"] = float(d[d <= v95].mean()) * 100.0
        v99 = float(np.quantile(d, 0.01))
        out["cvar99_pct_daily"] = float(d[d <= v99].mean()) * 100.0
        out["worst_day_pct"] = float(d.min()) * 100.0
    eq = np.asarray(eq_curve, dtype=np.float64)
    if eq.size:
        peak = np.maximum.accumulate(eq)
        out["max_dd_pct"] = float((1.0 - eq / np.where(peak <= 0, 1.0, peak)).max()) * 100.0
    if vega_pct is not None and len(vega_pct):
        out["max_net_vega_per_100k"] = float(np.nanmax(vega_pct) * 1e5)
    return out


class VRPSleeve(ReturnStreamSleeve):
    """The BTC options variance-risk-premium sleeve (a Deribit return-stream account).

    Runs the validated short-straddle sim (``options_vrp_sim.simulate_asset``) on the BTC
    panel, converts the native 24/7 daily returns to epoch-SECONDS and resamples them onto
    the portfolio's business-day calendar (MS-ADR-3), and emits a :class:`SleeveStream`
    (``union_weights=None``). Its sim is its own parity oracle; ``forward_recompute``
    re-assembles the panel (optionally via a broken assembler) to surface a look-ahead.

    Instrument knob (MS-ADR-10): only ``naked_straddle`` is validated/gated. ``iron_fly`` /
    ``strangle`` are rejected here until each clears its OWN phase-1 falsification gate
    (new payoff/greeks = a different strategy); selection among gated variants is by
    tail-adjusted return (Sortino/CVaR), surfaced in ``risk_extra``.
    """

    def __init__(self, config: Mapping, *, name: str = "vrp") -> None:
        self.name = name
        vrp_cfg = dict(config.get("sleeves", {}).get("vrp", {}))
        self.asset = str(vrp_cfg.get("asset", "BTC"))
        sim_block = dict(vrp_cfg.get("sim", {}))
        instrument = str(sim_block.pop("instrument", "naked_straddle"))
        if instrument != "naked_straddle":
            raise NotImplementedError(
                f"VRP instrument {instrument!r} is not gated — only 'naked_straddle' is "
                f"validated (MS-ADR-10). A new payoff must clear its own phase-1 "
                f"falsification gate (+ tail kills) before it can be selected live.")
        self.instrument = instrument
        valid = {f.name for f in fields(SimConfig)}
        self.sim_cfg = SimConfig(**{k: v for k, v in sim_block.items() if k in valid})

    # ------------------------------------------------------------------ #
    def sim_oracle(self, ctx: SleeveContext) -> SleeveStream:
        return self._build_stream(ctx, assembler=None)

    def forward_recompute(self, ctx: SleeveContext, *, broken_assembler: BrokenAssembler = None) -> SleeveStream:
        # A return-stream sleeve's broken_assembler is a callable(spot, iv, funding)->
        # (spot, iv, funding) that injects a look-ahead/calendar bug into the panel
        # assembly; None = the safe path (reproduces the oracle — the sim is causal/online).
        assembler = broken_assembler if callable(broken_assembler) else None
        return self._build_stream(ctx, assembler=assembler)

    # ------------------------------------------------------------------ #
    def _panel(self, ctx: SleeveContext):
        """Injected panel (tests / executor-prefetched) or a cache load (production)."""
        if ctx.data is not None and "panel" in ctx.data:
            return ctx.data["panel"]
        # Lazy import: keep paper/sleeves light + avoid pulling crypto-loader deps unless
        # a real disk load is needed (the executor/tests inject the panel).
        from finrl_pro_ds.crypto.data import deribit_options_loader as dol
        from finrl_pro_ds.crypto.data import options_array_builder as oab
        return oab.build_panels(dol.load({"universe": {"assets": [self.asset]}}))

    def _build_stream(self, ctx: SleeveContext, *, assembler: Callable | None) -> SleeveStream:
        panel = self._panel(ctx)
        i = list(panel.assets).index(self.asset)
        spot = np.asarray(panel.spot_ary)[:, i]
        iv = np.asarray(panel.iv_ary)[:, i]
        funding = np.asarray(panel.funding_ary)[:, i]
        if assembler is not None:
            spot, iv, funding = assembler(spot, iv, funding)
        res = simulate_asset(spot, iv, funding, self.sim_cfg)
        native_ret = np.nan_to_num(
            returns_from_pnl(res["daily_pnl"], res["eq_curve"]), nan=0.0)
        native_ts_s = np.asarray(panel.timestamps, dtype=np.int64) // 1000   # ms -> s (MS-ADR-3)

        target = ctx.union_timestamps
        if target is None:
            ts, ret = native_ts_s, native_ret           # standalone: native calendar
        else:
            ts, ret = resample_returns_to_calendar(native_ts_s, native_ret,
                                                    np.asarray(target, dtype=np.int64))
        risk_extra = _vrp_tail_stats(native_ret, res["eq_curve"], res.get("vega_pct"))
        risk_extra["instrument"] = self.instrument
        # crypto_vol = the sleeve's own cumulative gross-return attribution (the executor
        # α-weights it when combining class_pnl across sleeves for the drift gate).
        cum = float(np.prod(1.0 + ret) - 1.0) if ret.size else 0.0
        return SleeveStream(
            name=self.name, timestamps=ts, step_returns=ret,
            union_weights=None, class_pnl={"crypto_vol": cum}, risk_extra=risk_extra)
