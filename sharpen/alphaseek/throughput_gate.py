"""AlphaSeek v3 throughput gate (S500 ADR-005).

Caps median fills/episode at 150 (10× fewer than v1/v2's 500–1,555) so the
HPO study never wastes GPU on agents that have rediscovered v2's
fee-vulnerable scalping behaviour. Floor of 10 catches reward-shaping
degeneracies where the agent learns to "never trade" to avoid the per-trade
penalty.

Two surfaces:

    - ``ThroughputTracker`` — pure stats accumulator, OK to instantiate from
      tests or eval scripts where Optuna is not in scope.
    - ``ThroughputPruningCallback`` — the Optuna-aware variant; wraps
      ``ThroughputTracker`` and calls ``trial.should_prune()`` after warmup.

The simulator emits ``info["fill_count"]`` per step (cumulative per-sim fills
since reset). The gate snapshots fill_count at episode end (truncated/done)
and computes the median across sims, which is the metric of record.

HPO config flow (M5):
    throughput_gate:
      enabled: true
      n_warmup_steps: 20000
      max_fills_per_episode: 150
      min_fills_per_episode: 10
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from typing import Any

import torch as th

logger = logging.getLogger(__name__)


@dataclass
class ThroughputTracker:
    """Accumulates per-episode fill counts across simulations.

    Calling code calls ``record_episode_end(fill_counter)`` once per episode
    boundary, where ``fill_counter`` is the per-sim cumulative count from
    ``LOBTradeSimulator`` immediately before reset.
    """

    max_fills_per_episode: int = 150
    min_fills_per_episode: int = 10
    history: list[float] = field(default_factory=list)

    def record_episode_end(self, fill_counter: th.Tensor) -> float:
        """Record the median fills-per-episode across the sim batch.

        Returns the recorded median for logging convenience.
        """
        if fill_counter.numel() == 0:
            return 0.0
        median_fills = float(fill_counter.float().median().item())
        self.history.append(median_fills)
        return median_fills

    @property
    def median_episode_fills(self) -> float:
        """Median fills/episode over all recorded episodes (NaN-safe)."""
        if not self.history:
            return 0.0
        return float(statistics.median(self.history))

    def verdict(self) -> tuple[bool, str]:
        """Return ``(should_prune, reason)`` against cap + floor.

        Empty history returns ``(False, "insufficient_data")`` so callers can
        defer the prune check until at least one episode has rolled.
        """
        if not self.history:
            return False, "insufficient_data"
        m = self.median_episode_fills
        if m > self.max_fills_per_episode:
            return True, (
                f"throughput_cap_breach: median {m:.1f} > "
                f"max {self.max_fills_per_episode}"
            )
        if m < self.min_fills_per_episode:
            return True, (
                f"throughput_floor_breach: median {m:.1f} < "
                f"min {self.min_fills_per_episode} (degenerate never-trade)"
            )
        return False, f"ok: median {m:.1f} fills/episode"

    def reset(self) -> None:
        self.history.clear()


class ThroughputPruningCallback:
    """Optuna-aware wrapper around ``ThroughputTracker``.

    Use inside the Stage-1 HPO loop:

        gate = ThroughputPruningCallback(
            trial=trial,
            n_warmup_steps=20000,
            max_fills_per_episode=150,
            min_fills_per_episode=10,
        )
        ...
        for step, info in training_loop():
            if done.any():
                gate.on_episode_end(step, info["fill_count"])

    On a verdict-trip after warmup the callback raises
    ``optuna.exceptions.TrialPruned`` and writes the reason to the trial's
    user attrs so post-mortems can sort failures by cause.
    """

    def __init__(
        self,
        trial: Any,
        n_warmup_steps: int = 20_000,
        max_fills_per_episode: int = 150,
        min_fills_per_episode: int = 10,
    ):
        self.trial = trial
        self.n_warmup_steps = int(n_warmup_steps)
        self.tracker = ThroughputTracker(
            max_fills_per_episode=int(max_fills_per_episode),
            min_fills_per_episode=int(min_fills_per_episode),
        )

    def on_episode_end(self, global_step: int, fill_counter: th.Tensor) -> None:
        median = self.tracker.record_episode_end(fill_counter)
        if global_step < self.n_warmup_steps:
            return
        should_prune, reason = self.tracker.verdict()
        # Log the metric for plotting regardless of prune outcome.
        try:
            self.trial.report(median, step=int(global_step))
        except Exception as exc:  # pragma: no cover — defensive only
            logger.debug("trial.report failed at step %d: %s", global_step, exc)
        if should_prune:
            try:
                self.trial.set_user_attr("throughput_gate_reason", reason)
            except Exception:
                pass
            try:
                from optuna.exceptions import TrialPruned
            except ImportError as exc:  # pragma: no cover — Optuna is required
                raise RuntimeError(
                    "ThroughputPruningCallback requires optuna; install via "
                    "pip install optuna",
                ) from exc
            logger.info("Throughput gate pruning trial: %s", reason)
            raise TrialPruned(reason)

    @property
    def median_episode_fills(self) -> float:
        return self.tracker.median_episode_fills
