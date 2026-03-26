"""Softmax Arbitrator for multi-agent action weighting.

Provides two arbitrator implementations for the Synapse Crypto 1H strategy:

- **SoftmaxArbitrator** (primary): converts rolling performance scores
  (Sortino or Sharpe) into softmax-temperature-scaled weights with
  configurable floors and emergency reweight triggers.
- **InvVarArbitrator** (fallback): inverse-variance weighting over a
  rolling window.

Both arbitrators wrap SB3 agent ``.predict()`` calls and maintain per-agent
return buffers populated via ``record_return()``.

Example usage::

    arb = SoftmaxArbitrator.from_config(config_dict)
    for agent_name, agent in agents.items():
        arb.register_agent(agent_name)

    # Inside the step loop:
    arb.record_return("ppo", step_return_ppo)
    arb.record_return("sac", step_return_sac)
    combined_action, _ = arb.predict(obs, agents)
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Optional, Sequence

import numpy as np

_arb_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rolling statistics helpers (self-contained; full-sample versions live in
# finrl_pro.eval.statistics)
# ---------------------------------------------------------------------------

def rolling_mean(returns: Sequence[float]) -> float:
    """Arithmetic mean of *returns*. Returns 0.0 for empty input."""
    n = len(returns)
    if n == 0:
        return 0.0
    return float(np.mean(returns))


def rolling_std(returns: Sequence[float], ddof: int = 1) -> float:
    """Sample standard deviation with *ddof* correction."""
    n = len(returns)
    if n <= ddof:
        return 0.0
    return float(np.std(returns, ddof=ddof))


def rolling_downside_std(
    returns: Sequence[float],
    threshold: float = 0.0,
    ddof: int = 1,
) -> float:
    """True downside deviation: sqrt(mean(min(r - threshold, 0)^2)).

    Uses ALL observations (positive returns contribute zero), matching
    the standard Sortino formula.
    """
    if not returns:
        return 0.0
    downside_sq = np.array([min(0.0, r - threshold) ** 2 for r in returns])
    n = len(downside_sq)
    if n <= ddof:
        return 0.0
    return float(np.sqrt(np.sum(downside_sq) / (n - ddof)))


def rolling_sharpe(
    returns: Sequence[float],
    risk_free: float = 0.0,
    annualization: float = 8760.0,
) -> float:
    """Annualized Sharpe ratio over *returns*.

    Parameters
    ----------
    returns : sequence of float
        Per-bar simple returns.
    risk_free : float
        Annualized risk-free rate (default 0).
    annualization : float
        Bars per year (default 8760 for hourly).
    """
    n = len(returns)
    if n < 2:
        return 0.0
    mu = rolling_mean(returns) - risk_free / annualization
    s = rolling_std(returns)
    if s == 0.0:
        return 0.0
    return (mu / s) * (annualization ** 0.5)


def rolling_sortino(
    returns: Sequence[float],
    target: float = 0.0,
    annualization: float = 8760.0,
) -> float:
    """Annualized Sortino ratio over *returns*.

    Parameters
    ----------
    returns : sequence of float
        Per-bar simple returns.
    target : float
        Minimum acceptable return per bar (default 0).
    annualization : float
        Bars per year (default 8760 for hourly).
    """
    n = len(returns)
    if n < 2:
        return 0.0
    mu = rolling_mean(returns) - target
    ds = rolling_downside_std(returns, threshold=target)
    if ds == 0.0:
        return 0.0
    return (mu / ds) * (annualization ** 0.5)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class _BaseArbitrator:
    """Shared bookkeeping for arbitrators."""

    def __init__(
        self,
        lookback_bars: int = 504,
        annualization: float = 8760.0,
    ) -> None:
        self.lookback_bars = lookback_bars
        self.annualization = annualization
        # agent_name -> deque of per-bar returns
        self._buffers: dict[str, deque] = {}

    # -- agent management ---------------------------------------------------

    def register_agent(self, name: str) -> None:
        """Register an agent by name, initializing its return buffer."""
        if name not in self._buffers:
            self._buffers[name] = deque(maxlen=self.lookback_bars)

    @property
    def agent_names(self) -> list[str]:
        return list(self._buffers.keys())

    # -- return recording ---------------------------------------------------

    def record_return(self, agent_name: str, step_return: float) -> None:
        """Append a single-bar return for *agent_name*.

        The agent must have been registered first via ``register_agent``.

        Raises
        ------
        KeyError
            If *agent_name* has not been registered.
        """
        if agent_name not in self._buffers:
            raise KeyError(
                f"Unknown agent '{agent_name}'. "
                f"Call register_agent('{agent_name}') first.",
            )
        self._buffers[agent_name].append(float(step_return))

    def _get_returns(self, agent_name: str, n: Optional[int] = None) -> list[float]:
        """Return the last *n* recorded returns for *agent_name*.

        If *n* is None the full buffer (up to ``lookback_bars``) is used.
        """
        buf = self._buffers[agent_name]
        if n is None or n >= len(buf):
            return list(buf)
        return list(buf)[-n:]

    # -- step (no-op by default, overridden in SoftmaxArbitrator) -----------

    def step(self) -> None:
        """Advance internal state. Subclasses may override for reweight logic."""
        pass

    # -- prediction ---------------------------------------------------------

    def get_weights(self) -> dict[str, float]:
        raise NotImplementedError

    def predict(
        self,
        obs: np.ndarray,
        agents_dict: dict[str, Any],
        deterministic: bool = True,
    ) -> tuple:
        """Combine agent predictions using current weights.

        Parameters
        ----------
        obs : np.ndarray
            Current observation, compatible with each agent's ``.predict()``.
        agents_dict : dict[str, agent]
            Mapping of agent name to an SB3 agent (must expose ``.predict()``).
        deterministic : bool
            Forwarded to each agent's ``.predict()`` call.

        Returns
        -------
        combined_action : np.ndarray
            Weighted sum of individual agent actions.
        states : None
            Placeholder for API compatibility.
        """
        weights = self.get_weights()
        actions: list[np.ndarray] = []
        w_vec: list[float] = []

        for name, agent in agents_dict.items():
            if name not in weights:
                _arb_logger.warning(
                    "Agent '%s' not registered with arbitrator — skipping in predict()", name,
                )
                continue
            action, _ = agent.predict(obs, deterministic=deterministic)
            actions.append(np.asarray(action, dtype=np.float64))
            w_vec.append(weights[name])

        if not actions:
            raise ValueError(
                "No actions produced — ensure agents_dict keys match "
                "registered agent names.",
            )

        w_arr = np.array(w_vec, dtype=np.float64)
        # Normalize (should already sum to 1, but guard against rounding)
        w_arr /= w_arr.sum()

        stacked = np.stack(actions, axis=0)  # (n_agents, *action_shape)
        combined = np.tensordot(w_arr, stacked, axes=([0], [0]))
        return combined, None


# ---------------------------------------------------------------------------
# SoftmaxArbitrator
# ---------------------------------------------------------------------------

class SoftmaxArbitrator(_BaseArbitrator):
    """Performance-adaptive agent weighting via softmax temperature scaling.

    Tracks each agent's rolling Sortino (or Sharpe) ratio and converts the
    scores to weights through a softmax with configurable temperature.  A
    per-agent minimum weight floor prevents any agent from being silenced
    entirely.

    Parameters
    ----------
    lookback_bars : int
        Number of bars for the rolling performance window (default 504 =
        3 weeks hourly).
    temperature : float
        Softmax temperature.  Higher values produce more uniform weights;
        lower values concentrate weight on the best performer.
    min_weight : float
        Floor weight per agent.  The softmax output is mixed with a
        uniform distribution so that no agent falls below this value.
    reweight_every : int
        How often (in bars) to recompute weights under normal conditions.
    performance_metric : str
        ``"sortino"`` (default) or ``"sharpe"``.
    annualization : float
        Bars per year for ratio annualization (default 8760 for hourly).
    emergency_lookback : int
        Number of recent bars to evaluate emergency triggers (default 72).
    emergency_sortino_floor : float
        If any agent's rolling Sortino over the emergency lookback drops
        below this value, an immediate reweight is triggered.
    emergency_spread : float
        If the spread between the best and worst agent Sortino (over the
        emergency lookback) exceeds this value, an immediate reweight is
        triggered.
    """

    def __init__(
        self,
        lookback_bars: int = 504,
        temperature: float = 1.5,
        min_weight: float = 0.15,
        reweight_every: int = 24,
        performance_metric: str = "sortino",
        annualization: float = 8760.0,
        emergency_lookback: int = 72,
        emergency_sortino_floor: float = 0.0,
        emergency_spread: float = 2.0,
    ) -> None:
        super().__init__(lookback_bars=lookback_bars, annualization=annualization)
        self.temperature = temperature
        self.min_weight = min_weight
        self.reweight_every = reweight_every
        self.performance_metric = performance_metric.lower()
        self.emergency_lookback = emergency_lookback
        self.emergency_sortino_floor = emergency_sortino_floor
        self.emergency_spread = emergency_spread

        if self.performance_metric not in ("sortino", "sharpe"):
            raise ValueError(
                f"Unsupported performance_metric '{performance_metric}'. "
                "Use 'sortino' or 'sharpe'.",
            )

        # Internal state
        self._step_counter: int = 0
        self._cached_weights: dict[str, float] = {}
        self._circuit_breaker_fired: bool = False

    # -- factory ------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "SoftmaxArbitrator":
        """Instantiate from a flat or nested YAML config dict.

        Expected keys (all optional, sensible defaults apply)::

            arbitrator:
              lookback_bars: 504
              temperature: 1.5
              min_weight: 0.15
              reweight_every: 24
              performance_metric: sortino
              annualization: 8760
              emergency_lookback: 72
              emergency_sortino_floor: 0.0
              emergency_spread: 2.0

        If the config has a top-level ``"arbitrator"`` key the nested dict is
        used; otherwise the dict itself is treated as the parameter source.
        """
        d = cfg.get("arbitrator", cfg)
        return cls(
            lookback_bars=int(d.get("lookback_bars", 504)),
            temperature=float(d.get("temperature", 1.5)),
            min_weight=float(d.get("min_weight", 0.15)),
            reweight_every=int(d.get("reweight_every", 24)),
            performance_metric=str(d.get("performance_metric", "sortino")),
            annualization=float(d.get("annualization", 8760)),
            emergency_lookback=int(d.get("emergency_lookback", 72)),
            emergency_sortino_floor=float(d.get("emergency_sortino_floor", 0.0)),
            emergency_spread=float(d.get("emergency_spread", 2.0)),
        )

    # -- performance scores -------------------------------------------------

    def _score(self, returns: Sequence[float]) -> float:
        """Compute the chosen performance metric over *returns*."""
        if self.performance_metric == "sortino":
            return rolling_sortino(returns, annualization=self.annualization)
        return rolling_sharpe(returns, annualization=self.annualization)

    def _agent_scores(
        self, n: Optional[int] = None,
    ) -> dict[str, float]:
        """Compute performance scores for all agents."""
        scores: dict[str, float] = {}
        for name in self.agent_names:
            rets = self._get_returns(name, n=n)
            scores[name] = self._score(rets) if len(rets) >= 2 else 0.0
        return scores

    # -- softmax weighting --------------------------------------------------

    @staticmethod
    def _softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
        """Numerically stable softmax with *temperature* scaling."""
        s = scores / temperature
        s = s - s.max()  # shift for numerical stability
        exp_s = np.exp(s)
        return exp_s / exp_s.sum()

    def _apply_floor(self, raw_weights: dict[str, float]) -> dict[str, float]:
        """Enforce ``min_weight`` floor per agent.

        Strategy: clamp every weight to at least ``min_weight``, then
        redistribute the excess proportionally from above-floor agents so
        the weights still sum to 1.  If ``n_agents * min_weight >= 1``, fall
        back to uniform.
        """
        n = len(raw_weights)
        if n == 0:
            return {}
        total_floor = self.min_weight * n
        if total_floor >= 1.0:
            # Floor is too high — uniform is the only feasible solution
            w = 1.0 / n
            return {k: w for k in raw_weights}

        names = list(raw_weights.keys())
        weights = np.array([raw_weights[k] for k in names], dtype=np.float64)

        below = weights < self.min_weight
        if not below.any():
            return dict(zip(names, weights.tolist()))

        # Set below-floor agents to the floor
        deficit = np.sum(self.min_weight - weights[below])
        weights[below] = self.min_weight

        # Remove deficit proportionally from above-floor agents
        above = ~below
        above_sum = weights[above].sum()
        if above_sum > 0:
            weights[above] -= deficit * (weights[above] / above_sum)

        # Guard against negative values from redistribution
        weights = np.clip(weights, self.min_weight, None)
        weights /= weights.sum()

        return dict(zip(names, weights.tolist()))

    def _compute_weights(self) -> dict[str, float]:
        """Recompute weights from current return buffers."""
        names = self.agent_names
        n = len(names)
        if n == 0:
            return {}
        if n == 1:
            return {names[0]: 1.0}

        scores = self._agent_scores()
        score_arr = np.array([scores[k] for k in names], dtype=np.float64)
        raw = self._softmax(score_arr, self.temperature)
        raw_dict = dict(zip(names, raw.tolist()))
        return self._apply_floor(raw_dict)

    # -- emergency reweight -------------------------------------------------

    def _check_emergency(self) -> bool:
        """Return True if an emergency reweight should fire.

        Conditions (any one triggers):
        1. Any agent's rolling Sortino over ``emergency_lookback`` bars drops
           below ``emergency_sortino_floor``.
        2. The spread between the best and worst agent Sortino over
           ``emergency_lookback`` bars exceeds ``emergency_spread``.
        3. An external circuit breaker has been signalled.
        """
        if self._circuit_breaker_fired:
            self._circuit_breaker_fired = False  # consume the flag
            return True

        names = self.agent_names
        if len(names) < 2:
            return False

        # Use Sortino specifically for emergency checks, regardless of the
        # primary performance_metric setting.
        sortinos: list[float] = []
        for name in names:
            rets = self._get_returns(name, n=self.emergency_lookback)
            if len(rets) < 2:
                sortinos.append(0.0)
            else:
                sortinos.append(
                    rolling_sortino(rets, annualization=self.annualization),
                )

        # Condition 1: any agent below floor
        if any(s < self.emergency_sortino_floor for s in sortinos):
            return True

        # Condition 2: spread too wide
        spread = max(sortinos) - min(sortinos)
        if spread > self.emergency_spread:
            return True

        return False

    def signal_circuit_breaker(self) -> None:
        """External hook for paper/live mode circuit breaker events.

        When called, the next ``get_weights()`` invocation triggers an
        immediate reweight regardless of the bar cycle.
        """
        self._circuit_breaker_fired = True

    # -- public interface ---------------------------------------------------

    def step(self) -> None:
        """Advance the internal bar counter and refresh weights if due.

        Call this once per environment step *after* ``record_return``
        calls for all agents.  Weights are recomputed when:

        - ``_step_counter`` hits the ``reweight_every`` cadence, **or**
        - an emergency condition is detected.
        """
        self._step_counter += 1
        due = (self._step_counter % self.reweight_every == 0)
        # C1 fix: Force reweight on first step (when cache is empty) so that
        # seeded validation returns are reflected in initial weights instead
        # of falling back to uniform weights for the first reweight_every bars.
        if due or not self._cached_weights or self._check_emergency():
            self._cached_weights = self._compute_weights()

    def get_weights(self) -> dict[str, float]:
        """Return the current agent weight mapping.

        If weights have not been computed yet (no ``step()`` called),
        returns uniform weights.
        """
        if not self._cached_weights:
            names = self.agent_names
            n = len(names)
            if n == 0:
                return {}
            return {k: 1.0 / n for k in names}
        return dict(self._cached_weights)

    def get_scores(self) -> dict[str, float]:
        """Return current performance scores for all agents (diagnostic)."""
        return self._agent_scores()


# ---------------------------------------------------------------------------
# InvVarArbitrator (fallback)
# ---------------------------------------------------------------------------

class InvVarArbitrator(_BaseArbitrator):
    """Inverse-variance weighting over a rolling window.

    Each agent's weight is proportional to ``1 / variance`` of its recent
    returns.  Agents with lower variance receive higher weight, favouring
    consistency.  Agents with zero or insufficient data receive uniform
    weight.

    Parameters
    ----------
    lookback_bars : int
        Rolling window length (default 504).
    annualization : float
        Bars per year (default 8760, used only for bookkeeping parity with
        ``SoftmaxArbitrator``).
    min_variance : float
        Floor on per-agent variance to avoid division by zero or extreme
        concentration (default 1e-10).
    """

    def __init__(
        self,
        lookback_bars: int = 504,
        annualization: float = 8760.0,
        min_variance: float = 1e-10,
    ) -> None:
        super().__init__(lookback_bars=lookback_bars, annualization=annualization)
        self.min_variance = min_variance

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "InvVarArbitrator":
        """Instantiate from config dict (same nesting convention)."""
        d = cfg.get("arbitrator", cfg)
        return cls(
            lookback_bars=int(d.get("lookback_bars", 504)),
            annualization=float(d.get("annualization", 8760)),
            min_variance=float(d.get("min_variance", 1e-10)),
        )

    def get_weights(self) -> dict[str, float]:
        """Compute inverse-variance weights from current buffers.

        Returns uniform weights when fewer than 2 returns are available
        for any agent.
        """
        names = self.agent_names
        n = len(names)
        if n == 0:
            return {}

        uniform = {k: 1.0 / n for k in names}

        variances: list[float] = []
        for name in names:
            rets = self._get_returns(name)
            if len(rets) < 2:
                return uniform
            var = float(np.var(rets, ddof=1))
            variances.append(max(var, self.min_variance))

        inv_vars = np.array([1.0 / v for v in variances], dtype=np.float64)
        weights = inv_vars / inv_vars.sum()
        return dict(zip(names, weights.tolist()))
