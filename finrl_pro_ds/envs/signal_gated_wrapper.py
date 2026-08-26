"""
Signal-Gated Environment Wrapper
=================================

Wraps ContinuousSwingEnv (V7) to filter out low-signal bars. The agent only
receives observations and chooses actions on "gated" bars where signal strength
exceeds a threshold. During skipped bars, the position is held and rewards
accumulate.

Gate signals are read from the handler's pre-computed scale features (causal,
LEAK-1 compliant). Gate opens if ANY signal exceeds its configured threshold.

Design rationale:
  - Wrapper (not data-level filtering) preserves multi-scale windowed observations
  - Hold action = current_position guarantees zero trades during skipped bars
  - DSR accumulates naturally across skipped bars (inner env computes per-bar)
  - Compatible with SyncVectorEnv, SACTrainer, fee curriculum, HPO, backtest
"""
import logging
from typing import Any

import gymnasium as gym
import numpy as np

logger = logging.getLogger(__name__)


class SignalGatedWrapper(gym.Wrapper):
    """Wraps ContinuousSwingEnv to gate agent actions to high-signal bars.

    On each wrapper step:
      1. Apply the agent's action (one inner env step)
      2. While the LAST CLOSED bar's gate is shut (and max_hold_bars not
         reached):
         - Hold current position (inner env step with hold action)
         - Accumulate reward
      3. Return final obs, accumulated reward, done flags, augmented info

    Causality contract (GATE-CAUSAL-01):
      The gate is evaluated on ``handler._ptr - 1`` -- the bar that has already
      CLOSED -- to decide whether to hold through the next one. It must never be
      evaluated on ``_ptr`` itself: that bar has not closed, and the agent's next
      action is applied to it, so gating on it selects decision points with
      hindsight (LEAK-2). This also matches the live engine, which can only ever
      see the last closed bar (``_check_signal_gate`` reads ``features[-1]``);
      sim and live gate on the same bar. Guarded by
      ``tests/test_signal_gated_wrapper.py::TestGateCausal01``.

    Info aggregation contract (GATE-TRADED-01):
      One wrapper step spans 1 + ``gate_skipped_bars`` inner bars, so the info
      dict mixes two semantics. State and cumulative keys (portfolio_value,
      position, trade_count, cumulative_fees, drawdown_pct, regime_code) carry
      LAST-BAR values and are correct as-is. Per-bar EVENT keys must be
      aggregated or they are lost: ``traded`` is the OR across the aggregated
      bars, and ``gate_inner_trades`` counts the inner BARS on which at least
      one trade occurred.

      ``gate_inner_trades`` is a bar count, NOT a trade count -- one inner bar
      can log up to three ``trade_count`` increments (deadband move, then a
      stop-loss flat, then a max-holding flat; see ContinuousSwingEnv.step).
      The authoritative trade total is the env's cumulative ``trade_count``,
      diffed across outer steps. Summing ``traded`` undercounts further still,
      because a bool cannot express two traded bars inside one outer step.

    Gate signals from handler._scale_features[base_scale]:
      idx 1: atr_norm    — ATR(14)/EMA(ATR,50) - 1.0, >0 = above-average vol
      idx 2: parkinson   — sqrt(log(H/L)^2 / (4*ln2)), intra-bar realized vol
      idx 7: volume_z    — SymLog -> EMA-Z -> tanh normalized volume
    """

    def __init__(self, env: gym.Env, gate_config: dict[str, Any]):
        super().__init__(env)
        self.gate_config = gate_config

        # Gate thresholds (gate opens if ANY signal exceeds its threshold)
        self.atr_threshold = float(gate_config.get("atr_threshold", 0.3))
        self.parkinson_threshold = float(gate_config.get("parkinson_threshold", 0.02))
        self.volume_threshold = float(gate_config.get("volume_threshold", 0.5))

        # abs_log_return threshold (for gate_mode="return")
        self.return_threshold = float(gate_config.get("return_threshold", 0.002))

        # Safety cap: force gate open after this many consecutive hold bars
        self.max_hold_bars = int(gate_config.get("max_hold_bars", 20))

        # Always open gate on first bar after reset
        self.gate_always_on_first = bool(gate_config.get("gate_always_on_first", True))

        # Normalize accumulated reward by dividing by (1 + skipped_bars)
        self.normalize_reward = bool(gate_config.get("normalize_accumulated_reward", False))

        # Gate mode
        self.gate_mode = gate_config.get("gate_mode", "composite")

        # Track statistics
        self._total_skipped = 0
        self._total_gated_steps = 0
        self._is_first_step = True

        # Cache handler reference
        self._handler = getattr(env, 'handler', None)
        self._scale_features = None
        self._base_scale = None

        if self._handler is not None:
            self._base_scale = getattr(self._handler, '_base_scale', None)
            sf = getattr(self._handler, '_scale_features', None)
            if sf is not None and self._base_scale is not None:
                self._scale_features = sf.get(self._base_scale)

        if self._scale_features is None:
            logger.warning(
                "SignalGatedWrapper: Could not access handler scale features. "
                "Gate will always be open (passthrough mode).",
            )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._is_first_step = True
        self._total_skipped = 0
        self._total_gated_steps = 0

        # Re-cache features after reset (handler may have changed)
        if self._handler is not None:
            sf = getattr(self._handler, '_scale_features', None)
            if sf is not None and self._base_scale is not None:
                self._scale_features = sf.get(self._base_scale)

        return obs, info

    def step(self, action):
        # 1. Apply agent's action to inner env (one bar)
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._total_gated_steps += 1

        # GATE-TRADED-01 (S553-cont-170). `traded` is a PER-BAR EVENT, but one
        # wrapper step aggregates 1 + `skipped` inner bars and returns the LAST
        # inner info dict. Left un-accumulated, a trade on the decision bar is
        # erased by any hold bar that follows it, so the flag reads False on
        # roughly half of all real trades (sg1-btc decay01-X1 fold_00 solo_123:
        # 145 position changes, 69 flags). Downstream that silently understates
        # turnover and fees, and mis-selects "untraded" bars.
        #
        # Only per-bar EVENT keys need accumulating. State and cumulative keys
        # (portfolio_value, position, trade_count, cumulative_fees,
        # drawdown_pct, regime_code) are correct with last-bar semantics and are
        # deliberately left alone.
        #
        # A hold bar can legitimately trade: the inner env force-flattens on
        # stop_loss_bps / max_holding_bars regardless of the action it was
        # handed. So this accumulates observed flags rather than assuming holds
        # never trade.
        traded_any = bool(info.get("traded", False))
        inner_trades = int(traded_any)

        if terminated or truncated:
            info["traded"] = traded_any
            info["gate_inner_trades"] = inner_trades
            info["gate_skipped_bars"] = 0
            info["gate_total_skipped"] = self._total_skipped
            info["gate_total_steps"] = self._total_gated_steps
            return obs, reward, terminated, truncated, info

        # 2. Accumulate through hold bars while gate is closed
        total_reward = reward
        skipped = 0

        while skipped < self.max_hold_bars:
            # GATE-CAUSAL-01 (S553-cont-170). Decide on the LAST CLOSED bar.
            #
            # `handler._ptr` is the index of the bar NOT YET consumed. Gating on
            # it was a LEAK-2 look-ahead: the gate opened on bar k+1's own
            # atr_norm / parkinson / volume_z, and the agent's next action was
            # then applied to that very bar. Those values are not knowable until
            # bar k+1 closes -- they are built from bar k+1's own high/low/close
            # (no shift), and `_ema_zscore_tanh`'s shift(1) makes only the EMA
            # STATISTICS causal, never the value itself.
            #
            # `_ptr - 1` is the bar just consumed, i.e. the last CLOSED bar --
            # which is exactly what the live engine can see
            # (`_check_signal_gate` reads `features[-1]`). Sim and live now gate
            # on the same bar; before this they were off by one.
            next_ptr = getattr(self._handler, '_ptr', None)
            gate_ptr = None if next_ptr is None else next_ptr - 1
            if gate_ptr is None or self._gate_open(gate_ptr):
                break

            # Gate closed: hold current position
            # Multi-asset: use positions array; single-asset: scalar
            positions = getattr(self.env, 'positions', None)
            if positions is not None and hasattr(positions, '__len__') and len(positions) > 1:
                hold_action = positions.astype(np.float32)
            else:
                hold_action = np.array([getattr(self.env, 'current_position', 0.0)], dtype=np.float32)
            obs, r, terminated, truncated, info = self.env.step(hold_action)
            total_reward += r
            skipped += 1

            # GATE-TRADED-01: read the flag off THIS inner info before the next
            # iteration overwrites `info`.
            if info.get("traded", False):
                traded_any = True
                inner_trades += 1

            if terminated or truncated:
                break

        self._total_skipped += skipped
        self._is_first_step = False

        # Optionally normalize reward by hold duration
        if self.normalize_reward and skipped > 0:
            total_reward = total_reward / (1 + skipped)

        # Augment info
        # GATE-TRADED-01: `traded` is the OR across the aggregated bars, so it
        # keeps its documented bool contract. `gate_inner_trades` adds the count
        # of traded BARS, which a bool cannot express once an outer step holds
        # two of them. It is deliberately NOT a trade count: one bar can log
        # three `trade_count` increments (deadband move / stop-loss flat /
        # max-holding flat), so the env's cumulative `trade_count` stays the
        # authoritative total.
        info["traded"] = traded_any
        info["gate_inner_trades"] = inner_trades
        info["gate_skipped_bars"] = skipped
        info["gate_total_skipped"] = self._total_skipped
        info["gate_total_steps"] = self._total_gated_steps

        return obs, total_reward, terminated, truncated, info

    def _gate_open(self, ptr: int) -> bool:
        """Check whether the bar at ``ptr`` passes the signal gate.

        ``ptr`` MUST be the index of a bar that has already CLOSED (the caller
        passes ``handler._ptr - 1``). This is LEAK-2 load-bearing: the gate
        signals encode the bar's own OHLCV, so evaluating an unclosed bar
        selects the agent's decision points with hindsight. See GATE-CAUSAL-01
        in the caller.

        Feature causality, stated precisely because the two levels are easy to
        conflate: the features are LEAK-1 compliant (``_ema_zscore_tanh``
        shifts the EMA mean/std by 1, so the normalisation statistics never see
        the current bar), but the feature VALUE at index ``i`` is still built
        from bar ``i``'s own data -- ``atr_norm[i]`` and ``parkinson[i]`` from
        its high/low/close with no shift, ``volume_z[i]``'s numerator from its
        volume. So ``features[i]`` is knowable only once bar ``i`` has closed,
        which is what makes the caller's ``-1`` mandatory rather than cosmetic.

        Supports both single-asset (T, 8) and multi-asset (T, N, 8) features.
        For multi-asset, averages signal across assets for portfolio-level gating.
        """
        # Passthrough if no features available
        if self._scale_features is None:
            return True

        # First step always open
        if self._is_first_step and self.gate_always_on_first:
            return True

        # Boundary check
        if ptr < 0 or ptr >= len(self._scale_features):
            return True

        features = self._scale_features[ptr]

        # Multi-asset features: (N, 8) — average across assets
        if features.ndim == 2:
            features = features.mean(axis=0)

        # Feature indices in MultiScaleOHLCVHandler:
        #   0: log_return, 1: atr_norm, 2: parkinson_vol,
        #   3-6: OHLC z-scores, 7: volume_z
        atr_norm = float(features[1])
        parkinson = float(features[2])
        volume_z = float(features[7]) if len(features) > 7 else 0.0

        if self.gate_mode == "atr":
            return atr_norm > self.atr_threshold
        elif self.gate_mode == "parkinson":
            return parkinson > self.parkinson_threshold
        elif self.gate_mode == "volume":
            return abs(volume_z) > self.volume_threshold
        elif self.gate_mode == "return":
            # abs_log_return gate — oracle-validated best single signal
            log_return = float(features[0])
            return abs(log_return) > self.return_threshold
        else:
            # Composite: ANY signal exceeding threshold opens the gate
            if atr_norm > self.atr_threshold:
                return True
            if parkinson > self.parkinson_threshold:
                return True
            if abs(volume_z) > self.volume_threshold:
                return True
            return False
