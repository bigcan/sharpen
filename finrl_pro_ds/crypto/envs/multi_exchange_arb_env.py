"""
multi_exchange_arb_env.py — Gymnasium environment for multi-exchange funding rate arbitrage.

Imported from https://github.com/bigcan/Funding-Rate-Arb.git
Adapted imports for FinRL-Pro_DS project structure.

State: flattened market data + position info + portfolio summary
Actions: per-slot [action_type, symbol_idx, size_idx] × max_open_positions
Reward: (funding_income - costs - basis_risk - opportunity_cost) / initial_capital * 100
"""

from dataclasses import dataclass, field
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class EnvConfig:
    """Configuration for the funding rate arbitrage environment."""
    symbols: list[str] = field(default_factory=lambda: [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "ARB/USDT", "DOGE/USDT"
    ])
    exchanges: list[str] = field(default_factory=lambda: ["binance", "bybit", "okx"])
    initial_capital: float = 100_000
    max_open_positions: int = 3
    max_steps: int = 500
    taker_fee: float = 0.0004
    slippage_bps: float = 0.0001
    opportunity_cost_rate: float = 0.00005
    basis_risk_weight: float = 0.5


@dataclass
class Position:
    """Tracks an open carry-trade position."""
    symbol: str
    short_exchange: str
    sym_idx: int
    exchange_idx: int
    direction: int  # +1 = positive carry (short perp), -1 = reverse carry
    entry_basis: float
    size_usd: float
    funding_accumulated: float = 0.0
    entry_step: int = 0
    unrealized_pnl: float = 0.0


class MultiExchangeArbEnv(gym.Env):
    """
    Gymnasium environment for multi-exchange funding rate arbitrage.

    Observation: flat vector of market features + position info + portfolio state.
    Action: MultiDiscrete array — per position slot: [action_type, symbol_idx, size_idx].

    Key differences from FundingArbEnv:
      - Multi-exchange support (cross-exchange arb opportunities)
      - 4D market data (steps × symbols × exchanges × features)
      - Per-slot position management with explicit open/close actions
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        market_data: Optional[np.ndarray] = None,
        config: Optional[EnvConfig] = None,
    ):
        super().__init__()
        self.config = config or EnvConfig()
        cfg = self.config

        self.n_sym = len(cfg.symbols)
        self.n_ex = len(cfg.exchanges)
        self.n_features = 7

        # Market data: (n_steps, n_sym, n_ex, 7)
        if market_data is not None:
            self.market_data = market_data.astype(np.float32)
        else:
            self.market_data = self._generate_synthetic_data()

        # Size options: index → fraction of capital
        self._size_fractions = {0: 0.0, 1: 0.15, 2: 0.20}

        # Observation space
        market_size = self.n_sym * self.n_ex * self.n_features
        pos_size = (self.n_sym + self.n_ex + 4) * cfg.max_open_positions
        portfolio_size = 3
        self._obs_size = market_size + pos_size + portfolio_size

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._obs_size,), dtype=np.float32,
        )

        # Action space: max_open_positions × 3 values each
        self.action_space = spaces.MultiDiscrete(
            [4, self.n_sym, 3] * cfg.max_open_positions
        )

        # State
        self.positions: list[Position] = []
        self.trade_log: list[dict] = []
        self.current_step: int = 0
        self.cash: float = cfg.initial_capital
        self.portfolio_value: float = cfg.initial_capital

    # ------------------------------------------------------------------ #
    #  Gym interface                                                       #
    # ------------------------------------------------------------------ #

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        cfg = self.config
        self.current_step = 0
        self.cash = cfg.initial_capital
        self.portfolio_value = cfg.initial_capital
        self.positions = []
        self.trade_log = []

        obs = self._get_obs()
        info = {"portfolio_value": self.portfolio_value}
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=int).flatten()
        cfg = self.config

        step_funding = 0.0
        step_costs = 0.0
        step_basis_risk = 0.0
        slots_to_remove = set()

        # --- 1. Process actions per slot ---
        for slot in range(cfg.max_open_positions):
            offset = slot * 3
            if offset + 2 >= len(action):
                break
            act_type = int(action[offset])
            sym_idx = int(action[offset + 1]) % self.n_sym
            size_idx = int(action[offset + 2]) % 3

            if act_type == 0:
                # HOLD
                continue

            elif act_type in (1, 2):
                # OPEN position
                if slot < len(self.positions):
                    continue  # slot occupied
                frac = self._size_fractions.get(size_idx, 0.0)
                if frac == 0.0:
                    continue
                size_usd = frac * cfg.initial_capital
                cost = size_usd * (cfg.taker_fee + cfg.slippage_bps) * 2  # both legs
                if self.cash < size_usd + cost:
                    continue  # insufficient cash
                if len(self.positions) >= cfg.max_open_positions:
                    continue

                # Find best exchange for this symbol
                best_ex = 0
                best_fr = abs(self._get_funding_rate(sym_idx, 0))
                for e_i in range(1, self.n_ex):
                    fr_abs = abs(self._get_funding_rate(sym_idx, e_i))
                    if fr_abs > best_fr:
                        best_fr = fr_abs
                        best_ex = e_i

                direction = 1 if act_type == 1 else -1
                data_idx = min(self.current_step, len(self.market_data) - 1)
                entry_basis = float(self.market_data[data_idx, sym_idx, best_ex, 4])

                pos = Position(
                    symbol=cfg.symbols[sym_idx],
                    short_exchange=cfg.exchanges[best_ex],
                    sym_idx=sym_idx,
                    exchange_idx=best_ex,
                    direction=direction,
                    entry_basis=entry_basis,
                    size_usd=size_usd,
                    entry_step=self.current_step,
                )
                self.cash -= size_usd + cost
                step_costs += cost
                self.positions.append(pos)

                self.trade_log.append({
                    "action": "OPEN",
                    "step": self.current_step,
                    "symbol": pos.symbol,
                    "exchange": pos.short_exchange,
                    "direction": direction,
                    "size_usd": size_usd,
                    "cost": cost,
                })

            elif act_type == 3:
                # CLOSE position
                if slot >= len(self.positions):
                    continue
                pos = self.positions[slot]
                data_idx = min(self.current_step, len(self.market_data) - 1)
                current_basis = float(
                    self.market_data[data_idx, pos.sym_idx, pos.exchange_idx, 4]
                )
                basis_pnl = pos.direction * (current_basis - pos.entry_basis) * pos.size_usd
                closing_cost = pos.size_usd * (cfg.taker_fee + cfg.slippage_bps) * 2
                net_pnl = pos.funding_accumulated + basis_pnl - closing_cost

                self.cash += pos.size_usd + net_pnl
                step_costs += closing_cost
                slots_to_remove.add(slot)

                self.trade_log.append({
                    "action": "CLOSE",
                    "step": self.current_step,
                    "symbol": pos.symbol,
                    "exchange": pos.short_exchange,
                    "direction": pos.direction,
                    "size_usd": pos.size_usd,
                    "net_pnl": net_pnl,
                    "funding_collected": pos.funding_accumulated,
                    "basis_pnl": basis_pnl,
                    "cost": closing_cost,
                    "holding_period": self.current_step - pos.entry_step,
                })

        # --- 2. Compact positions (remove closed) ---
        if slots_to_remove:
            self.positions = [
                p for i, p in enumerate(self.positions) if i not in slots_to_remove
            ]

        # --- 3. Settle funding on open positions ---
        for pos in self.positions:
            fr = self._get_funding_rate(pos.sym_idx, pos.exchange_idx)
            funding = pos.direction * fr * pos.size_usd
            pos.funding_accumulated += funding
            step_funding += funding

            # Update unrealized P&L
            data_idx = min(self.current_step, len(self.market_data) - 1)
            current_basis = float(
                self.market_data[data_idx, pos.sym_idx, pos.exchange_idx, 4]
            )
            pos.unrealized_pnl = (
                pos.direction * (current_basis - pos.entry_basis) * pos.size_usd
            )

            # Basis risk
            step_basis_risk += abs(current_basis - pos.entry_basis) * pos.size_usd

        # --- 4. Update portfolio value ---
        position_value = sum(
            p.size_usd + p.unrealized_pnl + p.funding_accumulated
            for p in self.positions
        )
        self.portfolio_value = self.cash + position_value

        # --- 5. Compute reward ---
        idle_cash = max(self.cash, 0)
        opportunity_cost = idle_cash * cfg.opportunity_cost_rate
        reward = (
            step_funding - step_costs
            - cfg.basis_risk_weight * step_basis_risk
            - opportunity_cost
        ) / cfg.initial_capital * 100

        # --- 6. Advance step ---
        self.current_step += 1
        terminated = (
            self.current_step >= cfg.max_steps
            or self.current_step >= len(self.market_data)
        )
        drawdown = 1 - self.portfolio_value / cfg.initial_capital
        truncated = drawdown >= 0.5

        obs = self._get_obs()
        info = {"portfolio_value": self.portfolio_value}

        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------ #
    #  Observation builder                                                 #
    # ------------------------------------------------------------------ #

    def _get_obs(self) -> np.ndarray:
        cfg = self.config
        data_idx = min(self.current_step, len(self.market_data) - 1)

        # Market slice: flatten (n_sym, n_ex, 7)
        market_flat = self.market_data[data_idx].flatten()

        # Position encoding
        pos_parts = []
        for slot in range(cfg.max_open_positions):
            if slot < len(self.positions):
                pos = self.positions[slot]
                sym_oh = np.zeros(self.n_sym, dtype=np.float32)
                sym_oh[pos.sym_idx] = 1.0
                ex_oh = np.zeros(self.n_ex, dtype=np.float32)
                ex_oh[pos.exchange_idx] = 1.0
                direction = float(pos.direction)
                size_ratio = pos.size_usd / cfg.initial_capital
                funding_norm = pos.funding_accumulated / max(pos.size_usd, 1.0)
                steps_held = (self.current_step - pos.entry_step) / max(cfg.max_steps, 1)
                pos_parts.append(np.concatenate([
                    sym_oh, ex_oh,
                    np.array(
                        [direction, size_ratio, funding_norm, steps_held],
                        dtype=np.float32,
                    ),
                ]))
            else:
                pos_parts.append(
                    np.zeros(self.n_sym + self.n_ex + 4, dtype=np.float32)
                )

        pos_flat = np.concatenate(pos_parts)

        # Portfolio summary
        cash_ratio = self.cash / cfg.initial_capital
        n_pos_norm = len(self.positions) / max(cfg.max_open_positions, 1)
        pv_ratio = self.portfolio_value / cfg.initial_capital
        portfolio = np.array([cash_ratio, n_pos_norm, pv_ratio], dtype=np.float32)

        obs = np.concatenate([market_flat, pos_flat, portfolio]).astype(np.float32)

        # Safety: replace NaN/Inf
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs

    # ------------------------------------------------------------------ #
    #  Public helpers                                                       #
    # ------------------------------------------------------------------ #

    def _get_funding_rate(self, sym_idx: int, exchange_idx: int) -> float:
        """Return current funding rate for a symbol-exchange pair."""
        data_idx = min(self.current_step, len(self.market_data) - 1)
        return float(self.market_data[data_idx, sym_idx, exchange_idx, 0])

    # ------------------------------------------------------------------ #
    #  Synthetic data generation                                           #
    # ------------------------------------------------------------------ #

    def _generate_synthetic_data(self) -> np.ndarray:
        """Generate realistic synthetic market data for testing."""
        cfg = self.config
        rng = np.random.default_rng(42)
        n_steps = cfg.max_steps + 50
        data = np.zeros(
            (n_steps, self.n_sym, self.n_ex, self.n_features), dtype=np.float32
        )

        for s in range(self.n_sym):
            for e in range(self.n_ex):
                # Channel 0: FR current — mean-reverting with regime shifts
                fr = np.zeros(n_steps)
                fr[0] = rng.normal(0.0001, 0.0001)
                regime = 0.0001
                for t in range(1, n_steps):
                    if rng.random() < 0.02:
                        regime = rng.choice(
                            [-0.0003, -0.0001, 0.0001, 0.0003, 0.0005]
                        )
                    fr[t] = (
                        fr[t - 1]
                        + 0.3 * (regime - fr[t - 1])
                        + rng.normal(0, 0.00005)
                    )
                data[:, s, e, 0] = fr

                # Channel 1: FR predicted (EMA, alpha=0.7)
                ema = np.zeros(n_steps)
                ema[0] = fr[0]
                for t in range(1, n_steps):
                    ema[t] = 0.7 * fr[t] + 0.3 * ema[t - 1]
                data[:, s, e, 1] = ema

                # Channel 2: FR 7d mean (21-period rolling)
                for t in range(n_steps):
                    start = max(0, t - 20)
                    data[t, s, e, 2] = np.mean(fr[start:t + 1])

                # Channel 3: FR 7d std (21-period rolling)
                for t in range(n_steps):
                    start = max(0, t - 20)
                    window = fr[start:t + 1]
                    data[t, s, e, 3] = np.std(window) if len(window) > 1 else 0.0

                # Channel 4: Basis spread — correlated with FR + noise
                basis = fr * 0.5 + rng.normal(0, 0.0002, n_steps)
                data[:, s, e, 4] = basis.astype(np.float32)

                # Channel 5: ATR normalized — autocorrelated, range [0.005, 0.1]
                atr = np.zeros(n_steps)
                atr[0] = rng.uniform(0.01, 0.03)
                for t in range(1, n_steps):
                    atr[t] = 0.95 * atr[t - 1] + 0.05 * rng.uniform(0.005, 0.1)
                data[:, s, e, 5] = np.clip(atr, 0.005, 0.1).astype(np.float32)

                # Channel 6: Volume change — normal(0, 0.3) clipped
                vol_chg = rng.normal(0, 0.3, n_steps)
                data[:, s, e, 6] = np.clip(vol_chg, -2, 2).astype(np.float32)

        return data
