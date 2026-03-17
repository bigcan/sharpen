"""
hummingbot_bridge.py — Live execution bridge for funding rate arb via Hummingbot Gateway API.

Imported from https://github.com/bigcan/Funding-Rate-Arb.git
Adapted for FinRL-Pro_DS project structure.

Bridges the trained PPO agent to live order execution:
  1. Fetches live market data (funding rates, OHLCV)
  2. Builds observations in the same format as the training env
  3. Runs the trained model to get actions
  4. Translates actions into Hummingbot API calls

NOTE: Actual API integration requires Hummingbot Gateway running
and exchange credentials configured.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class BridgeConfig:
    """Configuration for the Hummingbot bridge."""
    hummingbot_host: str = "localhost"
    hummingbot_port: int = 8000
    poll_interval_seconds: int = 30
    model_path: str = "models/fr_arb_ppo_v1/policy"
    symbols: list = field(default_factory=lambda: [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "ARB/USDT", "DOGE/USDT"
    ])
    exchanges: list = field(default_factory=lambda: ["binance", "bybit", "okx"])
    max_open_positions: int = 3
    initial_capital: float = 100_000

    @property
    def base_url(self) -> str:
        return f"http://{self.hummingbot_host}:{self.hummingbot_port}"


class HummingbotBridge:
    """
    Connects a trained PPO model to live markets via Hummingbot Gateway.

    Lifecycle:
        bridge = HummingbotBridge(config)
        bridge.run()  # blocks, polls every N seconds
    """

    ACTION_NAMES = {
        0: "HOLD",
        1: "OPEN_POSITIVE_CARRY",
        2: "OPEN_REVERSE_CARRY",
        3: "CLOSE",
    }

    def __init__(self, config: BridgeConfig):
        self.config = config
        self.model = None
        self.n_sym = len(config.symbols)
        self.n_ex = len(config.exchanges)
        self.open_positions: list[dict] = []
        self._observation_history: list[np.ndarray] = []

    def load_model(self):
        """Load the trained PPO model."""
        from stable_baselines3 import PPO

        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        self.model = PPO.load(str(model_path))
        logger.info(f"Loaded model from {model_path}")

    def build_live_observation(self) -> np.ndarray:
        """
        Build observation vector from live market data.

        Format matches MultiExchangeArbEnv._get_obs():
          - Market: n_sym * n_ex * 7 floats
          - Positions: max_open_positions * (n_sym + n_ex + 4) floats
          - Portfolio: 3 floats
        """
        cfg = self.config

        # Fetch live market data
        market_flat = np.zeros(self.n_sym * self.n_ex * 7, dtype=np.float32)
        for s_i, symbol in enumerate(cfg.symbols):
            for e_i, exchange in enumerate(cfg.exchanges):
                features = self._fetch_market_features(symbol, exchange)
                offset = (s_i * self.n_ex + e_i) * 7
                market_flat[offset:offset + 7] = features

        # Encode positions
        pos_parts = []
        for slot in range(cfg.max_open_positions):
            if slot < len(self.open_positions):
                pos = self.open_positions[slot]
                sym_oh = np.zeros(self.n_sym, dtype=np.float32)
                sym_oh[pos["sym_idx"]] = 1.0
                ex_oh = np.zeros(self.n_ex, dtype=np.float32)
                ex_oh[pos["exchange_idx"]] = 1.0
                pos_parts.append(np.concatenate([
                    sym_oh, ex_oh,
                    np.array([
                        pos["direction"],
                        pos["size_usd"] / cfg.initial_capital,
                        pos.get("funding_accumulated", 0) / max(pos["size_usd"], 1),
                        min(pos.get("steps_held", 0) / 500.0, 1.0),
                    ], dtype=np.float32),
                ]))
            else:
                pos_parts.append(
                    np.zeros(self.n_sym + self.n_ex + 4, dtype=np.float32)
                )

        pos_flat = np.concatenate(pos_parts)

        # Portfolio summary
        total_position_value = sum(p["size_usd"] for p in self.open_positions)
        cash = cfg.initial_capital - total_position_value
        portfolio = np.array([
            cash / cfg.initial_capital,
            len(self.open_positions) / max(cfg.max_open_positions, 1),
            1.0,
        ], dtype=np.float32)

        obs = np.concatenate([market_flat, pos_flat, portfolio])
        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    def execute_action(self, action: np.ndarray):
        """
        Translate action array into Hummingbot API calls.

        Action format: [act_type, sym_idx, size_idx] × max_open_positions
        """
        cfg = self.config
        action = np.asarray(action, dtype=int).flatten()

        for slot in range(cfg.max_open_positions):
            offset = slot * 3
            if offset + 2 >= len(action):
                break
            act_type = int(action[offset])
            sym_idx = int(action[offset + 1]) % self.n_sym
            size_idx = int(action[offset + 2]) % 3

            act_name = self.ACTION_NAMES.get(act_type, "UNKNOWN")
            symbol = cfg.symbols[sym_idx]

            if act_type == 0:
                continue
            elif act_type in (1, 2):
                direction = 1 if act_type == 1 else -1
                size_pct = {0: 0, 1: 15, 2: 20}.get(size_idx, 0)
                if size_pct == 0:
                    continue
                logger.info(
                    f"[{act_name}] {symbol} | {size_pct}% | dir={direction}"
                )
                self._open_position(symbol, sym_idx, direction, size_pct)
            elif act_type == 3:
                if slot < len(self.open_positions):
                    pos = self.open_positions[slot]
                    logger.info(f"[CLOSE] {pos['symbol']}")
                    self._close_position(slot)

    def run(self):
        """Main execution loop — observe, predict, execute."""
        self.load_model()
        logger.info("Starting live execution loop...")
        logger.info(f"  Poll interval: {self.config.poll_interval_seconds}s")
        logger.info(f"  Symbols: {self.config.symbols}")
        logger.info(f"  Exchanges: {self.config.exchanges}")

        while True:
            try:
                obs = self.build_live_observation()
                action, _ = self.model.predict(obs, deterministic=True)
                self.execute_action(action)
                time.sleep(self.config.poll_interval_seconds)
            except KeyboardInterrupt:
                logger.info("Shutting down bridge...")
                break
            except Exception as e:
                logger.error(f"Error in execution loop: {e}")
                time.sleep(self.config.poll_interval_seconds)

    # ------------------------------------------------------------------ #
    #  Private — API interactions (skeleton)                               #
    # ------------------------------------------------------------------ #

    def _fetch_market_features(self, symbol: str, exchange: str) -> np.ndarray:
        """
        Fetch 7 features for a symbol-exchange pair from live data.

        Returns: [fr_current, fr_predicted, fr_7d_mean, fr_7d_std,
                  basis_spread, atr_normalized, volume_change]

        TODO: Implement actual API calls to fetch:
          - Funding rate from exchange API via CCXT
          - OHLCV for ATR and volume computation
          - Spot vs perp prices for basis spread
        """
        logger.debug(f"Fetching features for {symbol} on {exchange}")
        return np.zeros(7, dtype=np.float32)

    def _open_position(
        self, symbol: str, sym_idx: int, direction: int, size_pct: int
    ):
        """
        Open a carry trade position via Hummingbot API.

        TODO: Implement actual order placement via Hummingbot Gateway:
          POST {base_url}/clob/place_order
        """
        size_usd = self.config.initial_capital * size_pct / 100
        self.open_positions.append({
            "symbol": symbol,
            "sym_idx": sym_idx,
            "exchange_idx": 0,
            "direction": direction,
            "size_usd": size_usd,
            "funding_accumulated": 0,
            "steps_held": 0,
        })
        logger.info(f"  → Position opened: {symbol} ${size_usd:,.0f} dir={direction}")

    def _close_position(self, slot: int):
        """
        Close a carry trade position via Hummingbot API.

        TODO: Implement actual order cancellation/closing via Hummingbot Gateway:
          DELETE {base_url}/clob/cancel_order
        """
        if slot < len(self.open_positions):
            pos = self.open_positions.pop(slot)
            logger.info(
                f"  → Position closed: {pos['symbol']} ${pos['size_usd']:,.0f}"
            )

    def check_gateway_health(self) -> bool:
        """Check if Hummingbot Gateway is reachable."""
        try:
            resp = requests.get(f"{self.config.base_url}/", timeout=5)
            return resp.status_code == 200
        except requests.ConnectionError:
            return False


if __name__ == "__main__":
    config = BridgeConfig()
    bridge = HummingbotBridge(config)

    if bridge.check_gateway_health():
        bridge.run()
    else:
        logger.error(
            f"Hummingbot Gateway not reachable at {config.base_url}. "
            "Start it with: `gateway start` or check configs."
        )
