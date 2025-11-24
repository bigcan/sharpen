"""
Paper Trading Driver for FinRL Pro (Phase 8).

Orchestrates the full loop:
1. Data Fetch (Alpaca)
2. Feature Engineering (ProFeatureAssembler)
3. Regime Detection (HMM)
4. Inference (Ensemble Soft Voting)
5. Risk Checks (Circuit Breaker)
6. Execution (Alpaca)
"""

import os
import time
import logging
import pandas as pd
import numpy as np
import torch
import mlflow
import json
from datetime import datetime
from dotenv import load_dotenv

from finrl_pro.execution.alpaca_broker import AlpacaBroker
from finrl_pro.data.loader_pro import ProFeatureAssembler
from finrl_pro.data.regimes import HMMRegimeDetector, MarketRegime
from finrl_pro.agents.ppo import PPOAgent

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("paper_trading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load Env Vars
load_dotenv()

# Configuration
TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "JPM", "V", "JNJ",
    "WMT", "PG", "XOM", "UNH", "MA", "HD", "CVX", "MRK", "ABBV", "KO"
]
# Hardcoded Model Paths (or MLflow URIs)
MODEL_PATHS = {
    "bull": "models/phase6/bull_agent.pth", # Placeholder paths, need to verify
    "bear": "models/phase6/bear_agent.pth",
    "sideways": "models/phase6/sideways_agent.pth"
}
# MLflow Run IDs (Alternative)
RUN_IDS = {
    "bull": "74324eece6cd446c837e055cf34f6415",
    "bear": "b3bd16fc4b5349a9a267a1fc84427e61",
    "sideways": "cf31189fd7884338be0958d6761056dc"
}

CIRCUIT_BREAKER_DRAWDOWN = 0.15 # 15% Max Drawdown
RISK_LIMIT_LEVERAGE = 1.0

class PaperTradingSystem:
    def __init__(self):
        self.broker = AlpacaBroker(paper=True)
        self.assembler = ProFeatureAssembler()
        self.regime_detector = HMMRegimeDetector(n_components=3, random_state=42)
        
        self.agents = {}
        self.device = "cpu" # Force CPU for inference safety/portability
        self.state_dim = 0
        self.action_dim = 0
        
        self.state_file = "trading_state.json"
        self._load_models()
        
    def _load_state(self):
        if os.path.exists(self.state_file):
            with open(self.state_file, "r") as f:
                return json.load(f)
        return {"peak_equity": 0.0}

    def _save_state(self, state):
        with open(self.state_file, "w") as f:
            json.dump(state, f)

    def _load_models(self):
        """Load ensemble agents from MLflow."""
        logger.info("Loading Ensemble Agents...")
        # We need to know dims. Hardcoded for now based on Phase 6.
        # 183 state dim, 20 action dim (18 tickers? Wait, TICKERS list has 20).
        # Phase 6 config had 20 tickers. 
        # Let's verify dims dynamically if possible or assume standard.
        
        # Assuming 183 state, 20 action (if universe is 20)
        # Wait, previous logs showed State Dim: 183, Action Dim: 18.
        # Why 18? TICKERS list in run_phase6 had 20 items.
        # Ah, the run_phase6_ensemble.py script filtered tickers?
        # Let's stick to the list used in training.
        
        # Check run_phase6_ensemble.py:
        # TICKERS = ["AAPL", ... "KO"] (Length 20)
        # But logs said Action Dim: 18. 
        # Maybe 2 tickers failed to load data or were filtered?
        # Proceed with caution.
        self.state_dim = 183 # From logs
        self.action_dim = 18 # From logs
        
        for regime, run_id in RUN_IDS.items():
            try:
                local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model/temp_model.pth")
                agent = PPOAgent(state_dim=self.state_dim, action_dim=self.action_dim, device=self.device)
                agent.policy.load_state_dict(torch.load(local_path, map_location=self.device))
                self.agents[regime] = agent
                logger.info(f"Loaded {regime} agent from {run_id}")
            except Exception as e:
                logger.error(f"Failed to load {regime} agent: {e}")
                raise e

    def fetch_data_and_features(self):
        """Fetch live bars and compute features."""
        logger.info("Fetching live data...")
        # Fetch 252 days to ensure valid rolling windows for features
        df = self.broker.get_bar_data(TICKERS, timeframe="1Day", limit=300)
        
        # Reset Index to get 'tic' and 'date' columns
        # Alpaca returns multi-index (symbol, timestamp)
        df = df.reset_index()
        df.rename(columns={"symbol": "tic", "timestamp": "date"}, inplace=True)
        
        # Ensure datetime is tz-naive or consistent
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        
        # Feature Assembly
        features_cfg = {
            "stockstats_overrides": [
                "macd", "boll_ub", "boll_lb", "rsi_30", "dx_30", "close_30_sma", "close_60_sma"
            ],
            "use_turbulence": True
        }
        
        # Assemble features
        # This returns an Assembly object with arrays.
        # We need the LAST row of the arrays for inference.
        asm = self.assembler.assemble_from_df(df=df, features_cfg=features_cfg, dataset_hash="live")
        
        return asm

    def detect_regime(self, df):
        """Detect regime using HMM on market proxy."""
        # Calculate Market Returns
        piv = df.pivot(index="date", columns="tic", values="close")
        returns = piv.pct_change().fillna(0)
        market_returns = returns.mean(axis=1).dropna()
        
        # Rolling Probabilities
        # We only need the latest probability
        # But we need history to fit?
        # HMMRegimeDetector.rolling_predict_proba fits on windows.
        
        probs_df = self.regime_detector.rolling_predict_proba(market_returns, window=252, min_periods=60)
        
        # Apply Hysteresis (3-day)
        hysteresis = probs_df.rolling(window=3).mean().iloc[-1]
        hysteresis = hysteresis / hysteresis.sum() # Normalize
        
        return hysteresis

    def run_cycle(self):
        """Main execution cycle."""
        logger.info("=== Starting Trading Cycle ===")
        
        # 1. Risk Check (Circuit Breaker)
        acct = self.broker.get_account()
        equity = acct["equity"]
        
        state = self._load_state()
        peak_equity = max(state.get("peak_equity", 0.0), equity)
        state["peak_equity"] = peak_equity
        self._save_state(state)
        
        drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        logger.info(f"Equity: ${equity:.2f}, Peak: ${peak_equity:.2f}, Drawdown: {drawdown:.2%}")
        
        if drawdown > CIRCUIT_BREAKER_DRAWDOWN:
            logger.critical(f"CIRCUIT BREAKER TRIGGERED! Drawdown {drawdown:.2%} > Limit {CIRCUIT_BREAKER_DRAWDOWN:.2%}")
            logger.critical("Closing ALL positions and halting.")
            self.broker.close_all_positions()
            return

        # 2. Data & Features
        try:
            asm = self.fetch_data_and_features()
            # Reconstruct DataFrame for Regime Detection
            # (We need raw prices for market proxy)
            # asm has price_ary, but we need pandas for HMM detector convenience
            # Or we can just use the df from fetch_data step.
            # Let's refactor fetch_data to return df as well.
            # For now, fetch again or improve flow. 
            # Let's assume fetch_data handles it.
            
            # Re-fetch specifically for regime (inefficient but safe for MVP)
            df_raw = self.broker.get_bar_data(TICKERS, limit=300).reset_index()
            df_raw.rename(columns={"symbol": "tic", "timestamp": "date"}, inplace=True)
            df_raw["date"] = pd.to_datetime(df_raw["date"]).dt.tz_localize(None)
            
            probs = self.detect_regime(df_raw)
            logger.info(f"Regime Probabilities: {probs.to_dict()}")
            
        except Exception as e:
            logger.error(f"Data pipeline failed: {e}")
            return

        # 3. Inference (Soft Voting)
        # Get latest observation
        # asm.tech_ary shape (T, stock*7) -> take -1
        # asm.price_ary shape (T, stock) -> take -1
        # Construct state vector matching PPO input
        # State: [Amount, Turb, TurbBool, Prices, Stocks, CD, Tech]
        # This state construction logic MUST match `ProStockEnv`.
        # This is tricky without the Env object.
        # Solution: Instantiate a dummy ProStockEnv with the assembly and reset/step to end?
        # Or manually build the vector.
        
        # Better: Use `make_pro_env` with the assembly, reset, and set `day` to last day.
        from finrl_pro.envs.factory import make_pro_env
        env = make_pro_env(asm, initial_capital=equity)
        env.reset()
        # Fast forward to last day? 
        # Env automatically starts at day 0.
        # We want the state for the *next* trading day (which relies on data up to T).
        # Actually, training env steps through history.
        # Live env: The "current state" is the features derived from data up to NOW.
        # So we treat the assembled data as "history" and we want the state at T.
        
        # If we set env.day = len(asm.dates) - 1, that's the last valid data point.
        # obs = env.get_state(day)
        env.day = len(asm.dates) - 1
        obs = env._get_observation()
        
        # Soft Voting
        weighted_action = np.zeros(self.action_dim)
        
        mapping = {
            MarketRegime.BULL.value: "bull",
            MarketRegime.BEAR.value: "bear",
            MarketRegime.CRISIS.value: "bear", # Map Crisis to Bear
            MarketRegime.SIDEWAYS.value: "sideways"
        }
        
        for regime_val, prob in probs.items():
            regime_key = mapping.get(int(regime_val))
            if regime_key and regime_key in self.agents:
                agent = self.agents[regime_key]
                action, _ = agent.select_action(obs, deterministic=True)
                weighted_action += prob * action
                
        logger.info(f"Ensemble Action: {weighted_action}")
        
        # 4. Execution
        # Convert Action (Allocation weights or quantities) to Orders
        # PPO output is usually weights/actions in [-1, 1].
        # Need to normalize to target portfolio weights.
        
        # Normalize to [0, 1] for long-only
        # actions = softmax(actions)
        weights = np.exp(weighted_action) / np.sum(np.exp(weighted_action))
        logger.info(f"Target Weights: {weights}")
        
        # Diff against current positions
        current_positions = self.broker.get_positions()
        pos_dict = {p["symbol"]: p["market_value"] for p in current_positions}
        
        # Rebalance logic
        # For each ticker, calc target value, diff, and submit order.
        # Note: asm.tickers might not match TICKERS list order?
        # asm.tickers is sorted. PPO action corresponds to asm.tickers.
        
        for i, tic in enumerate(asm.tickers):
            target_val = equity * weights[i]
            current_val = pos_dict.get(tic, 0.0)
            diff = target_val - current_val
            
            # Threshold to avoid tiny trades ($100?)
            if abs(diff) < 100:
                continue
                
            # Get price for quantity
            quotes = self.broker.get_latest_quotes([tic])
            price = quotes.get(tic, 0.0)
            if price == 0: continue
            
            qty = int(abs(diff) / price)
            if qty == 0: continue
            
            side = "buy" if diff > 0 else "sell"
            logger.info(f"Submitting Order: {side.upper()} {qty} {tic} (Diff: ${diff:.2f})")
            
            try:
                self.broker.submit_order(tic, qty, side)
            except Exception as e:
                logger.error(f"Order failed: {e}")

        logger.info("Cycle Complete.")

if __name__ == "__main__":
    system = PaperTradingSystem()
    system.run_cycle()
