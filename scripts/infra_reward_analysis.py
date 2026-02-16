"""
Infrastructure Diagnostic 2: Reward Signal Analysis
=====================================================
Tests whether the DeepScalper reward is learnable by running 4 baselines:
  1. Random actions
  2. Always-hold (no trading)  
  3. Simple momentum (buy if recent returns positive)
  4. Oracle (peeks at next mid-price)

If random ≈ momentum → features are noise.
If oracle >> random → signal exists but agents can't extract it.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy import stats as sp_stats
import yaml


def load_config():
    config_path = "configs/deepscalper_rtx5090_production.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def make_env(config, start_date, end_date):
    """Create a DeepScalper env from config for a specific date range."""
    from finrl_pro_ds.data.parquet_handler import ParquetDataHandler
    from finrl_pro_ds.envs.deep_scalper_env import DeepScalperEnv

    handler = ParquetDataHandler(
        file_path=config['data']['file_path'],
        ticker=config['data']['ticker'],
        feature_config=config.get('features', {}),
        start_date=start_date,
        end_date=end_date,
    )

    env_cfg = config.get('env', {})
    env = DeepScalperEnv(config=env_cfg, data_handler=handler)
    return env


class RandomAgent:
    """Samples random actions from action space."""
    def __init__(self, env):
        self.action_space = env.action_space

    def act(self, obs):
        return self.action_space.sample()


class HoldAgent:
    """Always holds: price_idx=mid, qty_idx=center (0.0 = no trade)."""
    def __init__(self, env):
        n_price = env.action_space.nvec[0]
        n_qty = env.action_space.nvec[1]
        self.action = np.array([n_price // 2, n_qty // 2])

    def act(self, obs):
        return self.action.copy()


class MomentumAgent:
    """Buys if last N mid-prices trending up, sells if down."""
    def __init__(self, env, lookback=10):
        self.lookback = lookback
        self.n_price = env.action_space.nvec[0]
        self.n_qty = env.action_space.nvec[1]
        self.mid_history = []
        # action indices
        self.buy_price = self.n_price - 1      # highest offset = aggressive buy
        self.sell_price = 0                      # lowest offset = aggressive sell
        self.buy_qty = self.n_qty - 1            # max buy qty
        self.sell_qty = 0                        # max sell qty
        self.hold_qty = self.n_qty // 2          # center = 0

    def act(self, obs):
        # Try to extract mid-price from raw observation
        # Private state slot: position is in obs['private'][-1, 0]
        # We track via called update
        if len(self.mid_history) < self.lookback:
            return np.array([self.n_price // 2, self.hold_qty])

        recent = np.array(self.mid_history[-self.lookback:])
        momentum = recent[-1] - recent[0]
        if momentum > 0:
            return np.array([self.buy_price, self.buy_qty])
        elif momentum < 0:
            return np.array([self.sell_price, self.sell_qty])
        else:
            return np.array([self.n_price // 2, self.hold_qty])

    def update(self, mid_price):
        self.mid_history.append(mid_price)


class OracleAgent:
    """Cheats by looking at future mid-price to decide direction."""
    def __init__(self, env):
        self.n_price = env.action_space.nvec[0]
        self.n_qty = env.action_space.nvec[1]
        self.buy_price = self.n_price - 1
        self.sell_price = 0
        self.buy_qty = self.n_qty - 1
        self.sell_qty = 0
        self.hold_qty = self.n_qty // 2
        self.env = env

    def act(self, obs):
        """Peek at next mid-price to trade optimally."""
        env = self.env
        idx = env.current_step
        handler = env.handler

        # Get current and next mid-prices
        try:
            cur_bid = handler._data_arrays['bid_price_1'][idx]
            cur_ask = handler._data_arrays['ask_price_1'][idx]
            cur_mid = (cur_bid + cur_ask) / 2.0

            if idx + 1 < handler._len:
                nxt_bid = handler._data_arrays['bid_price_1'][idx + 1]
                nxt_ask = handler._data_arrays['ask_price_1'][idx + 1]
                nxt_mid = (nxt_bid + nxt_ask) / 2.0
            else:
                return np.array([self.n_price // 2, self.hold_qty])

            if nxt_mid > cur_mid:
                return np.array([self.buy_price, self.buy_qty])
            elif nxt_mid < cur_mid:
                return np.array([self.sell_price, self.sell_qty])
            else:
                return np.array([self.n_price // 2, self.hold_qty])
        except Exception:
            return np.array([self.n_price // 2, self.hold_qty])


def run_baseline(env, agent, name, max_steps=20000):
    """Run a baseline agent and collect metrics."""
    print(f"\n  Running {name}...", end=" ", flush=True)
    obs, info = env.reset()
    rewards = []
    trades = 0
    portfolio_values = [env._get_portfolio_value()]

    step = 0
    done = False
    while not done and step < max_steps:
        action = agent.act(obs)

        # Update momentum agent with current mid-price
        if hasattr(agent, 'update'):
            mid = (env.current_best_bid + env.current_best_ask) / 2.0
            agent.update(mid)

        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        portfolio_values.append(env._get_portfolio_value())

        if abs(env.position) > 0:
            trades += 1

        done = terminated or truncated
        step += 1

    rewards = np.array(rewards)
    portfolio_values = np.array(portfolio_values)

    # Metrics
    mean_r = rewards.mean()
    std_r = rewards.std() if len(rewards) > 1 else 0
    sharpe = mean_r / std_r if std_r > 1e-10 else 0
    cumulative_r = rewards.sum()
    pnl_pct = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
    
    # Max drawdown
    peak = np.maximum.accumulate(portfolio_values)
    dd = (portfolio_values - peak) / peak
    max_dd = dd.min() * 100

    print(f"done ({step} steps)")
    result = {
        'name': name,
        'steps': step,
        'mean_reward': mean_r,
        'std_reward': std_r,
        'sharpe': sharpe,
        'cumulative_reward': cumulative_r,
        'pnl_pct': pnl_pct,
        'trades': trades,
        'max_drawdown': max_dd,
        'rewards': rewards,
    }
    return result


def hr(title):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")


if __name__ == "__main__":
    hr("INFRASTRUCTURE DIAGNOSTIC 2: REWARD SIGNAL ANALYSIS")

    config = load_config()
    
    # Run on TRAIN data first (agent should at least learn here)
    for period_name, start, end in [
        ("TRAIN (Jan-Mar sample)", "2025-01-01 00:00:00", "2025-03-31 23:59:59"),
        ("TEST  (Dec)",           "2025-12-01 00:00:00", "2025-12-31 23:59:59"),
    ]:
        hr(f"PERIOD: {period_name}")
        
        env = make_env(config, start, end)
        results = {}

        # 1. Random
        agent = RandomAgent(env)
        results['Random'] = run_baseline(env, agent, "Random Agent")

        # 2. Hold
        env2 = make_env(config, start, end)
        agent2 = HoldAgent(env2)
        results['Hold'] = run_baseline(env2, agent2, "Hold Agent")

        # 3. Momentum
        env3 = make_env(config, start, end)
        agent3 = MomentumAgent(env3)
        results['Momentum'] = run_baseline(env3, agent3, "Momentum Agent")

        # 4. Oracle
        env4 = make_env(config, start, end)
        agent4 = OracleAgent(env4)
        results['Oracle'] = run_baseline(env4, agent4, "Oracle Agent")

        # Summary table
        hr(f"RESULTS: {period_name}")
        header = f"  {'Agent':<15} {'Steps':>7} {'Mean R':>10} {'Std R':>10} {'Sharpe':>8} {'Cum R':>10} {'PnL%':>8} {'Trades':>7} {'MaxDD%':>8}"
        print(header)
        print(f"  {'-'*15} {'-'*7} {'-'*10} {'-'*10} {'-'*8} {'-'*10} {'-'*8} {'-'*7} {'-'*8}")
        for name, r in results.items():
            print(f"  {name:<15} {r['steps']:>7} {r['mean_reward']:>10.4f} {r['std_reward']:>10.4f} "
                  f"{r['sharpe']:>8.4f} {r['cumulative_reward']:>10.1f} {r['pnl_pct']:>+7.2f}% "
                  f"{r['trades']:>7} {r['max_drawdown']:>+7.2f}%")

        # Statistical tests: is oracle significantly better than random?
        print(f"\n  STATISTICAL TESTS:")
        for pair in [('Random', 'Hold'), ('Random', 'Momentum'), ('Random', 'Oracle')]:
            if pair[0] in results and pair[1] in results:
                r1 = results[pair[0]]['rewards']
                r2 = results[pair[1]]['rewards']
                min_len = min(len(r1), len(r2))
                t_stat, p_val = sp_stats.ttest_ind(r1[:min_len], r2[:min_len], equal_var=False)
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                print(f"    {pair[0]} vs {pair[1]}: t={t_stat:+.3f}  p={p_val:.2e}  {sig}")

    hr("VERDICT")
    print("""
  INTERPRETATION GUIDE:
  - If Oracle >> Random (p < 0.01): Signal EXISTS in the data, agents can't extract it.
    → Problem is in the agent/observation space, not the environment.
  - If Oracle ≈ Random: NO learnable signal at minute-level.
    → Problem is fundamental: BTC at 1-min resolution may not have alpha.
  - If Momentum > Hold (p < 0.05): Simple features have predictive power.
    → Observation space should work; check agent capacity.
  - If all rewards near zero: Fees + slippage dominate any signal.
    → Need to reduce fees or widen the reward scale.
    """)
    hr("DIAGNOSTIC 2 COMPLETE")
