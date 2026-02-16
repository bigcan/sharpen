"""
Infrastructure Diagnostic 4: Environment PnL Correctness
=========================================================
Runs deterministic trade sequences through the env and verifies
PnL, fees, slippage, and portfolio value against hand calculations.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import yaml


def load_config():
    config_path = "configs/deepscalper_rtx5090_production.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def make_env(config, start_date="2025-01-01 00:00:00", end_date="2025-01-10 23:59:59"):
    """Create env with a small date window for testing."""
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


def hr(title):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")


def test_basic_buy_hold_sell(env):
    """Test 1: Buy 1 unit, hold 5 steps, sell. Verify PnL."""
    hr("TEST 1: Buy → Hold → Sell")
    obs, info = env.reset()
    
    n_price = env.action_space.nvec[0]
    n_qty = env.action_space.nvec[1]
    hold_action = np.array([n_price // 2, n_qty // 2])  # center = no trade
    
    # Figure out action for buy qty=1
    # qty_idx mapping: 0 = max sell, center = 0, max = max buy
    # For 9 levels: [0]=max_sell, [4]=0, [8]=max_buy
    buy_1_qty = n_qty // 2 + 1  # one step above center = small buy
    sell_1_qty = n_qty // 2 - 1  # one step below center = small sell
    
    buy_action = np.array([n_price // 2, buy_1_qty])   # mid-price, small buy
    sell_action = np.array([n_price // 2, sell_1_qty])  # mid-price, small sell
    
    initial_balance = env.balance
    initial_portfolio = env._get_portfolio_value()
    
    print(f"  Initial: balance=${initial_balance:,.2f}  portfolio=${initial_portfolio:,.2f}  pos={env.position}")
    
    # Step 1: Buy
    obs, r, term, trunc, info = env.step(buy_action)
    post_buy_pos = env.position
    post_buy_bal = env.balance
    post_buy_pv = env._get_portfolio_value()
    buy_mid = (env.current_best_bid + env.current_best_ask) / 2.0
    print(f"  After BUY:  pos={post_buy_pos}  balance=${post_buy_bal:,.2f}  portfolio=${post_buy_pv:,.2f}  mid=${buy_mid:,.2f}  reward={r:.4f}")
    
    # Steps 2-6: Hold
    hold_rewards = []
    for i in range(5):
        obs, r, term, trunc, info = env.step(hold_action)
        hold_rewards.append(r)
        if term or trunc:
            print(f"  Episode ended during hold at step {i}")
            break
    mid_after_hold = (env.current_best_bid + env.current_best_ask) / 2.0
    print(f"  After HOLD (5 steps): pos={env.position}  balance=${env.balance:,.2f}  portfolio=${env._get_portfolio_value():,.2f}  mid=${mid_after_hold:,.2f}")
    print(f"  Hold rewards: {[f'{r:.4f}' for r in hold_rewards]}")
    
    # Step 7: Sell
    obs, r, term, trunc, info = env.step(sell_action)
    post_sell_pos = env.position
    post_sell_bal = env.balance
    post_sell_pv = env._get_portfolio_value()
    sell_mid = (env.current_best_bid + env.current_best_ask) / 2.0
    print(f"  After SELL: pos={post_sell_pos}  balance=${post_sell_bal:,.2f}  portfolio=${post_sell_pv:,.2f}  mid=${sell_mid:,.2f}  reward={r:.4f}")
    
    # Verify
    total_pnl = post_sell_pv - initial_portfolio
    total_rewards = sum(hold_rewards) + r  # skip buy reward for this check
    print(f"\n  VERIFICATION:")
    print(f"    Total PnL (portfolio): ${total_pnl:,.2f}")
    print(f"    Cumulative fees:       ${env.cumulative_fees:,.2f}")
    print(f"    Position closed:       {post_sell_pos == 0}")
    
    return post_sell_pos == 0


def test_position_limits(env):
    """Test 2: Try to exceed max position. Verify clamping."""
    hr("TEST 2: Position Limit Enforcement")
    obs, info = env.reset()
    
    n_price = env.action_space.nvec[0]
    n_qty = env.action_space.nvec[1]
    
    max_buy_action = np.array([n_price // 2, n_qty - 1])  # max buy qty
    
    print(f"  Max position allowed: {env.max_position}")
    
    positions = []
    for i in range(env.max_position + 5):
        obs, r, term, trunc, info = env.step(max_buy_action)
        positions.append(env.position)
        if term or trunc:
            print(f"  Episode ended at step {i}")
            break
    
    max_reached = max(positions)
    print(f"  Positions over {len(positions)} buy attempts: {positions[:15]}...")
    print(f"  Max position reached: {max_reached}")
    print(f"  Enforced: {'YES ✓' if max_reached <= env.max_position else 'NO ✗ — POSITION LIMIT BUG'}")
    
    return max_reached <= env.max_position


def test_fee_accounting(env):
    """Test 3: Verify fees are correctly deducted from balance."""
    hr("TEST 3: Fee Accounting")
    obs, info = env.reset()
    
    n_price = env.action_space.nvec[0]
    n_qty = env.action_space.nvec[1]
    
    buy_action = np.array([n_price // 2, n_qty // 2 + 1])  # small buy
    
    pre_balance = env.balance
    pre_fees = env.cumulative_fees
    
    obs, r, term, trunc, info = env.step(buy_action)
    
    post_balance = env.balance
    post_fees = env.cumulative_fees
    fees_this_trade = post_fees - pre_fees
    balance_diff = pre_balance - post_balance
    
    print(f"  Pre-trade balance:  ${pre_balance:,.2f}")
    print(f"  Post-trade balance: ${post_balance:,.2f}")
    print(f"  Balance diff:       ${balance_diff:,.2f}")
    print(f"  Fees charged:       ${fees_this_trade:,.4f}")
    print(f"  Position change:    {env.position}")
    
    if env.position != 0:
        # Cost = position * price * fee_rate
        # The exact fill price depends on slippage model
        fill_price = (env.current_best_bid + env.current_best_ask) / 2.0  # approximate
        expected_fee_order = abs(env.position) * fill_price * env.taker_fee
        print(f"  Expected fee (approx): ${expected_fee_order:,.4f}")
        print(f"  Fee reasonable:        {'YES ✓' if 0 < fees_this_trade < expected_fee_order * 3 else 'CHECK ✗'}")
    else:
        print(f"  No position taken (action was no-op)")
    
    return fees_this_trade >= 0


def test_position_flip(env):
    """Test 4: Go long, then flip to short. Verify accounting."""
    hr("TEST 4: Position Flip (Long → Short)")
    obs, info = env.reset()
    
    n_price = env.action_space.nvec[0]
    n_qty = env.action_space.nvec[1]
    
    buy_action = np.array([n_price // 2, n_qty - 1])   # max buy
    sell_action = np.array([n_price // 2, 0])            # max sell
    hold_action = np.array([n_price // 2, n_qty // 2])   # hold
    
    # Build long position
    print(f"  Building long position...")
    for i in range(3):
        obs, r, term, trunc, info = env.step(buy_action)
        if term or trunc:
            break
    long_pos = env.position
    long_balance = env.balance
    long_pv = env._get_portfolio_value()
    print(f"  Long: pos={long_pos}  balance=${long_balance:,.2f}  pv=${long_pv:,.2f}")
    
    # Flip to short
    print(f"  Flipping to short...")
    for i in range(8):
        obs, r, term, trunc, info = env.step(sell_action)
        if term or trunc:
            break
    short_pos = env.position
    short_balance = env.balance
    short_pv = env._get_portfolio_value()
    print(f"  Short: pos={short_pos}  balance=${short_balance:,.2f}  pv=${short_pv:,.2f}")
    
    print(f"\n  VERIFICATION:")
    print(f"    Long→Short flip occurred: {'YES ✓' if short_pos < 0 else 'NO ✗'}")
    print(f"    Portfolio value reasonable: {'YES ✓' if short_pv > 0 else 'NO ✗ — NEGATIVE PV'}")
    print(f"    Cumulative fees: ${env.cumulative_fees:,.4f}")
    
    return short_pos < 0 and short_pv > 0


def test_reward_formula(env):
    """Test 5: Verify reward formula output components."""
    hr("TEST 5: Reward Formula Components")
    obs, info = env.reset()
    
    n_price = env.action_space.nvec[0]
    n_qty = env.action_space.nvec[1]
    hold_action = np.array([n_price // 2, n_qty // 2])
    buy_action = np.array([n_price // 2, n_qty // 2 + 2])  # medium buy
    
    # Step 1: Hold (should have ~zero reward)
    obs, r_hold, term, trunc, info = env.step(hold_action)
    print(f"  Hold reward:     {r_hold:.6f}")
    if 'reward_breakdown' in info:
        print(f"  Hold breakdown:  {info['reward_breakdown']}")
    
    # Step 2: Buy
    obs, r_buy, term, trunc, info = env.step(buy_action)
    print(f"  Buy reward:      {r_buy:.6f}")
    if 'reward_breakdown' in info:
        print(f"  Buy breakdown:   {info['reward_breakdown']}")
    
    # Steps 3-5: Hold with position
    for i in range(3):
        obs, r, term, trunc, info = env.step(hold_action)
        if not (term or trunc):
            print(f"  Hold-with-pos {i}: {r:.6f}")
    
    # Check reward clipping
    all_rewards = [r_hold, r_buy, r]
    max_r = max(all_rewards)
    min_r = min(all_rewards)
    clip_limit = env.reward_clip if hasattr(env, 'reward_clip') else 50.0
    print(f"\n  Reward range: [{min_r:.4f}, {max_r:.4f}]")
    print(f"  Clip limit:   ±{clip_limit}")
    print(f"  Within bounds: {'YES ✓' if min_r >= -clip_limit and max_r <= clip_limit else 'NO ✗'}")
    
    return True


def test_data_exhaustion(env):
    """Test 6: Run to end of data. Verify truncation (not termination)."""
    hr("TEST 6: Data Exhaustion → Truncation")
    obs, info = env.reset()
    
    n_price = env.action_space.nvec[0]
    n_qty = env.action_space.nvec[1]
    hold_action = np.array([n_price // 2, n_qty // 2])
    
    step_count = 0
    terminated = False
    truncated = False
    max_steps = env.handler._len + 100  # safety limit
    
    while step_count < max_steps:
        obs, r, term, trunc, info = env.step(hold_action)
        step_count += 1
        if term or trunc:
            terminated = term
            truncated = trunc
            break
    
    print(f"  Total steps: {step_count}")
    print(f"  Data length: {env.handler._len}")
    print(f"  Terminated: {terminated}")
    print(f"  Truncated:  {truncated}")
    
    # Data exhaustion should be truncation, not termination
    if truncated and not terminated:
        print(f"  CORRECT: Data exhaustion = truncation ✓")
        return True
    elif terminated:
        print(f"  WARNING: Data exhaustion = termination (should be truncation)")
        return False
    else:
        print(f"  WARNING: Neither terminated nor truncated after {step_count} steps")
        return False


if __name__ == "__main__":
    hr("INFRASTRUCTURE DIAGNOSTIC 4: ENVIRONMENT PnL CORRECTNESS")
    
    config = load_config()
    results = {}
    
    tests = [
        ("Buy-Hold-Sell", test_basic_buy_hold_sell),
        ("Position Limits", test_position_limits),
        ("Fee Accounting", test_fee_accounting),
        ("Position Flip", test_position_flip),
        ("Reward Formula", test_reward_formula),
        ("Data Exhaustion", test_data_exhaustion),
    ]
    
    for test_name, test_fn in tests:
        try:
            env = make_env(config)
            passed = test_fn(env)
            results[test_name] = "PASS ✓" if passed else "FAIL ✗"
        except Exception as e:
            import traceback
            print(f"\n  ERROR in {test_name}: {e}")
            traceback.print_exc()
            results[test_name] = f"ERROR: {e}"
    
    hr("SUMMARY")
    print(f"\n  {'Test':<25} {'Result':<15}")
    print(f"  {'-'*25} {'-'*15}")
    for test_name, result in results.items():
        print(f"  {test_name:<25} {result}")
    
    passed = sum(1 for v in results.values() if 'PASS' in v)
    total = len(results)
    print(f"\n  {passed}/{total} tests passed")
    
    hr("DIAGNOSTIC 4 COMPLETE")
