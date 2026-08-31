"""
funding_arb_sanity_test.py — Quick validation that the multi-exchange arb env works correctly.

Imported from https://github.com/bigcan/Funding-Rate-Arb.git
Adapted for FinRL-Pro_DS project structure.

Checks:
  1. Synthetic data shape is 4D
  2. Observation shape matches observation_space
  3. Random rollout completes without error
  4. info["portfolio_value"] present at every step
  5. _get_funding_rate returns float
  6. SB3 check_env passes

Usage:
    python scripts/funding_arb_sanity_test.py
"""

import sys

import numpy as np

from sharpen.crypto.envs.multi_exchange_arb_env import (
    EnvConfig,
    MultiExchangeArbEnv,
)


def run_checks():
    passed = 0
    failed = 0

    def check(name: str, condition: bool, detail: str = ""):
        nonlocal passed, failed
        if condition:
            print(f"  [PASS] {name}")
            passed += 1
        else:
            print(f"  [FAIL] {name} — {detail}")
            failed += 1

    cfg = EnvConfig(max_steps=100)
    env = MultiExchangeArbEnv(config=cfg)

    # 1. Synthetic data shape
    print("\n--- Data checks ---")
    check(
        "market_data is 4D",
        env.market_data.ndim == 4,
        f"got ndim={env.market_data.ndim}",
    )
    expected_shape_suffix = (len(cfg.symbols), len(cfg.exchanges), 7)
    check(
        "market_data shape (sym, ex, feat)",
        env.market_data.shape[1:] == expected_shape_suffix,
        f"got {env.market_data.shape[1:]}",
    )

    # 2. Observation shape
    print("\n--- Observation checks ---")
    obs, info = env.reset()
    check(
        "obs shape matches observation_space",
        obs.shape == env.observation_space.shape,
        f"obs={obs.shape}, space={env.observation_space.shape}",
    )
    check("obs dtype is float32", obs.dtype == np.float32, f"got {obs.dtype}")
    check("no NaN in obs", not np.any(np.isnan(obs)))

    # 3 & 4. Random rollout
    print("\n--- Rollout checks ---")
    check("info has portfolio_value on reset", "portfolio_value" in info)

    done = False
    steps = 0
    pv_present = True
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if "portfolio_value" not in info:
            pv_present = False
        done = terminated or truncated
        steps += 1

    check(f"rollout completed ({steps} steps)", steps > 0)
    check("info['portfolio_value'] present every step", pv_present)
    check(
        "obs shape consistent after rollout",
        obs.shape == env.observation_space.shape,
        f"got {obs.shape}",
    )

    # 5. _get_funding_rate
    print("\n--- API checks ---")
    env.reset()
    fr = env._get_funding_rate(0, 0)
    check("_get_funding_rate returns float", isinstance(fr, float), f"got {type(fr)}")

    # 6. SB3 check_env
    print("\n--- SB3 compatibility ---")
    try:
        from stable_baselines3.common.env_checker import check_env
        check_env(env, warn=True)
        check("SB3 check_env passed", True)
    except Exception as e:
        check("SB3 check_env passed", False, str(e))

    # Summary
    total = passed + failed
    print(f"\n{'=' * 40}")
    print(f"  Results: {passed}/{total} passed, {failed}/{total} failed")
    print(f"{'=' * 40}")
    return failed == 0


if __name__ == "__main__":
    success = run_checks()
    sys.exit(0 if success else 1)
