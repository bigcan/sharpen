# Phase 8: Paper Trading & Monitoring

This document describes how to run the Phase 8 Paper Trading Loop.

## Prerequisites

1.  **Alpaca Account:** You need an Alpaca Paper Trading account.
2.  **API Keys:** Set the following environment variables (e.g., in `.env` file):
    ```bash
    ALPACA_API_KEY_ID=your_key_id
    ALPACA_API_SECRET_KEY=your_secret_key
    ```
3.  **MLflow Models:** The script expects the Phase 6 models to be logged in MLflow (Run IDs hardcoded in `finrl_pro_ds/execution/paper_trade.py`).

## Architecture

The paper trading loop (`finrl_pro_ds/execution/paper_trade.py`) performs the following steps:

1.  **Data Fetching:** Downloads the last 300 days of daily bars from Alpaca for the 20-ticker universe.
2.  **Feature Engineering:** Computes the 7 standard tech indicators + turbulence using `ProFeatureAssembler`.
3.  **Regime Detection:** Computes rolling HMM probabilities (Bull/Bear/Sideways) on the market proxy (equal-weight index) with a 3-day hysteresis window.
4.  **Inference:**
    *   Loads the Specialist Agents (Bull, Bear, Sideways).
    *   Computes a **Soft Voting** weighted action based on regime probabilities.
5.  **Risk Management (Circuit Breaker):**
    *   Tracks `peak_equity` in `trading_state.json`.
    *   Calculates current drawdown.
    *   **Halt Condition:** If Drawdown > 15%, closes all positions and stops execution.
6.  **Execution:**
    *   Calculates target portfolio weights (Softmax of action).
    *   Diffs against current Alpaca positions.
    *   Submits Market Orders to rebalance.

## Running the Loop

To start the daily trading loop:

```bash
python scripts/run_paper_trading.py
```

## Monitoring

*   **Logs:** Execution logs are saved to `paper_trading.log` and printed to stdout.
*   **State:** `trading_state.json` persists the peak equity for drawdown calculations.
*   **Alpaca Dashboard:** Monitor open positions and orders on the Alpaca web dashboard.
