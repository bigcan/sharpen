# Synapse Crypto 1H — Implementation Plan

**Strategy ID**: `sync-1H`
**Version**: 0.3.0 (Audited)
**Date**: 2026-03-10
**Base**: Synapse V9.5 (Production)
**Target**: Cryptocurrency perpetual futures, long/short, 1-hour timeframe
**Branch**: `feat/synapse-crypto-1h`

---

## 1. Executive Summary

Adapt the production Synapse V9.5 multi-agent DRL ensemble (SAC + A2C + PPO_GAE with Softmax Arbitrator) from **US equities / daily** to **cryptocurrency perpetual futures / 1-hour**. The strategy trades **both long and short** positions with **no leverage** (max 1x net exposure). The core architecture (Lean Trinity agents, probabilistic arbitrator, walk-forward evaluation) remains intact. Major changes target data ingestion, feature engineering cadence, environment dynamics (perpetual futures mechanics), and exchange execution.

### Key Design Decisions

- **Instrument**: Perpetual futures (not spot) — enables short positions, funding rate dynamics
- **Directionality**: Long and short positions enabled; agents output signed weights
- **Leverage**: None (1x max). Gross exposure ≤ 100% of margin balance at all times
- **Risk management**: Full module implemented and ready, but **disabled during backtesting** to measure raw strategy alpha. Enabled for paper/live trading
- **Paper trading**: Deployed via **Bybit Testnet** (native perpetual futures support)

### What Changes vs What Stays

| Layer | Status | Notes |
|:------|:-------|:------|
| Agent algorithms (SAC, A2C, PPO) | **Keep** | Architecture unchanged; hyperparameters re-tuned |
| Arbitrator (Softmax weighted) | **Keep** | Rolling window shortened from days to hours |
| Walk-Forward evaluation | **Keep** | Window sizes re-scaled to hourly bars |
| AutoScaler V2 | **Keep** | Hardware detection unchanged |
| Data pipeline | **Rewrite** | CCXT/exchange APIs replace yfinance |
| Feature engineering | **Adapt** | Same families, different windows (252d → 168h etc.) |
| Environment | **Rewrite** | Perpetual futures: long/short, funding rates, 24/7 |
| Risk controls | **Adapt** | Full module ready, disabled for backtest, enabled for live |
| Execution / Broker | **Rewrite** | Bybit (CCXT) for paper/live trading; Binance for data |
| Regime detection | **Adapt** | Faster HMM stride, shorter CUSUM/KAMA windows |

---

## 2. Asset Universe

### 2.1 Core Universe (Top 20 by Liquidity)

```
BTC, ETH, SOL, BNB, XRP, ADA, AVAX, DOGE, DOT, LINK,
POL, UNI, ATOM, LTC, FIL, APT, ARB, OP, NEAR, INJ
```

**Selection criteria**:
- 24h volume > $50M on primary exchange
- Listed on ≥ 2 major CEXs (Binance, OKX, Bybit)
- Sufficient history (≥ 2 years of 1H data)
- Exclude stablecoins, wrapped tokens, memecoins (except DOGE for liquidity)

### 2.2 Quote Currency

- **USDT** pairs as primary (deepest liquidity)
- Fallback to **USDC** pairs if USDT unavailable
- ~~BUSD~~ — fully sunset by Binance (Dec 2023), not available

### 2.3 Instrument Type

**Perpetual Futures (Linear, USDT-margined)**

| Property | Value |
|:---------|:------|
| Contract type | Linear perpetual (USDT-settled) |
| Margin mode | Cross margin |
| Leverage | 1x (no leverage) |
| Funding interval | Every 8 hours |
| Position direction | Long and short |
| Settlement | No expiry — perpetual |

### 2.4 Exchange Priority

| Exchange | Role | Reason |
|:---------|:-----|:-------|
| **Binance Futures** | Primary data source | Deepest perp liquidity, best 1H OHLCV history |
| **OKX** | Secondary / validation | Cross-exchange data validation |
| **Bybit** | Paper + live execution | Native USDT perp support, testnet for paper trading, CCXT-compatible |

> **Note**: Alpaca Crypto was considered but **does not support perpetual futures** (spot only). Bybit provides native perp trading with a full-featured testnet for paper trading.

---

## 3. Data Pipeline

### 3.1 Data Source

Replace `yfinance` with **CCXT** (unified crypto exchange library).

```
New module: finrl_pro/data/crypto_loader.py
```

| Parameter | Equities (Current) | Crypto (New) |
|:----------|:-------------------|:-------------|
| **Library** | yfinance | ccxt (async) |
| **Frequency** | Daily | 1-Hour |
| **Market** | Spot equities | Perpetual futures (USDT-margined) |
| **Fields** | OHLCV | OHLCV + funding_rate + open_interest + liquidations |
| **History** | 2010–2025 (~15yr) | 2022–2026 (~4yr, exchange-dependent) |
| **Market Hours** | 6.5h/day, 252 days/yr | 24/7/365 = 8,760 bars/yr |
| **Rate Limits** | Minimal | Respect exchange limits (1200 req/min Binance) |

### 3.2 Data Architecture (Medallion)

```
Bronze: Raw CCXT OHLCV + funding rates (Parquet, partitioned by symbol)
Silver: Cleaned (Hampel filter, gap-fill for exchange outages, volume spike filter)
Gold:   Feature-engineered, normalized, ready for training
```

**Crypto-specific cleaning**:
- Exchange maintenance gap detection and forward-fill (max 4h gap)
- Flash crash filter: **adaptive threshold** — flag candles where |return| > 10σ of rolling 720-bar realized vol (per asset). Hard cap at 15% for BTC/ETH, 25% for altcoins. Flag, don't drop
- Volume anomaly detection: z-score > 10 on volume → cap at 10σ
- Funding rate interpolation for missing intervals

### 3.3 Macro / Cross-Asset Features

| Feature | Source | Transform | Publication Lag |
|:--------|:-------|:----------|:----------------|
| `btc_dominance` | CoinGecko / on-chain | BTC market cap / total crypto cap | Use T-1h value |
| `total_market_cap_ffd` | CoinGecko | FFD on log(total_cap) | Use T-1h value |
| `funding_rate` | Exchange API | Per-asset, raw (already normalized) | Real-time (no lag) |
| `open_interest_change` | Exchange API | 1h pct change in OI | Use T-1h value |
| `fear_greed_index` | Alternative.me API | Daily, forward-filled to 1H | Use T-24h value (published daily) |
| `dxy_ffd` | Yahoo Finance (^DXY) | FFD on log(DXY), resampled to 1H | Use T-1h value (market hours only) |

> **Look-Ahead Bias Prevention**: All external features use lagged values matching their real-world publication delay. Features are never available at time T if they weren't published by time T in production. This is enforced programmatically in the feature pipeline — each external feature has a registered `publication_lag_bars` parameter.

---

## 4. Feature Engineering

### 4.1 Window Rescaling

The current system uses `window=252` (≈1 trading year of daily bars). For 1H crypto:

| Concept | Daily Equivalent | 1H Equivalent | Rationale |
|:--------|:----------------|:---------------|:----------|
| 1 trading year | 252 bars | 8,760 bars | Too large; use 168 (1 week) as "short" |
| 1 month | 21 bars | 720 bars | ~30 days of hourly bars |
| Short-term | 14 bars | 168 bars (1 week) | Intra-week patterns |
| Medium-term | 30 bars | 720 bars (1 month) | Monthly cycles |
| Long-term | 252 bars | 2,160 bars (3 months) | Quarterly trends |

### 4.2 Technical Indicators (15 Features — Same Families)

| Group | Indicators | Window |
|:------|:-----------|:-------|
| **Trend** | macd(12,26,9), dx_168, trix_168, cci_168 | 168h (1 week) |
| **Momentum** | rsi_168, wr_168, kdjk_168, roc_168 | 168h |
| **Volatility** | atr_168, close_168_mstd, boll_ub, boll_lb | 168h |
| **Volume** | mfi_168, vr_168, obv | 168h |

### 4.3 Derived Features (Same as V9.5)

| Feature | Adaptation |
|:--------|:-----------|
| Log Returns (5) | Unchanged — `log(close/prev_close)` for OHLCV |
| ATR Normalization | `atr_168 / close` |
| Signed ADX | `dx_168 × sign(macd)` |
| Bollinger Width | `(boll_ub - boll_lb) / close` |
| OBV FFD | Same transform, shorter auto-tune window |

### 4.4 Ensemble Features (4 Consensus Signals)

Same Z-score averaging approach, using 168h rolling Z-score window (was 252d).

### 4.5 Crypto-Specific Features (8 New Features)

| Feature | Description | Window |
|:--------|:------------|:-------|
| `funding_rate` | Perpetual swap funding rate | Raw (8h intervals) |
| `oi_change_pct` | Open interest % change | 1h |
| `btc_correlation` | Rolling correlation with BTC | 168h |
| `volume_profile_skew` | Buy vol / total vol ratio | 24h |
| `liquidation_intensity` | Normalized liquidation volume | 24h |
| `exchange_netflow` | Net exchange inflow/outflow (if available) | 24h |
| `btc_dominance_regime` | BTC dominance regime indicator: >55% = BTC season, <40% = alt season peak, between = neutral | Rolling 168h |
| `cost_to_rebalance` | Estimated transaction cost if current position were fully rebalanced to new target | Per-step (derived) |

### 4.6 Total Feature Count Per Asset

| Category | Count |
|:---------|:------|
| Technical indicators | 15 |
| Log returns | 5 |
| Macro/cross-asset | 6 |
| Ensemble signals | 4 |
| Regime features | 4 |
| Entropy features | 2 |
| Crypto-specific | 8 |
| **Total** | **44** |

### 4.7 Normalization

| Feature Type | Method | Parameters |
|:-------------|:-------|:-----------|
| Indicators | Rolling Z-score | window=720 (1 month), clip=(-5, 5) |
| Log returns | Hard clip | (-0.5, 0.5) |
| FFD features | Rolling Z-score post-FFD | window=720 |
| Entropy | Rolling MinMax | window=720 |
| Funding rate | Hard clip | (-0.01, 0.01) |
| BTC correlation | Raw (already bounded [-1, 1]) | — |
| BTC dominance regime | Categorical encoding (0, 0.5, 1) | — |
| Cost to rebalance | Rolling MinMax | window=720 |

---

## 5. Environment Adaptations

### 5.1 Market Dynamics Differences

| Property | Equities (Current) | Crypto Perps 1H (New) |
|:---------|:-------------------|:----------------------|
| Market hours | 6.5h/day, 5d/week | 24/7/365 |
| Instrument | Spot equities | USDT-margined perpetual futures |
| Avg daily vol | 1-2% | 3-8% (per hour: 0.3-1%) |
| Transaction cost | 10 bps (0.1%) | 2 bps maker / 5 bps taker (futures) |
| Slippage model | Fixed 5 bps | Volume-dependent: `3 + 15 × (order_size / hourly_volume)` bps |
| Short selling | Not supported | **Enabled** — native perpetual futures short |
| Leverage | 1.0x | **1.0x (no leverage)** — gross exposure ≤ 100% |
| Funding costs | N/A | Every 8h: long pays short (or vice versa) |
| Margin | N/A | Cross margin, USDT collateral |
| Corporate actions | Dividends, splits | Airdrops, forks, delistings (handled as events) |

### 5.2 Perpetual Futures Environment

```
New module: finrl_pro/envs/crypto_perp_env.py
```

This is the most significant architectural change from the equities system. The environment models perpetual futures mechanics natively.

**Key design**:
- `tech_dim = 44` (was 32 for equities)
- **Signed positions**: each asset has a position in `[-max_weight, +max_weight]`
  - Positive weight = long position
  - Negative weight = short position
  - Zero = flat (no position)
- **No leverage**: `sum(|weights|) ≤ 1.0` enforced (gross exposure ≤ 100% of margin)
- **Funding rate**: applied at **UTC-aligned times** (00:00, 08:00, 16:00 UTC) matching exchange mechanics, not on a fixed modular bar count. The environment maps each bar's timestamp to determine funding application
  - Long pays funding when rate > 0; short receives
  - Short pays funding when rate < 0; long receives
- **Mark-to-market PnL**: positions valued at mark price (≈ last traded price for backtesting)
- **Margin balance**: `initial_capital + realized_PnL + unrealized_PnL - cumulative_fees - cumulative_funding`
- Transaction cost: maker/taker fee applied to notional value of position changes
- Slippage: volume-dependent model (backtest); L2 book depth model (live — see §10.5)
- No market close/open gaps — continuous 24/7 trading

**Position sizing constraint** (no leverage — constraint-aware policy):

The agents use **tanh output activation** in the policy network's final layer, producing raw actions in `(-1, 1)`. The no-leverage constraint is then enforced while preserving relative conviction:

```
For raw policy outputs a = [a_1, ..., a_20] where a_i ∈ (-1, 1) via tanh:

    # Proportional scaling (preserves relative magnitudes and signs)
    gross = sum(|a_i|)
    If gross > 1.0:
        a_i = a_i × (1.0 / gross)  # Scale proportionally

    # The agents are TRAINED with this constraint active (not applied post-hoc)
    # so they learn to self-allocate within the exposure budget.

    Net exposure = sum(a_i)           # Can range from -1.0 to +1.0
    Gross exposure = sum(|a_i|) ≤ 1.0  # Always ≤ 100% (no leverage)
```

> **Design rationale**: Unlike naive L1 norm clipping (which uniformly scales all positions and destroys conviction signals), the constraint is baked into the environment's `step()` function during training. Agents learn to produce actions that naturally respect the budget, producing higher-quality allocations.

### 5.3 Reward Function

Keep **Sortino-focused Softmax reward** (production default), but adjust:
- Downside deviation window: 168 bars (1 week) instead of 21 bars (1 month in daily)
- Turnover penalty coefficient: increase from 0.001 to 0.002 (discourage over-trading on 1H)
- Funding cost is a **real P&L component** (not a penalty) — already reflected in margin balance via UTC-aligned application
- Short positions generate positive returns when asset price decreases

### 5.4 Action Space

| Parameter | Value |
|:----------|:------|
| Space | Continuous [-1, 1] per asset |
| Semantics | **Negative = short, Positive = long, Zero = flat** |
| Assets | 20 |
| Normalization | **Tanh + proportional scaling** — constraint-aware training preserves conviction and sign |
| Gross exposure | ≤ 100% (no leverage): `sum(\|weights\|) ≤ 1.0` |
| Net exposure | Unconstrained within [-1.0, +1.0] (can be net short) |
| Rebalance frequency | Every 1 hour (each step) |

### 5.5 Observation Space

```
State vector per step:
    [margin_balance_pct]                           # 1 value
    + [position_weight_per_asset × n_features]     # 20 × 44 = 880 values
    + [current_position_per_asset]                  # 20 values (signed: +long, -short)
    + [unrealized_pnl_per_asset]                    # 20 values
    + [funding_rate_per_asset]                      # 20 values
    + [cost_to_rebalance_per_asset]                 # 20 values (estimated txn cost of full rebalance)
    + [portfolio_concentration]                     # 1 value (effective number of bets — see §9.4)

Total observation dim: 1 + 880 + 20 + 20 + 20 + 20 + 1 = 962
```

---

## 6. Agent Configuration

### 6.1 Lean Trinity — Hyperparameter Starting Points

All three agents keep their architecture but need re-tuning for 1H crypto dynamics.

#### SAC (Soft Actor-Critic)

| Parameter | Equities V9.5 | Crypto 1H (Initial) | Rationale |
|:----------|:-------------|:---------------------|:----------|
| learning_rate | 3e-4 | 1e-4 | More conservative for volatile market |
| buffer_size | 1M | 500K | Sufficient with 8,760 bars/yr |
| batch_size | 256 | 512 | More data per update for noisy signals |
| tau | 0.005 | 0.005 | Keep |
| gamma | 0.99 | 0.98 | Effective horizon ~50 bars (2 days). 0.995 was too long — noisy 1H signals cause slow credit assignment |
| network | [1024, 1024, 512] | [1024, 1024, 512] | Keep |
| ent_coef | auto | auto | Keep |

#### A2C (Advantage Actor-Critic)

| Parameter | Equities V9.5 | Crypto 1H (Initial) | Rationale |
|:----------|:-------------|:---------------------|:----------|
| learning_rate | 7e-4 | 3e-4 | |
| n_steps | 5 | 24 | 1 day of hourly bars; n_steps=8 produced very short rollouts insufficient for A2C advantage estimation |
| gamma | 0.99 | 0.98 | Matches SAC/PPO; effective horizon ~50 bars |
| network | [1024, 1024, 512] | [1024, 1024, 512] | |

#### PPO (with GAE)

| Parameter | Equities V9.5 | Crypto 1H (Initial) | Rationale |
|:----------|:-------------|:---------------------|:----------|
| learning_rate | 3e-4 | 1e-4 | |
| n_steps | 2048 | 4096 | |
| batch_size | 64 | 128 | |
| n_epochs | 10 | 10 | |
| gamma | 0.99 | 0.98 | Matches SAC/A2C; effective horizon ~50 bars |
| gae_lambda | 0.95 | 0.95 | |
| clip_range | 0.2 | 0.15 | |
| network | [1024, 1024, 512] | [1024, 1024, 512] | |

### 6.2 Training Configuration

| Parameter | Equities V9.5 | Crypto 1H |
|:----------|:-------------|:----------|
| Total timesteps | 1M | 2M |
| Vectorized envs (N_ENVS) | 20 | 20 |
| Mixed precision | FP16 | FP16 |
| torch.compile | Yes | Yes |
| Checkpoint frequency | 100K steps | 200K steps |

---

## 7. Walk-Forward Configuration

### 7.1 Window Sizing

Hourly bars accumulate fast: 1 year ≈ 8,760 bars vs 252 daily bars.

| Window | Equities V9.5 | Crypto 1H | Bar Count | Rationale |
|:-------|:-------------|:----------|:----------|:----------|
| **Train** | 3 years | **12 months** | **~8,760 bars** | 962-dim state space needs sufficient data; 6mo was too thin |
| **Validation** | 6 months | 1 month | ~720 bars | |
| **Test (OOS)** | 12 months | 2 months | ~1,440 bars | |
| **Step** | 6 months | 1 month | ~720 bars | |
| **Embargo** | 5 days (FFD) | **1 month (FFD)** | **720 bars** | Must match normalization window (720 bars) to prevent serial correlation leakage through rolling Z-scores |

### 7.2 Expected Windows

With ~4 years of data (2022-01 to 2026-03):
- Total span: ~35,000 bars
- Train+Val+Test+Embargo per window: 8,760 + 720 + 1,440 + 720 = ~11,640 bars
- Step size: ~720 bars
- **~20-25 walk-forward windows** (good statistical power; quality per window prioritized over quantity)

### 7.3 Validation Gates (Risk Summary Gate)

| Metric | Threshold | Notes |
|:-------|:----------|:------|
| Sharpe ratio (OOS) | > 0.5 | Lower bar than equities due to higher vol |
| Max drawdown (OOS) | < 25% | Higher tolerance for crypto |
| Sortino ratio (OOS) | > 0.7 | |
| Turnover | < 200% / month | Prevent churn |
| Win rate | > 45% | |

---

## 8. Arbitrator Configuration

### 8.1 Softmax Arbitrator (Primary)

| Parameter | Equities V9.5 | Crypto 1H |
|:----------|:-------------|:----------|
| Lookback window | 63 bars (3 months daily) | 504 bars (3 weeks hourly) |
| Temperature | 2.0 | 1.5 (sharper selection in volatile regime) |
| Performance metric | Sortino | Sortino |
| Min weight | 0.1 | 0.15 (more balanced in uncertain market) |
| Reweight frequency | Daily | Every 24 bars (daily) |

### 8.2 Emergency Reweight Trigger

The 24-bar reweight cycle can leave stale weights during fast regime changes. An **emergency reweight** fires immediately when:
- Any agent's rolling 72-bar Sortino drops below 0.0 (agent is actively losing)
- The spread between best and worst agent Sortino exceeds 2.0 over the last 72 bars
- A circuit breaker event occurs (paper/live mode)

This prevents a hemorrhaging agent from dragging the ensemble for up to 24 hours.

### 8.3 Fallback: InvVar Arbitrator

Same inverse-variance weighting, using 504-bar rolling window.

---

## 9. Risk Management

> **IMPORTANT**: The risk management module is **fully implemented and ready** but **disabled during backtesting** to measure raw strategy alpha without interference. It is **enabled for paper trading and live trading**.

### 9.1 Risk Controls

| Control | Value | Backtest | Paper/Live |
|:--------|:------|:---------|:-----------|
| Max drawdown (circuit breaker) | 20% | **OFF** | ON |
| Max gross exposure | 100% (no leverage) | **Enforced always** | Enforced always |
| Max position per asset | 15% gross | **OFF** | ON |
| Max net short exposure | -50% | **OFF** | ON |
| Daily turnover limit | 150% | **OFF** | ON |
| Min margin reserve | 5% | **OFF** | ON |
| Circuit breaker cooldown | 48 hours | **OFF** | ON |
| Transaction cost budget | 20 bps/day | **OFF** | ON |
| Funding rate alert | \|rate\| > 0.1% per 8h | **OFF** | ON |
| Max cluster exposure | 30% gross in correlated cluster (ρ > 0.7) | **OFF** | ON |
| Min effective bets (ENB) | ENB > 4 (1/HHI of position weights) | **OFF** | ON |

**Implementation pattern**:
```python
class RiskManager:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled  # False for backtest, True for paper/live

    def check(self, action, state) -> Action:
        if not self.enabled:
            return action  # Pass-through in backtest mode
        # ... full risk checks ...
```

The **no-leverage constraint** (`sum(|weights|) ≤ 1.0`) is the one exception — it is **always enforced**, even during backtesting, as it is a structural property of the strategy, not a risk overlay.

### 9.2 Crypto-Specific Risk Events (Paper/Live Only)

| Event | Detection | Action |
|:------|:----------|:-------|
| Exchange outage | No new candle for > 2h | Pause trading, alert |
| Flash crash | > 15% drop in 1h across 5+ assets | Circuit breaker → flatten all |
| Depegging (stablecoins) | USDT/USD deviation > 0.5% | Halt all trades, alert |
| Regulatory event | Manual flag | Flatten to cash |
| Funding rate spike | \|funding\| > 0.1% per 8h | Reduce exposed positions |
| Liquidation cascade | OI drops > 20% in 1h | Reduce gross exposure to 50% |

### 9.3 Perpetual Futures Risk Considerations

| Risk | Description | Mitigation |
|:-----|:------------|:-----------|
| **Funding drag** | Persistent long bias when funding > 0 erodes returns | Agent learns funding cost via P&L; can go short to collect funding |
| **Short squeeze** | Rapid price increase forces short covering | No leverage limits max loss to position size |
| **Basis risk** | Perp price diverges from spot | Mark price used for P&L, not last traded |
| **Counterparty risk** | Exchange insolvency | Bybit Proof-of-Reserves monitoring; keep only trading capital on exchange; multi-exchange fallback (OKX) |
| **Auto-deleveraging** | Exchange ADL in extreme moves | No leverage = lowest ADL priority |

### 9.4 Correlation & Concentration Risk

Crypto assets are highly correlated (especially during drawdowns). The portfolio tracks:

| Metric | Description | Action |
|:-------|:------------|:-------|
| **Effective Number of Bets (ENB)** | `1 / HHI(position_weights)` — measures portfolio diversification | Included in observation space; alert when ENB < 4 |
| **Correlation clustering** | Rolling 168-bar pairwise correlation matrix → hierarchical clustering (ρ > 0.7 threshold) | In paper/live: max 30% gross exposure per cluster |
| **Directional concentration** | Track net long/short by correlation cluster | Alert when >60% of gross exposure is in a single directional cluster |

The ENB value is fed into the observation space (§5.5) so agents learn to diversify. The cluster-based exposure limits are enforced by the risk manager in paper/live mode only.

---

## 10. Execution Layer

### 10.1 Architecture Overview

Two execution paths:
1. **Backtesting**: Simulated execution within the environment (no broker needed)
2. **Paper/Live trading**: **Bybit** via CCXT (native perpetual futures support)

Data sourcing uses CCXT (Binance Futures) for historical data and features. Execution uses Bybit (testnet for paper, mainnet for live).

> **Why not Alpaca?** Alpaca Crypto supports **spot trading only** — no perpetual futures, no native short selling via perps, no funding rate mechanics. Since the entire strategy is built on perpetual futures, a futures-native exchange is required.

### 10.2 Bybit Broker (Paper + Live Trading)

```
New module: finrl_pro/execution/bybit_perp_broker.py
```

Bybit provides native USDT perpetual futures with a full-featured testnet for paper trading:

| Feature | Implementation |
|:--------|:---------------|
| API | Bybit V5 API via CCXT (unified interface) |
| Paper trading | Bybit Testnet (testnet.bybit.com) — full perp simulation |
| Order types | Limit (default), market (fallback after 60s) |
| Position direction | Long and short via `side` + `positionIdx` |
| Order sizing | Respect Bybit min qty, tick size, and lot size rules |
| Position tracking | Bybit positions API + local reconciliation |
| Fee model | Bybit futures: 1 bps maker / 5.5 bps taker |
| WebSocket | Bybit V5 WebSocket for real-time prices, fills, positions |
| Authentication | API key + secret, stored via python-dotenv, never committed |
| Rate limits | 120 req/s (order), 10 req/s (position query) — enforced by CCXT |

### 10.3 Execution Flow

```
Arbitrator Decision (every 1H)
    → Target Signed Weights (20 assets, each in [-max, +max])
    → Tanh + Proportional Scaling (ensure gross exposure ≤ 100%)
    → Delta Calculation (target - current positions, preserving sign)
    → Risk Manager Check (if enabled: paper/live mode)
    → Concentration Check (ENB, cluster exposure — if enabled)
    → Filter: skip if |delta| < 0.5% of portfolio (avoid dust trades)
    → Order Generation:
        - Positive delta → buy / cover short
        - Negative delta → sell / open short
    → Execution (Bybit V5 API via CCXT)
    → Fill Monitoring (WebSocket)
    → Position Reconciliation (Bybit positions API vs local state)
    → Funding Rate Tracking (Bybit API — real funding, not simulated)
    → Log to MLflow
```

### 10.4 Backtesting Execution (Environment-Internal)

During backtesting, execution is fully simulated within `crypto_perp_env.py`:
- Fills at next bar's open price + slippage model
- Maker/taker fees applied to notional change
- Funding rates applied at **UTC-aligned timestamps** (00:00, 08:00, 16:00) from historical data — not on a fixed modular bar count
- No broker API calls — pure simulation
- Risk manager **disabled** — raw strategy performance measured

### 10.5 Live Slippage Model

For backtesting, the volume-dependent model (`3 + 15 × (size/volume)` bps) is used. For **paper and live trading**, an enhanced model incorporating L2 order book depth is used:

```
slippage_bps = base_bps + impact_bps × (order_notional / available_book_depth_at_price)
```

Where `available_book_depth_at_price` is the cumulative book depth within 10 bps of mid price, fetched via Bybit WebSocket. This provides more accurate cost estimation for illiquid altcoins during volatility spikes.

---

## 11. Infrastructure

### 11.1 Data Collection Service

```
New module: finrl_pro/data/crypto_collector.py
```

Continuous background service:
- Fetch 1H candles every hour via CCXT
- Store in Parquet (Bronze layer), append-only
- Backfill on startup if gaps detected
- Funding rate collection every 8 hours
- Open interest snapshots every 1 hour

### 11.2 Deployment

| Component | Target |
|:----------|:-------|
| Training | GPUHub / RunPod (existing infra) |
| Data collector | Always-on VPS (lightweight, no GPU) |
| Live execution | Co-located VPS near Bybit servers (Singapore/Tokyo) for low latency |
| Monitoring | MLflow + Grafana dashboard |

### 11.3 Docker

```
New file: docker/synapse_crypto_1h.Dockerfile
```

Extends `synapse_v9.Dockerfile` with:
- CCXT library
- WebSocket dependencies
- Timezone set to UTC
- Cron job for data collection

---

## 12. Implementation Phases

### Phase 1: Data & Features (Week 1-2)

| # | Task | Files | Priority |
|:--|:-----|:------|:---------|
| 1.1 | Create `crypto_loader.py` with CCXT integration | `finrl_pro/data/crypto_loader.py` | P0 |
| 1.2 | Implement Bronze/Silver/Gold pipeline for crypto | `finrl_pro/data/crypto_loader.py` | P0 |
| 1.3 | Backfill 4 years of 1H data for 20 assets | Script | P0 |
| 1.4 | Adapt feature windows (252→168) | `finrl_pro/data/feature_factory.py` | P0 |
| 1.5 | Add crypto-specific features (funding, OI, etc.) | `finrl_pro/features/crypto_features.py` | P1 |
| 1.6 | Adapt regime detection for 1H | `finrl_pro/data/regime_features.py` | P1 |
| 1.7 | Data validation & quality checks | Tests | P0 |

### Phase 2: Perpetual Futures Environment & Training (Week 2-3)

| # | Task | Files | Priority |
|:--|:-----|:------|:---------|
| 2.1 | Create `crypto_perp_env.py` with long/short, funding rates | `finrl_pro/envs/crypto_perp_env.py` | P0 |
| 2.2 | Implement L1 norm action clipping (no leverage) | `finrl_pro/envs/crypto_perp_env.py` | P0 |
| 2.3 | Implement volume-dependent slippage model | `finrl_pro/envs/crypto_perp_env.py` | P0 |
| 2.4 | Implement funding rate mechanics (every 8 bars) | `finrl_pro/envs/crypto_perp_env.py` | P0 |
| 2.5 | Implement risk manager (ready but disabled for backtest) | `finrl_pro/mlops/crypto_risk_manager.py` | P0 |
| 2.6 | Create crypto experiment config YAML | `finrl_pro/configs/experiments/synapse_crypto_1h.yaml` | P0 |
| 2.7 | Adapt walk-forward windows for 1H | Config | P0 |
| 2.8 | Initial training run (single agent, single window) | Script | P0 |
| 2.9 | Full Lean Trinity training | Script | P0 |

### Phase 3: Evaluation & Tuning (Week 3-4)

| # | Task | Files | Priority |
|:--|:-----|:------|:---------|
| 3.1 | Walk-forward evaluation across all windows | `finrl_pro/eval/walk_forward.py` | P0 |
| 3.2 | Arbitrator calibration for 1H cadence | `finrl_pro/execution/arbitrator.py` | P0 |
| 3.3 | HPO with Optuna (50 trials per agent) | `finrl_pro/automl/walk_forward.py` | P1 |
| 3.4 | Risk gate validation | `finrl_pro/eval/risk_summary_gate.py` | P0 |
| 3.5 | Benchmark comparison (BTC Buy&Hold, Equal Weight) | `finrl_pro/agents/baselines.py` | P0 |
| 3.6 | Performance reporting (Pyfolio/VBT) | `finrl_pro/analytics/` | P1 |

### Phase 4: Bybit Paper Trading & Deployment (Week 4-5)

| # | Task | Files | Priority |
|:--|:-----|:------|:---------|
| 4.1 | Create `bybit_perp_broker.py` via CCXT | `finrl_pro/execution/bybit_perp_broker.py` | P0 |
| 4.2 | Enable risk manager for paper trading mode | `finrl_pro/mlops/crypto_risk_manager.py` | P0 |
| 4.3 | Data collector service (CCXT → Parquet) | `finrl_pro/data/crypto_collector.py` | P0 |
| 4.4 | Docker image for crypto | `docker/synapse_crypto_1h.Dockerfile` | P1 |
| 4.5 | Deploy to Bybit Testnet paper trading | — | P0 |
| 4.6 | Paper trading validation (2+ weeks) | — | P0 |
| 4.7 | Monitoring, alerting, & model staleness detection setup | `finrl_pro/mlops/monitoring.py` | P1 |

### Phase 5: Live Trading (Week 6+)

| # | Task | Priority |
|:--|:-----|:---------|
| 5.1 | Deploy to Bybit mainnet with 10% of target capital | P0 |
| 5.2 | Monitor for 2 weeks, compare to paper results | P0 |
| 5.3 | Scale to full capital if metrics pass gates | P0 |
| 5.4 | Multi-exchange execution (OKX, Binance where permitted) | P2 |
| 5.5 | Model staleness monitor — auto-retrain when rolling 720-bar OOS Sharpe < 0.5 | P1 |

---

## 13. Benchmarks

### 13.1 Strategy Benchmarks

| Benchmark | Description |
|:----------|:------------|
| **BTC Buy & Hold** | 100% BTC allocation |
| **Equal Weight** | 1/N allocation across 20 assets, monthly rebalance |
| **Momentum (Top 5)** | Top 5 by 7-day return, equal weight, hourly rebalance |
| **Market Cap Weighted** | Weight by market cap, monthly rebalance |

### 13.2 Target Metrics (OOS)

| Metric | Target | Stretch |
|:-------|:-------|:--------|
| Sharpe ratio | > 1.0 | > 1.5 |
| Sortino ratio | > 1.5 | > 2.0 |
| Max drawdown | < 25% | < 15% |
| CAGR | > 30% | > 50% |
| Win rate (hourly) | > 48% | > 52% |
| Turnover | < 200%/mo | < 100%/mo |
| Beat BTC B&H | Yes | Yes with lower vol |

---

## 14. Key Risks & Mitigations

| Risk | Impact | Mitigation |
|:-----|:-------|:-----------|
| Exchange API instability | Missed trades, stale data | Multi-exchange fallback, local cache |
| Crypto regime shift (regulatory) | Model breakdown | Circuit breaker, manual override |
| Overfitting to 4yr history | Poor OOS performance | Walk-forward with 30+ windows, DSR validation |
| High transaction costs | Alpha erosion | Turnover penalty in reward, maker-only orders |
| Liquidity drying up (alt season) | Slippage | Volume-dependent position limits |
| Data quality (exchange manipulation) | Biased features | Cross-exchange validation, outlier filtering |
| Model latency > 1H | Missed rebalance | Pre-compute next actions, timeout fallback to hold |
| Concentration risk | Correlated drawdown across similar assets | ENB monitoring, cluster exposure limits (§9.4) |
| Model staleness | Regime shift degrades trained policy | Rolling OOS Sharpe monitor, auto-retrain trigger (§5.5) |
| Look-ahead bias | Inflated backtest results from future data leakage | Publication lag enforcement on all external features (§3.3) |

---

## 15. Configuration Template

```yaml
# synapse_crypto_1h.yaml
strategy:
  name: "synapse_crypto_1h"
  version: "1.0.0"
  base: "synapse_v9.5"
  instrument: "perpetual_futures"   # perpetual futures, not spot

universe:
  assets: ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "AVAX", "DOGE",
           "DOT", "LINK", "POL", "UNI", "ATOM", "LTC", "FIL", "APT",
           "ARB", "OP", "NEAR", "INJ"]
  quote: "USDT"
  data_exchange: "binance"          # CCXT source for historical data
  execution_broker: "bybit"         # Bybit for paper (testnet) / live trading
  n_assets: 20

data:
  source: "ccxt"
  market_type: "futures"            # Binance Futures (not spot)
  frequency: "1h"
  start_date: "2022-01-01"
  end_date: null  # rolling
  fields: ["open", "high", "low", "close", "volume"]
  extra_fields: ["funding_rate", "open_interest", "liquidations"]
  cache_dir: "./data/crypto_cache"

features:
  tech_window: 168       # 1 week in hours
  norm_window: 720       # 1 month in hours
  norm_clip: [-5, 5]
  log_return_clip: [-0.5, 0.5]
  tech_dim: 44
  enable_hmm: true
  hmm_stride: 5
  enable_crypto_features: true

environment:
  type: "perpetual_futures"
  initial_capital: 100000  # USDT margin balance
  # --- Perpetual futures settings ---
  long_short_enabled: true
  max_leverage: 1.0                 # NO leverage — gross exposure ≤ 100%
  margin_mode: "cross"
  funding_alignment: "utc"           # Funding at 00:00, 08:00, 16:00 UTC (not modular bar count)
  # --- Costs ---
  maker_fee_pct: 0.0002             # 2 bps maker
  taker_fee_pct: 0.0005             # 5 bps taker
  slippage_model: "volume_dependent"
  slippage_base_bps: 3
  slippage_impact_bps: 15
  # --- Constraints ---
  max_position_pct: 0.15            # Max 15% per asset (gross)
  max_gross_exposure: 1.0           # No leverage
  max_net_short_exposure: -0.50     # Max 50% net short
  # --- Reward ---
  turnover_penalty: 0.002
  reward_type: "sortino_softmax"

  # --- Risk management ---
  risk_manager:
    enabled_backtest: false          # DISABLED for backtesting
    enabled_paper: true              # ENABLED for paper trading
    enabled_live: true               # ENABLED for live trading

agents:
  lean_trinity: ["sac", "a2c", "ppo_gae"]
  total_timesteps: 2000000
  n_envs: 20
  use_fp16: true
  use_torch_compile: true
  network_arch: [1024, 1024, 512]

  sac:
    learning_rate: 1.0e-4
    buffer_size: 500000
    batch_size: 512
    gamma: 0.98
    tau: 0.005

  a2c:
    learning_rate: 3.0e-4
    n_steps: 24                       # 1 day of hourly bars
    gamma: 0.98

  ppo_gae:
    learning_rate: 1.0e-4
    n_steps: 4096
    batch_size: 128
    n_epochs: 10
    gamma: 0.98
    gae_lambda: 0.95
    clip_range: 0.15

arbitrator:
  method: "softmax"
  lookback_bars: 504      # 3 weeks
  temperature: 1.5
  min_weight: 0.15
  reweight_every: 24      # bars (daily)
  performance_metric: "sortino"
  emergency_reweight:
    enabled: true
    sortino_floor: 0.0              # Reweight immediately if any agent drops below this
    sortino_spread_trigger: 2.0     # Reweight if best-worst agent spread exceeds this
    emergency_lookback_bars: 72     # 3-day window for emergency checks

walk_forward:
  train_bars: 8760        # 12 months
  val_bars: 720           # 1 month
  test_bars: 1440         # 2 months
  step_bars: 720          # 1 month
  embargo_bars: 720       # 1 month (matches normalization window to prevent serial correlation leakage)
  n_hpo_trials: 50
  hpo_timesteps: 200000

risk:
  # All controls below are DISABLED during backtesting (risk_manager.enabled_backtest: false)
  # Only the no-leverage constraint (max_gross_exposure: 1.0) is always enforced
  max_drawdown_pct: 0.20
  circuit_breaker_cooldown_bars: 48  # 2 days
  daily_turnover_limit: 1.50
  daily_cost_budget_bps: 20
  flash_crash_threshold: 0.15
  flash_crash_min_assets: 5
  funding_rate_alert: 0.001
  max_cluster_exposure: 0.30        # Max 30% gross in correlated cluster (ρ > 0.7)
  min_effective_bets: 4             # Min ENB (1/HHI)
  correlation_window: 168           # Rolling window for pairwise correlation
  correlation_cluster_threshold: 0.7

execution:
  broker: "bybit"                    # Bybit for paper (testnet) / live
  data_source: "ccxt"                # CCXT (Binance) for historical data
  order_type: "limit"
  limit_offset_pct: 0.0005           # 0.05% inside spread
  market_fallback_timeout: 60        # seconds
  min_trade_pct: 0.005               # skip if |delta| < 0.5%
  reconciliation_interval: 300       # seconds
  paper_trade: true
  bybit_testnet: true                # Use testnet for paper trading
  slippage_model_live: "book_depth"  # L2 order book model for paper/live

model_staleness:
  enabled: true
  monitor_window_bars: 720           # Rolling 1-month OOS performance
  retrain_trigger_sharpe: 0.5        # Auto-retrain when rolling Sharpe drops below this
  retrain_cooldown_bars: 2160        # Min 3 months between retrains

deployment:
  training_target: "gpuhub"
  collector_target: "vps"
  execution_target: "vps"
  docker_image: "synapse_crypto_1h:latest"
```

---

## 16. File Structure (New/Modified)

```
finrl_pro/
├── data/
│   ├── crypto_loader.py          # NEW — CCXT data ingestion (Binance Futures)
│   ├── crypto_collector.py       # NEW — Background data collection service
│   ├── feature_factory.py        # MODIFIED — Configurable windows
│   └── regime_features.py        # MODIFIED — Shorter windows for 1H
├── features/
│   └── crypto_features.py        # NEW — Funding rate, OI, BTC correlation
├── envs/
│   └── crypto_perp_env.py        # NEW — Perpetual futures env (long/short, funding)
├── execution/
│   └── bybit_perp_broker.py      # NEW — Bybit perpetual futures broker via CCXT
├── mlops/
│   └── crypto_risk_manager.py    # NEW — Risk module (ready, disabled for backtest)
├── configs/
│   └── experiments/
│       └── synapse_crypto_1h.yaml # NEW — Full experiment config
├── analytics/
│   └── crypto_report.py          # NEW — Crypto-specific tearsheet additions
docker/
│   └── synapse_crypto_1h.Dockerfile  # NEW
scripts/
│   └── crypto_backtest_runner.py     # NEW — Adapted backtest orchestrator
experiments/
│   └── Synapse_Crypto/               # NEW — Experiment results directory
```

---

## 17. Dependencies (New)

```
ccxt>=4.0.0          # Unified crypto exchange API (data + execution)
websockets>=12.0     # Exchange WebSocket feeds
aiohttp>=3.9.0       # Async HTTP for data collection
python-dotenv>=1.0   # API key management (never committed to git)
pybit>=5.0.0         # Bybit V5 API client (optional, CCXT primary)
```

---

## 18. Success Criteria

| Gate | Criteria | Required |
|:-----|:---------|:---------|
| **Data Quality** | 4yr 1H data for 20 assets, <0.1% missing bars | Yes |
| **Training Convergence** | All 3 agents converge (reward improving over training) | Yes |
| **Walk-Forward OOS Sharpe** | Median Sharpe > 1.0 across all ~20-25 windows | Yes |
| **Beat BTC B&H** | Risk-adjusted returns (Sharpe) better than BTC B&H | Yes |
| **Drawdown Control** | No OOS window with drawdown > 25% | Yes |
| **Paper Trade Match** | Bybit Testnet paper trade results within 10% of backtest | Yes |
| **Model Freshness** | Rolling 720-bar OOS Sharpe > 0.5 (auto-retrain trigger) | Yes |
| **Latency** | Full inference + order execution < 30 seconds | Yes |

---

*This document will be updated as implementation progresses.*
