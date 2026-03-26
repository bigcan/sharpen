import logging
import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf

import wandb

logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class WandbFinRLEvaluator:
    """
    A production-grade evaluator for FinRL Ensemble Trading Systems.
    Handles performance metrics, ensemble analysis, and W&B logging.
    """

    def __init__(
        self,
        df_ensemble: pd.DataFrame,
        dict_agents: dict[str, pd.DataFrame],
        benchmark_ticker: str = "BTC-USD",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        """
        Initialize the evaluator.

        Args:
            df_ensemble: DataFrame with 'date', 'account_value', 'actions' (optional).
            dict_agents: Dictionary of DataFrames for each sub-agent.
            benchmark_ticker: Ticker symbol for benchmark comparison.
        """
        self.df_ensemble = df_ensemble.copy()
        self.dict_agents = {k: v.copy() for k, v in dict_agents.items()}
        self.benchmark_ticker = benchmark_ticker

        # Infer dates if not provided
        if start_date is None:
            self.start_date = self.df_ensemble['date'].min()
        else:
            self.start_date = start_date

        if end_date is None:
            self.end_date = self.df_ensemble['date'].max()
        else:
            self.end_date = end_date

        self.results = {} # Store calculated metrics
        self.daily_returns = pd.DataFrame() # Store aligned daily returns

        # Preprocess Data Immediately
        self._preprocess_data()

    def _preprocess_data(self):
        """
        Clean, align, and calculate returns for all agents and benchmark.
        Handles timezone mismatches and NaNs.
        """
        logger.info("Preprocessing data...")

        # 1. Standardize Dates
        def process_df(df):
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').set_index('date')
            # Handle potential timezone issues by localizing to None or UTC
            if df.index.tz is not None:
                df.index = df.index.tz_convert(None)
            return df

        self.df_ensemble = process_df(self.df_ensemble)
        for name, df in self.dict_agents.items():
            self.dict_agents[name] = process_df(df)

        # 2. Fetch Benchmark Data
        logger.info(f"Fetching benchmark data for {self.benchmark_ticker}...")
        try:
            # Fix: Ensure dates are converted to datetime objects and ADD BUFFER
            # yfinance sometimes fails if start==end or for intraday limits
            start_dt = pd.to_datetime(self.start_date) - pd.Timedelta(days=2)
            end_dt = pd.to_datetime(self.end_date) + pd.Timedelta(days=2)

            df_bench = yf.download(
                self.benchmark_ticker,
                start=start_dt.strftime('%Y-%m-%d'),
                end=end_dt.strftime('%Y-%m-%d'),
                progress=False,
            )

            # Handle yfinance 0.2+ multi-index columns (Ticker, Price)
            if isinstance(df_bench.columns, pd.MultiIndex):
                try:
                    df_bench = df_bench.xs(self.benchmark_ticker, axis=1, level=1)
                except KeyError:
                    # Fallback if structure is different
                    if 'Close' in df_bench.columns.get_level_values(0):
                         df_bench = df_bench['Close']

            # Flatten if still needed
            if isinstance(df_bench, pd.DataFrame):
                if 'Close' in df_bench.columns:
                     df_bench = df_bench[['Close']]
                else:
                     # Taking first column as close
                     df_bench = df_bench.iloc[:, 0].to_frame(name='Close')

            # Rename to standard column
            df_bench.columns = ['Close']

            if df_bench.empty:
                raise ValueError("Downloaded benchmark data is empty.")

            if df_bench.index.tz is not None:
                df_bench.index = df_bench.index.tz_convert(None)

            self.df_benchmark = df_bench
        except Exception as e:
            logger.warning(f"Failed to fetch benchmark data '{self.benchmark_ticker}': {e}. Using flat zero-return benchmark.")
            # Create dummy benchmark matching the ensemble index
            self.df_benchmark = pd.DataFrame({'Close': [100.0] * len(self.df_ensemble)}, index=self.df_ensemble.index)

        # 3. Align and Calculate Daily Returns
        # We assume 'account_value' exists.

        # Ensemble Returns
        ensemble_ret = self.df_ensemble['account_value'].pct_change().dropna()
        self.daily_returns['Ensemble'] = ensemble_ret

        # Agent Returns
        for name, df in self.dict_agents.items():
            agent_ret = df['account_value'].pct_change().dropna()
            self.daily_returns[name] = agent_ret

        # Benchmark Returns
        bench_ret = self.df_benchmark['Close'].pct_change().dropna()
        self.daily_returns['Benchmark'] = bench_ret

        # Fill NaNs with 0 for alignment (or forward fill depending on strictness, but 0 is safer for "no trade")
        self.daily_returns = self.daily_returns.fillna(0)

        # Align indexes to intersection to be fair
        # common_index = self.daily_returns.dropna().index
        # self.daily_returns = self.daily_returns.loc[common_index]

    def calculate_metrics(self, df: pd.DataFrame, name: str) -> dict:
        """
        Calculate generic financial metrics for a given DataFrame (Agent or Ensemble).
        """
        # Ensure daily returns exist
        if 'daily_return' not in df.columns:
            df['daily_return'] = df['account_value'].pct_change().fillna(0)

        returns = df['daily_return']

        # 1. Risk-Adjusted Returns
        # FIX WB-03: Use 525600 (minute-level crypto) not 252 (daily equity)
        ANN_FACTOR = np.sqrt(525600)  # minutes per year: 365.25 * 24 * 60
        mean_ret = returns.mean()
        std_ret = returns.std()
        sharpe = (mean_ret / std_ret * ANN_FACTOR) if (std_ret > 1e-9 and not np.isnan(std_ret)) else 0.0

        # FIX WB-02: Correct Sortino formula — sqrt(mean(min(r,0)²)), not std(negative_returns)
        # FIX XMATH-11: Use ddof=1 (sample) to match Sharpe (.std() uses ddof=1)
        # and all other modules (statistics.py, crypto_perp_env.py, etc.).
        downside_returns = returns.values
        downside_sq = np.minimum(downside_returns, 0.0) ** 2
        n_ds = len(downside_sq)
        downside_dev = np.sqrt(np.sum(downside_sq) / max(n_ds - 1, 1))
        sortino = (mean_ret / downside_dev * ANN_FACTOR) if downside_dev > 1e-9 else 0.0

        # Calmar Ratio
        # Need Cum Sum for MDD
        cum_ret = (1 + returns).cumprod()
        running_max = cum_ret.cummax()
        drawdown = (cum_ret - running_max) / running_max
        max_drawdown = drawdown.min() # Negative number

        # Geometric annualized return: (final/initial)^(365/days) - 1
        days = (df.index[-1] - df.index[0]).days
        if days > 0:
            total_ret_geo = (df['account_value'].iloc[-1] / df['account_value'].iloc[0])
            annualized_return_geo = (total_ret_geo ** (365/days)) - 1
        else:
            annualized_return_geo = 0

        if max_drawdown == 0:
            calmar = 0.0
        else:
            calmar = (annualized_return_geo / abs(max_drawdown)) if (not np.isnan(max_drawdown) and abs(max_drawdown) > 1e-9) else 0.0

        # 2. Risk Metrics
        annualized_vol = std_ret * ANN_FACTOR  # FIX WB-03: minute-level

        # 3. Trade Stats
        total_return_pct = (df['account_value'].iloc[-1] / df['account_value'].iloc[0]) - 1

        # Need to know actions for Wind Rate / Profit Factor
        # Assuming 'actions' column exists. If it's pure account value, we can't infer trade accuracy easily
        # without transaction logs. But if 'actions' is present (shares bought/sold).

        # Trade Counts (require 'actions' column)
        total_trades = 0
        long_trades = 0
        short_trades = 0

        if 'actions' in df.columns:
            actions = df['actions'].fillna(0)
            total_trades = int((actions != 0).sum())
            long_trades = int((actions > 0).sum())
            short_trades = int((actions < 0).sum())

        # Win Rate & Profit Factor (based on daily returns, always calculated)
        winning_days = returns[returns > 0]
        losing_days = returns[returns < 0]

        total_days = len(returns)
        win_rate = (len(winning_days) / total_days) if total_days > 0 else 0.0

        gross_profit = winning_days.sum()
        gross_loss = abs(losing_days.sum())
        # FIX WB-01: Cap profit_factor at 10.0 instead of inf (breaks WandB logging)
        profit_factor = (gross_profit / gross_loss) if gross_loss > 1e-12 else (10.0 if gross_profit > 1e-12 else 0.0)

        metrics = {
            "Agent": name,
            "Sharpe_Ratio": sharpe,
            "Sortino_Ratio": sortino,
            "Calmar_Ratio": calmar,
            "Max_Drawdown": max_drawdown,
            "Annual_Volatility": annualized_vol,
            "Total_Return": total_return_pct,
            "Win_Rate_Daily": win_rate,
            "Profit_Factor_Daily": profit_factor,
            "Total_Action_Count": total_trades,
            "Long_Action_Count": long_trades,
            "Short_Action_Count": short_trades,
        }

        return metrics

    def calculate_diversity(self):
        """
        Compute correlation matrix and Ensemble Lift.
        """
        # Correlation Matrix
        corr_matrix = self.daily_returns.corr(method='pearson')

        # Ensemble Lift
        best_single_agent_sharpe = -float('inf')

        # We need to ensure we have metrics calculated first
        if not self.results:
            self.compute_all_metrics()

        for name, metrics in self.results.items():
            if name != "Ensemble" and name != "Benchmark":
                if metrics['Sharpe_Ratio'] > best_single_agent_sharpe:
                    best_single_agent_sharpe = metrics['Sharpe_Ratio']

        ensemble_sharpe = self.results.get("Ensemble", {}).get("Sharpe_Ratio", 0)
        ensemble_lift = ensemble_sharpe - best_single_agent_sharpe

        return corr_matrix, ensemble_lift

    def _build_trade_log(self, df: pd.DataFrame, agent_name: str) -> pd.DataFrame:
        """
        Build a trade log with per-trade details and P&L.

        Args:
            df: DataFrame with 'date', 'price', 'quantity' columns.
            agent_name: Name of the agent for identification.

        Returns:
            DataFrame with trade details, or empty DataFrame if required columns missing.
        """
        # Alias 'actions' to 'quantity' if needed
        if 'quantity' not in df.columns and 'actions' in df.columns:
            df['quantity'] = df['actions']

        # Check for required columns (graceful degradation)
        if 'price' not in df.columns or 'quantity' not in df.columns:
            return pd.DataFrame()

        # Filter to actual trades (non-zero quantity)
        trades = df[df['quantity'] != 0].copy()

        if trades.empty:
            return pd.DataFrame()

        # Direction
        trades['direction'] = np.where(trades['quantity'] > 0, 'BUY', 'SELL')

        # Notional value
        trades['notional'] = trades['price'] * trades['quantity'].abs()

        # Per-trade P&L (simple: quantity * price change to next period)
        # Using shift to get next-period price difference
        trades['pnl'] = trades['quantity'] * trades['price'].diff().shift(-1).fillna(0)

        # Agent identification
        trades['agent'] = agent_name

        # Ticker (optional column)
        if 'ticker' not in trades.columns:
            trades['ticker'] = 'N/A'

        # Reset index to get date as column
        trades = trades.reset_index()

        # Select and order columns
        output_cols = ['date', 'agent', 'ticker', 'direction', 'price', 'quantity', 'notional', 'pnl']
        available_cols = [c for c in output_cols if c in trades.columns]

        return trades[available_cols]

    def compute_all_metrics(self):
        """
        Compute metrics for Ensemble and all Sub-Agents.
        """
        # Ensemble
        self.results["Ensemble"] = self.calculate_metrics(self.df_ensemble, "Ensemble")

        # Agents
        for name, df in self.dict_agents.items():
            self.results[name] = self.calculate_metrics(df, name)

    def log_to_wandb(self, run_name: str, project_name: str = "finrl-ensemble", entity: Optional[str] = None):
        """
        Log all results to Weights & Biases.
        """
        # Check if run exists
        should_finish = False
        if wandb.run is None:
            # Initialize W&B only if not active
            run = wandb.init(project=project_name, name=run_name, entity=entity, reinit=True)
            should_finish = True
        else:
            # Use existing run
            run = wandb.run
            logger.info(f"Logging metrics to active W&B run: {run.name}")

        try:
            # Ensure metrics are ready
            if not self.results:
                self.compute_all_metrics()

            corr_matrix, ensemble_lift = self.calculate_diversity()

            # 1. Leaderboard Table
            # Convert results dict to list of dicts
            data_list = list(self.results.values())
            # Add Ensemble Lift to Ensemble row
            for row in data_list:
                if row['Agent'] == "Ensemble":
                    row['Ensemble_Alpha'] = ensemble_lift
                else:
                    row['Ensemble_Alpha'] = 0.0 # N/A

            leaderboard_df = pd.DataFrame(data_list)
            leaderboard_table = wandb.Table(dataframe=leaderboard_df)
            wandb.log({"Leaderboard": leaderboard_table})

            # 2. Equity Curve Overlay (Matplotlib -> wandb.Image)
            cum_returns = (1 + self.daily_returns).cumprod()

            fig, ax = plt.subplots(figsize=(12, 6))
            for col in cum_returns.columns:
                ax.plot(cum_returns.index, cum_returns[col], label=col)
            ax.set_title("Cumulative Returns Comparison")
            ax.set_xlabel("Date")
            ax.set_ylabel("Cumulative Return")
            ax.legend()
            ax.grid(True)
            wandb.log({"Equity Curve": wandb.Image(fig)})
            plt.close(fig)

            # 3. Correlation Matrix Heatmap (Seaborn)
            # FIXED: Use explicit figure/ax instead of global state
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            ax.set_title("Agent Correlation Matrix")
            wandb.log({"Correlation Matrix": wandb.Image(fig)})
            plt.close(fig)

            # 4. Activity Heatmap (Vectorized approach)
            action_dfs = []

            if 'actions' in self.df_ensemble.columns:
                ens_actions = self.df_ensemble[['actions']].copy()
                ens_actions['Agent'] = 'Ensemble'
                ens_actions = ens_actions.reset_index().rename(columns={'actions': 'Action', 'date': 'Date'})
                action_dfs.append(ens_actions)

            for name, df in self.dict_agents.items():
                if 'actions' in df.columns:
                    agent_actions = df[['actions']].copy()
                    agent_actions['Agent'] = name
                    agent_actions = agent_actions.reset_index().rename(columns={'actions': 'Action', 'date': 'Date'})
                    action_dfs.append(agent_actions)

            if action_dfs:
                act_df = pd.concat(action_dfs, ignore_index=True)
                act_pivot = act_df.pivot_table(index='Agent', columns='Date', values='Action', aggfunc='sum')

                fig, ax = plt.subplots(figsize=(14, 6))
                sns.heatmap(act_pivot, cmap="vlag", center=0, cbar_kws={'label': 'Action Magnitude'}, ax=ax)
                ax.set_title("Trading Activity Heatmap")
                wandb.log({"Activity Heatmap": wandb.Image(fig)})
                plt.close(fig)

            # 5. Underwater Plot (Ensemble Only)
            # Drawdown over time
            ensemble_ret = self.daily_returns['Ensemble']
            cum = (1 + ensemble_ret).cumprod()
            running_max = cum.cummax()
            dd = (cum - running_max) / running_max

            # FIXED: Use explicit figure/ax instead of global state
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.fill_between(dd.index, dd, 0, color='red', alpha=0.3)
            ax.plot(dd.index, dd, color='red', linewidth=1)
            ax.set_title("Ensemble Underwater Plot")
            ax.set_ylabel("Drawdown")
            ax.grid(True)
            wandb.log({"Underwater Plot": wandb.Image(fig)})
            plt.close(fig)

            # 6. Gating Weights Plot (Ensemble Only)
            if all(col in self.df_ensemble.columns for col in ['weight_dqn', 'weight_ppo', 'weight_a2c']):
                fig, ax = plt.subplots(figsize=(12, 4))

                # Ensure index is datetime for nice plotting
                # self.df_ensemble index should be 'date' if preprocess worked well.
                # But preprocess set index to date.
                # So we can use index.

                y1 = self.df_ensemble['weight_dqn']
                y2 = self.df_ensemble['weight_ppo']
                y3 = self.df_ensemble['weight_a2c']

                # Stackplot
                ax.stackplot(self.df_ensemble.index, y1, y2, y3, labels=['DQN', 'PPO', 'A2C'], alpha=0.8)
                ax.set_title("Ensemble Gating Weights Over Time")
                ax.set_ylabel("Weight Assignment")
                ax.legend(loc='upper left')
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 1.0)

                wandb.log({"Gating Weights": wandb.Image(fig)})
                plt.close(fig)

            # 7. Trade Log Table (Detailed trade activities)
            trade_logs = []

            # Build trade logs for ensemble
            ens_trades = self._build_trade_log(self.df_ensemble, "Ensemble")
            if not ens_trades.empty:
                trade_logs.append(ens_trades)

            # Build trade logs for each agent
            for name, df in self.dict_agents.items():
                agent_trades = self._build_trade_log(df, name)
                if not agent_trades.empty:
                    trade_logs.append(agent_trades)

            if trade_logs:
                combined_log = pd.concat(trade_logs, ignore_index=True)
                trade_log_table = wandb.Table(dataframe=combined_log)
                wandb.log({"Trade Log": trade_log_table})
                logger.info(f"Logged {len(combined_log)} trade records to W&B.")
            else:
                logger.info("No detailed trade data available (missing price/quantity columns).")

            # 8. Account Balance Time-Series (USDT tracking)
            # Log balance progression for Ensemble and all agents
            balance_data = []

            # Ensemble balance
            ens_balance = self.df_ensemble[['account_value']].copy()
            ens_balance = ens_balance.reset_index()
            ens_balance['agent'] = 'Ensemble'
            ens_balance = ens_balance.rename(columns={'account_value': 'balance_usdt'})
            balance_data.append(ens_balance)

            # Agent balances
            for name, df in self.dict_agents.items():
                agent_balance = df[['account_value']].copy()
                agent_balance = agent_balance.reset_index()
                agent_balance['agent'] = name
                agent_balance = agent_balance.rename(columns={'account_value': 'balance_usdt'})
                balance_data.append(agent_balance)

            if balance_data:
                combined_balance = pd.concat(balance_data, ignore_index=True)
                balance_table = wandb.Table(dataframe=combined_balance)
                wandb.log({"Account Balance": balance_table})

                # Also log initial and final balance to summary
                initial_balance = self.df_ensemble['account_value'].iloc[0]
                final_balance = self.df_ensemble['account_value'].iloc[-1]
                wandb.run.summary["Initial_Balance_USDT"] = initial_balance
                wandb.run.summary["Final_Balance_USDT"] = final_balance
                wandb.run.summary["Absolute_PnL_USDT"] = final_balance - initial_balance
                logger.info(f"Balance: {initial_balance:,.2f} USDT → {final_balance:,.2f} USDT (P&L: {final_balance - initial_balance:+,.2f})")

            # 6. Summary Attributes
            ens_metrics = self.results.get("Ensemble", {})
            wandb.run.summary["Ensemble_Sharpe"] = ens_metrics.get("Sharpe_Ratio", 0)
            wandb.run.summary["Ensemble_Sortino"] = ens_metrics.get("Sortino_Ratio", 0)
            wandb.run.summary["Ensemble_Total_Return"] = ens_metrics.get("Total_Return", 0)

            logger.info(f"Results logged to W&B run: {run.name}")

        finally:
            if should_finish:
                run.finish()

def generate_wandb_report(
    df_ensemble: pd.DataFrame,
    dict_agents: dict[str, pd.DataFrame],
    run_name: str = "ensemble-eval-run",
    project_name: str = "finrl-ensemble",
    benchmark_ticker: str = "BTC-USD",
    entity: Optional[str] = None,
):
    """
    Helper function to instantiate and run the evaluator.

    Args:
        df_ensemble: DataFrame with 'date', 'account_value', 'actions' columns.
        dict_agents: Dictionary of DataFrames for each sub-agent.
        run_name: Name for the W&B run.
        project_name: W&B project name.
        benchmark_ticker: Ticker symbol for benchmark (default: BTC-USD).
        entity: W&B entity (team or username). Optional.

    Returns:
        WandbFinRLEvaluator: The evaluator instance with computed metrics.
    """
    evaluator = WandbFinRLEvaluator(
        df_ensemble=df_ensemble,
        dict_agents=dict_agents,
        benchmark_ticker=benchmark_ticker,
    )
    evaluator.log_to_wandb(run_name=run_name, project_name=project_name, entity=entity)
    return evaluator
