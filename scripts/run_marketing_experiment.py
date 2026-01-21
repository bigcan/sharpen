import sys
import os
import pandas as pd
import numpy as np
import time
from stable_baselines3 import PPO

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from finrl_pro_ds.data.feature_factory import MarketingFeatureFactory
from finrl_pro_ds.envs.pro_stock_env import ProStockEnv
from finrl_pro_ds.execution.ensemble import VotingEnsemble

def load_real_data(ticker='AAPL', start_date='2016-01-01', end_date='2020-01-01'):
    """Loads real S&P 500 data for the experiment."""
    file_path = f"data/sp500_multi_2016_2025/{ticker}.parquet"
    print(f"Loading real data from: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
        
    df = pd.read_parquet(file_path)
    
    # Rename timestamp to date if needed
    if 'timestamp' in df.columns:
        df = df.rename(columns={'timestamp': 'date'})
        
    # Ensure standard columns (lowercase)
    df.columns = [c.lower() for c in df.columns]
    
    # Filter Date Range
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
    
    # Add 'tic' if missing
    if 'tic' not in df.columns:
        df['tic'] = ticker
        
    print(f"Loaded {len(df)} rows for {ticker}.")
    return df

def prepare_env_data(df, feature_list, target_col='close'):
    """Prepares numpy arrays for ProStockEnv."""
    # Sort by date
    df = df.sort_values('date')
    
    # Prices
    price_ary = df[[target_col]].values.astype(np.float32)
    
    # Features
    tech_ary = df[feature_list].values.astype(np.float32)
    
    return price_ary, tech_ary

def train_agent(df, features, run_name, seed=42, verbose=True, **ppo_kwargs):

    """Trains a PPO agent using ProStockEnv."""

    if verbose:

        print(f"\n--- Starting Run: {run_name} (Seed {seed}) ---")

        print(f"Features ({len(features)}): {features[:5]}...")

    

    # Data Prep

    price_ary, tech_ary = prepare_env_data(df, features, target_col='close')

    

    # Environment Config

    # Use gamma from ppo_kwargs if available, else default 0.99

    gamma = ppo_kwargs.get('gamma', 0.99)

    

    env_kwargs = {

        "price_ary": price_ary,

        "tech_ary": tech_ary,

        "turbulence_ary": None, 

        "initial_capital": 100000,

        "reward_scaling": 1e-4,

        "gamma": gamma

    }

    

    e_train_gym = ProStockEnv(**env_kwargs)

    

    # Default PPO Params

    params = {

        'learning_rate': 3e-4, 

        'batch_size': 64, 

        'n_steps': 2048,

        'gamma': 0.99,

        'ent_coef': 0.0,

        'clip_range': 0.2

    }

    params.update(ppo_kwargs)

    

    start_time = time.time()

    try:

        agent = PPO("MlpPolicy", e_train_gym, verbose=0, seed=seed, 

                    policy_kwargs=dict(net_arch=[64, 64]), **params)

        agent.learn(total_timesteps=10000) 

        duration = time.time() - start_time

        

        # Eval on Train

        obs, _ = e_train_gym.reset()

        done = False

        while not done:

            action, _ = agent.predict(obs)

            obs, _, done, _, _ = e_train_gym.step(action)

            

        final_value = e_train_gym.total_asset

        if verbose:

            print(f"Training Completed in {duration:.2f} seconds.")

            print(f"Final Train Portfolio Value: ${final_value:,.2f}")

        

        return duration, final_value, agent, e_train_gym

    except Exception as e:

        print(f"Training Crashed: {e}")

        return 0, 0, None, None



def main():

    # 1. Get Real Data

    try:

        raw_df = load_real_data(ticker='AAPL')

    except Exception as e:

        print(f"Error loading data: {e}")

        return



    # HPO Results (Best found)

    OPTIMIZED_PARAMS = {

        'learning_rate': 5.77e-5,

        'batch_size': 64,

        'n_steps': 4096,

        'gamma': 0.933,

        'ent_coef': 0.025,

        'clip_range': 0.19

    }



    # ---------------------------------------------------------

    # Experiment A: Default FinRL Setup

    # ---------------------------------------------------------

    print("\n=== EXPERIMENT A: Default FinRL (StockStats) ===")

    from stockstats import StockDataFrame as Sdf

    

    df_a = raw_df.copy()

    stock = Sdf.retype(df_a.copy())

    tech_indicators = ['macd', 'rsi_30', 'cci_30', 'dx_30'] 

    for ind in tech_indicators:

        _ = stock[ind]

    

    df_a = stock.reset_index() 

    df_a = df_a.fillna(0)

    if 'date' not in df_a.columns:

        df_a['date'] = raw_df['date'].values

        

    time_a, val_a, _, _ = train_agent(df_a, tech_indicators, "Default FinRL")

    

    # ---------------------------------------------------------

    # Experiment B: Hybrid Feature Factory (mRMR + Optimized PPO)

    # ---------------------------------------------------------

    print("\n=== EXPERIMENT B: Hybrid Feature Factory (Optimized) ===")

    

    factory = MarketingFeatureFactory(windows=[14, 30]) 

    df_b = factory.transform(raw_df)

    df_b['date'] = raw_df['date']

    

    stationary_cols = factory.check_stationarity(df_b)

    selected_features = factory.select_features_mrmr(df_b, target_col='close', k=20)

    final_features = [f for f in selected_features if f in stationary_cols]

    if not final_features:

         final_features = stationary_cols[:20]

         

    print(f"Final Feature Count: {len(final_features)}")

    

    time_b, val_b, agent_b, env_b = train_agent(df_b, final_features, "Hybrid Factory", **OPTIMIZED_PARAMS)

    

    # ---------------------------------------------------------

    # Experiment C: Ensemble (3 Seeds, Optimized)

    # ---------------------------------------------------------

    print("\n=== EXPERIMENT C: Ensemble (3 Seeds, Optimized) ===")

    ensemble = VotingEnsemble()

    seeds = [42, 43, 44]

    ensemble_times = []

    

    for seed in seeds:

        # Use same df_b and final_features

        dt, val, ag, _ = train_agent(df_b, final_features, "Ensemble Member", seed=seed, verbose=False, **OPTIMIZED_PARAMS)

        if ag:

            ensemble.add_agent(ag)

            ensemble_times.append(dt)

            print(f"Member (Seed {seed}): ${val:,.2f}")

            

    # Evaluate Ensemble

    print("Evaluating Ensemble...")

    # Re-use env_b (reset it)

    obs, _ = env_b.reset()

    done = False

    start_eval = time.time()

    while not done:

        action, _ = ensemble.predict(obs) # Averaged action

        obs, _, done, _, _ = env_b.step(action)

    eval_time = time.time() - start_eval

    

    val_c = env_b.total_asset

    time_c = sum(ensemble_times) # Total training time

    

    print(f"Ensemble Final Value: ${val_c:,.2f}")

    

    # ---------------------------------------------------------

    # Comparison

    # ---------------------------------------------------------

    print("\n=== RESULTS COMPARISON ===")

    if time_a > 0:

        print(f"Default FinRL:   ${val_a:,.2f} | Time: {time_a:.2f}s")

        print(f"Hybrid Single:   ${val_b:,.2f} | Time: {time_b:.2f}s")

        print(f"Hybrid Ensemble: ${val_c:,.2f} | Time: {time_c:.2f}s (Train) + {eval_time:.2f}s (Eval)")

        

        winner = max([(val_a, "Default"), (val_b, "Single"), (val_c, "Ensemble")], key=lambda x: x[0])

        print(f"\n🏆 WINNER: {winner[1]} (${winner[0]:,.2f})")

    else:

        print("Experiment A failed.")

        

    print("Experiment Complete.")



if __name__ == "__main__":

    main()
