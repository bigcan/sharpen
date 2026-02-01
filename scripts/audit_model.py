import argparse
import yaml
import os
import torch
import pandas as pd
from typing import Dict, Any

from finrl_pro_ds.training.deepscalper_trainer import DeepScalperTrainer
from finrl_pro_ds.agents.deepscalper.dqn_agent import DeepScalperDQN
from finrl_pro_ds.agents.deepscalper.policy_agents import DeepScalperPPO, DeepScalperA2C
from finrl_pro_ds.agents.deepscalper.ensemble import DeepScalperEnsemble, SynapseGatingNetwork
from finrl_pro_ds.envs.deep_scalper_env import DeepScalperEnv
from finrl_pro_ds.data.parquet_handler import ParquetDataHandler
from finrl_pro_ds.configs.schema import ConfigLoader
from finrl_pro_ds.analytics.vbt_analyzer import VBTAnalyzer
from finrl_pro_ds.reporting.report_generator import AuditReportGenerator
from finrl_pro_ds.governance.research_logger import ResearchLogger
from finrl_pro_ds.governance.leaderboard import Leaderboard

def load_model(config_path, checkpoint_path, device="cpu"):
    # Load Unified Config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
        
    config = ConfigLoader.load_yaml(config_path)
    
    # Init Networks similar to Trainer/HPO
    net_conf = {
        "micro_config": {"input_size": config.network.micro_input_size, "hidden_size": config.network.hidden_size},
        "macro_config": {"input_size": config.network.macro_input_size, "hidden_sizes": [64]}
    }
    
    dqn = DeepScalperDQN(network_config=net_conf, device=device)
    ppo = DeepScalperPPO(net_conf, device=device)
    a2c = DeepScalperA2C(net_conf, device=device)
    
    # Fix: Calculate micro_shape for correct gating init
    window_size = config.env.window_size
    micro_feat_dim = config.network.micro_input_size
    micro_shape = (window_size, micro_feat_dim)
    gating = SynapseGatingNetwork(input_dim=config.network.macro_input_size, micro_shape=micro_shape)
    
    # Load Weights
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    dqn.policy_net.load_state_dict(checkpoint['dqn'])
    ppo.network.load_state_dict(checkpoint['ppo'])
    a2c.network.load_state_dict(checkpoint['a2c'])
    gating.load_state_dict(checkpoint['gating'])
    
    ensemble = DeepScalperEnsemble(dqn, ppo, a2c, gating, device=device)
    return ensemble, config

def run_backtest(ensemble, config, start_date, end_date):
    # Setup Data
    handler = ParquetDataHandler(
        file_path=config.data.file_path,
        ticker=config.data.ticker,
        feature_config=config.features.__dict__,
        start_date=start_date,
        end_date=end_date
    )
    
    # Env for Backtesting (Single Threaded, Deterministic)
    # We use vector environment wrapper even for single to match trainer logic if deeply integrated,
    # but simpler to use raw env if supported.
    # Trainer expects vector or single.
    
    env_conf = config.env.__dict__
    env_conf['reward'] = config.env.reward.__dict__
    env_conf['action'] = config.env.action.__dict__
    
    env = DeepScalperEnv(config=env_conf, data_handler=handler)
    
    # We need to run episode step-by-step and collect actions/rewards
    # Actually, we need executed size and price for VBT
    # DeepScalperEnv usually simulates fill.
    # We need to capture:
    # 1. Close Price Series (from data handler)
    # 2. Executed Size Series (from env actions or info)
    # 3. Executed Price Series (from env info)
    
    obs, info = env.reset()
    done = False
    
    data_log = []
    
    # Prepare Trainer wrapper for logic reuse (get_probs etc)
    # BUT we don't need update logic. Just inference.
    
    while not done:
        # Inference
        # Unpack obs manually matching Trainer._unpack_obs
        # Micro: (1, W, F), Private: (1, W, F), Macro: (1, F)
        # Assuming single env
        
        # We need a proper inference hook or reuse Trainer.evaluate logic
        # But Trainer.evaluate only returns aggregate metrics.
        # We want granular logs.
        
        micro = torch.tensor(obs['micro'], dtype=torch.float32).unsqueeze(0).to(ensemble.device)
        private = torch.tensor(obs['private'], dtype=torch.float32).unsqueeze(0).to(ensemble.device)
        macro = torch.tensor(obs['macro'], dtype=torch.float32).unsqueeze(0).to(ensemble.device)
        
        with torch.no_grad():
            actions, _ = ensemble.predict(micro, private, macro, deterministic=True) # Returns (1, 3), weights
            
        # Step
        next_obs, reward, terminated, truncated, info = env.step(actions[0])
        done = terminated or truncated
        
        # Log Data for VBT
        # Info contains: timestamp, current_price, executed_size, executed_price, etc?
        # Standard Env info usually has minimal data.
        # We need to rely on DataHandler indices or Env internal state?
        # Let's assume Env returns data in Info for now, or we access handler directly.
        # The 'timestamp' is in obs usually or info.
        
        log_entry = {
            "timestamp": info.get("timestamp", pd.Timestamp.now()), # Fallback
            "close": info.get("data_close", 0.0), # Need Env to provide this
            "action_type": action[0][0], # 0=Hold, 1=Buy, 2=Sell ? Depends on mapping
            "size": info.get("executed_size", 0.0), # Signed
            "price": info.get("executed_price", 0.0)
        }
        data_log.append(log_entry)
        
        obs = next_obs

    df_log = pd.DataFrame(data_log)
    if "timestamp" in df_log.columns:
        df_log.set_index("timestamp", inplace=True)
        
    return df_log

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--start_date", default="2023-02-01")
    parser.add_argument("--end_date", default="2023-02-28")
    parser.add_argument("--output", default="reports/audit")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ensemble, config = load_model(args.config, args.checkpoint, device)
    
    print(f"Running Backtest from {args.start_date} to {args.end_date}...")
    df_results = run_backtest(ensemble, config, args.start_date, args.end_date)
    
    print("Analyzing Metrics...")
    analyzer = VBTAnalyzer(
        close=df_results['close'],
        size=df_results['size'],
        price=df_results['price'],
        init_cash=config.env.initial_balance
    )
    
    metrics = analyzer.get_audit_metrics()
    
    # Thresholds (Institutional)
    thresholds = {
        "sharpe_ratio": 1.5,
        "max_drawdown": -20.0, # %, e.g. -20%
        "total_return": 5.0 # Min 5% return
    }
    
    print("Generating Report...")
    generator = AuditReportGenerator(output_dir=args.output)
    
    # Save Plot
    plot_path = os.path.join(args.output, "equity_curve.html")
    analyzer.plot(path=plot_path)
    
    report_path = generator.generate_report(
        model_name=os.path.basename(args.checkpoint),
        metrics=metrics,
        plots_paths=[plot_path],
        thresholds=thresholds
    )
    
    print(f"Audit Complete. Report saved to: {report_path}")

    # Governance Automation
    print("Updating Governance Logs...")
    logger = ResearchLogger()
    leaderboard = Leaderboard()
    
    # Check Pass/Fail via Generator Helper or custom logic
    # We reused threshold dict
    passed = generator._check_pass(metrics, thresholds) 
    status = "PASSED" if passed else "FAILED"
    
    logger.log_experiment(
        experiment_name=os.path.basename(args.checkpoint),
        config_summary={"config": args.config, "checkpoint": args.checkpoint},
        metrics=metrics,
        artifacts={"Report": report_path, "Equity Curve": plot_path},
        status=status
    )
    
    rank = leaderboard.add_entry(
        model_identifier=os.path.basename(args.checkpoint),
        metrics=metrics,
        config_hash=args.config
    )
    print(f"Leaderboard Rank: {rank}")

if __name__ == "__main__":
    main()
