import argparse
import sys
import os
import subprocess
from finrl_pro_ds.configs.schema import ConfigLoader
from finrl_pro_ds.ops.deploy import deploy_to_remote

def run_training(args):
    """Execution wrapper for training script"""
    print(f"🚀 Starting DeepScalper Training [Config: {args.config}]")
    cmd = [sys.executable, "scripts/train_deepscalper.py", "--config", args.config]
    if args.debug:
        cmd.append("--debug")
    subprocess.check_call(cmd)

def run_tuning(args):
    """Execution wrapper for HPO tuning"""
    print(f"🔧 Starting Hyperparameter Optimization [Trials: {args.trials}]")
    cmd = [
        sys.executable, "scripts/tune_deepscalper.py", 
        "--config", args.config,
        "--trials", str(args.trials),
        "--steps", str(args.steps)
    ]
    if args.resume:
        cmd.append("--resume")
    subprocess.check_call(cmd)

def run_audit(args):
    """Execution wrapper for auditing"""
    print(f"eag🔍 Starting Institutional Audit [Model: {args.checkpoint}]")
    cmd = [
        sys.executable, "scripts/audit_model.py",
        "--config", args.config,
        "--checkpoint", args.checkpoint,
        "--output", args.output
    ]
    subprocess.check_call(cmd)

def run_deploy(args):
    """Deployment wrapper"""
    print(f"☁️ Deploying to {args.target}...")
    deploy_to_remote(target=args.target, config_path=args.config, dry_run=args.dry_run)

def main():
    parser = argparse.ArgumentParser(prog="finrl_pro_ds", description="DeepScalper MLOps CLI")
    subparsers = parser.add_subparsers(dest="command", help="Pipeline Stages")
    
    # Train
    train_parser = subparsers.add_parser("train", help="Train DeepScalper Model")
    train_parser.add_argument("--config", default="configs/deepscalper_unified.yaml", help="Path to config")
    train_parser.add_argument("--debug", action="store_true", help="Run in debug mode (mock env)")
    
    # Tune
    tune_parser = subparsers.add_parser("tune", help="Run HPO Tuning")
    tune_parser.add_argument("--config", default="configs/deepscalper_unified.yaml")
    tune_parser.add_argument("--trials", type=int, default=10)
    tune_parser.add_argument("--steps", type=int, default=5000)
    tune_parser.add_argument("--resume", action="store_true")
    
    # Audit
    audit_parser = subparsers.add_parser("audit", help="Audit a trained model")
    audit_parser.add_argument("--config", default="configs/deepscalper_unified.yaml")
    audit_parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    audit_parser.add_argument("--output", default="reports/audit", help="Output directory")

    # Deploy
    deploy_parser = subparsers.add_parser("deploy", help="Deploy to Remote (GPUHub/RunPod)")
    deploy_parser.add_argument("--target", choices=["gpuhub", "runpod"], default="gpuhub")
    deploy_parser.add_argument("--config", default="configs/deepscalper_unified.yaml")
    deploy_parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    
    if args.command == "train":
        run_training(args)
    elif args.command == "tune":
        run_tuning(args)
    elif args.command == "audit":
        run_audit(args)
    elif args.command == "deploy":
        run_deploy(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
