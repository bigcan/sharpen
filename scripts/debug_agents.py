import argparse
import sys
import os

def main():
    print("Starting Agents/Torch Isolation Test...")
    
    # Argparse conformance
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--run_name", type=str)
    args = parser.parse_args()

    print("Importing Torch...")
    import torch
    print(f"Torch Version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print("CUDA is available.")
        try:
            print("Testing minimal CUDA op...")
            x = torch.tensor([1.0]).cuda()
            print(f"CUDA Op Success: {x}")
        except Exception as e:
            print(f"CUDA Op FAILED: {e}")
    else:
        print("CUDA NOT available.")

    print("Importing DeepScalper Agents...")
    try:
        from finrl_pro_ds.agents.deepscalper.dqn_agent import DeepScalperDQN
        from finrl_pro_ds.training.deepscalper_trainer import DeepScalperTrainer
        print("Agents Imported Successfully.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Agent Import FAILED: {e}")
        return

    print("Agents Debug SUCCESS.")

if __name__ == "__main__":
    main()
