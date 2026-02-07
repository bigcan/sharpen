import json
import yaml
import os
import shutil

STATE_FILE = '.agent/ralph_state.json'
# We will use the original config as a baseline to reset the LR
ORIGINAL_CONFIG = 'configs/deepscalper_rtx5090_production.yaml'

def patch_state():
    if not os.path.exists(STATE_FILE):
        print(f"State file {STATE_FILE} not found.")
        return

    with open(STATE_FILE, 'r') as f:
        state = json.load(f)

    print(f"Current Iteration: {state.get('iteration')}")
    print(f"Current Max Iterations: {state.get('max_iterations')}")

    # 1. Extend Max Iterations
    state['max_iterations'] = 20
    print(f"Extended Max Iterations to 20.")
    
    # 2. Reset Config pointer to a fresh start for iteration 11
    # We want to reset the Learning Rate because it was decimated.
    # Let's load the original config
    with open(ORIGINAL_CONFIG, 'r') as f:
        base_config = yaml.safe_load(f)
        
    # Create new config for iter 11
    os.makedirs('configs/autogen', exist_ok=True)
    new_config_path = f"configs/autogen/ralph_iter_{state['iteration'] + 1}.yaml"
    
    # Ensure LR is reasonable (e.g. 0.0001) if it's missing or zero
    if 'agents' not in base_config: base_config['agents'] = {'bdq': {}}
    base_config['agents']['bdq']['learning_rate'] = 0.0001
    
    with open(new_config_path, 'w') as f:
        yaml.dump(base_config, f)
        
    print(f"Created fresh config at {new_config_path} with LR=0.0001")
    state['config_file'] = new_config_path
    
    # 3. Optional: Clear 'fixes_applied' so it doesn't think we've already tried LR shift?
    # Actually, the user might want us to retry LR shift *correctly* this time.
    # The bug was that it *didn't* count them, so it kept doing it.
    # Now that it counts them, if we leave history as is, it will see 8 LR shifts (all lowercase 'learning_rate' won't match "Learning Rate" though? Wait.)
    # In the log file they are saved as: "Adjust HPO: Shift Learning Rate Range" (Title Case).
    # My fix was: `if 'learning_rate' in f['fix'].lower()`
    # So "Learning Rate".lower() -> "learning rate". "learning_rate" is NOT in "learning rate" (underscore vs space).
    # Ah, I need to check how the fix string is actually formatted in the code.
    # In `ralph_autonomous.py`: `fix_applied = "Adjust HPO: Shift Learning Rate Range"` (Spaces)
    # So `if 'learning_rate' in ...` will FAIL even with `.lower()` because of the underscore!
    
    # I NEED TO FIX THE LOGIC AGAIN in `ralph_autonomous.py` to match "learning rate" (no underscore) OR change the string to have underscore.
    
    # For now, let's just make sure the state is ready for restart.
    
    # Let's clean up the fixes history a bit so we don't immediately trigger "Increase Batch Size" check?
    # If `lr_fixes` >= 2, it switches to Batch Size.
    # We possess 8 "Shift Learning Rate" fixes in history.
    # If I fix the matching logic, it will count 8 fixes, and immediately try to increase Batch Size.
    # This might be what we want? The LR shift clearly didn't work (or rather, it worked too much).
    # But since we are resetting the LR to baseline, maybe we want to give LR shift another chance?
    # I think I should remove the last few "Shift Learning Rate" fixes from history to allow it to try again properly?
    # Let's just archive the history and clear `fixes_applied` to treat this as a fresh start for the new set of 10 iterations?
    # No, we want to keep the history of failures.
    
    # Let's just append a note to fixes saying "Manual Reset".
    state['fixes_applied'].append({
        'timestamp': "MANUAL_RESET",
        'iteration': state['iteration'],
        'fix': "Manual Reset of Learning Rate and Config",
        'reason': "Bug fix in logic",
        'new_config': new_config_path
    })

    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=4)
        
    print("State patched successfully.")

if __name__ == "__main__":
    patch_state()
