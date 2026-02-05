import json
from datetime import datetime
import os

state_path = '.agent/ralph_state.json'
if os.path.exists(state_path):
    with open(state_path, 'r') as f:
        state = json.load(f)
else:
    print("State file not found.")
    exit(1)

fix_description = 'Reverted HPO n_trials to 50 (User Request)'

state.setdefault('fixes_applied', []).append({
    'timestamp': datetime.now().isoformat(),
    'iteration': state.get('iteration', 1),
    'fix': fix_description
})

with open(state_path, 'w') as f:
    json.dump(state, f, indent=4)

print(f'Log added: {fix_description}')
