import json
from datetime import datetime
import os

state_path = '.agent/ralph_state.json'
if os.path.exists(state_path):
    with open(state_path, 'r') as f:
        state = json.load(f)
else:
    print("State file not found, creating new.")
    state = {'fixes_applied': []}

fix_description = 'Increased HPO n_trials from 50 to 75 due to Low Test Sharpe (-0.49)'

state.setdefault('fixes_applied', []).append({
    'timestamp': datetime.now().isoformat(),
    'iteration': state.get('iteration', 1),
    'fix': fix_description
})

# Make sure status is clean for Phase 1
state['status'] = 'idle'

with open(state_path, 'w') as f:
    json.dump(state, f, indent=4)

print(f'Fix logged: {fix_description}')
print('State updated. Ready for Ralph Phase 1.')
