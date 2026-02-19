"""Gather all data for the Research Roadmap Report."""
import sqlite3, json, os, sys

sys.stdout.reconfigure(encoding='utf-8')

# 1. DB metrics
db = sqlite3.connect('results/metrics.db')
rows = db.execute(
    "SELECT id, name, status, created_at, sharpe, total_return, tags FROM runs "
    "ORDER BY created_at ASC"
).fetchall()
print(f"=== METRICS DB: {len(rows)} runs ===\n")

for r in rows:
    rid, name, status, created, sharpe, ret, tags = r
    name = (name or '?')[:45]
    sharpe = float(sharpe or 0)
    ret = float(ret or 0)
    tags = (tags or '')[:70]
    date = (created or '')[:10]
    print(f"{rid:<12} {date:<12} {name:<47} {sharpe:>8.3f} {ret:>8.3f}% {tags}")

db.close()

# 2. Detailed JSON data for key roadmap runs
key_runs = ['ilu4exzb', '6tr6q5kq', 'fzi75q00', 'o2b9jiwi', 'ubtma4h2', 'jv4xlrg5']
print(f"\n\n=== KEY RUN DETAILS ===\n")
for rid in key_runs:
    jpath = f'results/run_data_{rid}.json'
    if not os.path.exists(jpath):
        print(f"\n--- {rid}: NO LOCAL DATA ---")
        continue
    d = json.load(open(jpath, encoding='utf-8'))
    s = d.get('summary', {})
    cfg = d.get('config', {})
    
    print(f"\n--- {rid}: {d.get('name', '?')} ---")
    print(f"  State: {d.get('state', '?')}")
    print(f"  Tags: {d.get('tags', [])}")
    print(f"  Agent: {cfg.get('agent_type', '?')}")
    
    # HPO
    hpo_trials = s.get('hpo/n_trials', '?')
    hpo_best = s.get('hpo/best_trial', '?')
    hpo_pf = s.get('hpo/best_profit_factor', '?')
    print(f"  HPO: {hpo_trials} trials, best=#{hpo_best}, PF={hpo_pf}")
    
    # Training
    print(f"  Training steps: {s.get('step', s.get('_step', '?'))}")
    print(f"  Final step: {s.get('train/final_step', '?')}")
    
    # Backtest
    for split in ['val', 'test']:
        prefix = f'backtest_{split}'
        sharpe = s.get(f'{prefix}/sharpe', '?')
        pf = s.get(f'{prefix}/profit_factor', '?')
        ret = s.get(f'{prefix}/total_return', '?')
        mdd = s.get(f'{prefix}/max_drawdown', '?')
        wr = s.get(f'{prefix}/win_rate', '?')
        steps = s.get(f'{prefix}/steps', '?')
        print(f"  {split.upper():>5}: Sharpe={sharpe}, PF={pf}, Return={ret}, MDD={mdd}, WR={wr}, Steps={steps}")
    
    # Debug eval
    eval_pf = s.get('_debug/eval_profit_factor', '?')
    eval_sharpe_h = s.get('_research/sharpe_hourly', '?')
    eval_steps = s.get('_debug/eval_steps', '?')
    buy_pct = s.get('_debug/eval_action_buy', '?')
    hold_pct = s.get('_debug/eval_action_hold', '?')
    sell_pct = s.get('_debug/eval_action_sell', '?')
    print(f"  EVAL: PF={eval_pf}, Sharpe(h)={eval_sharpe_h}, steps={eval_steps}")
    print(f"  Actions: Buy={buy_pct}, Hold={hold_pct}, Sell={sell_pct}")
    
    # Entropy
    entropy = s.get('agent/entropy', '?')
    print(f"  Entropy: {entropy}")

# 3. Existing report files
print(f"\n\n=== EXISTING REPORTS ===")
for f in sorted(os.listdir('results')):
    if f.endswith('.md'):
        size = os.path.getsize(f'results/{f}')
        print(f"  {f} ({size:,} bytes)")

# 4. V5 roadmap artifact
roadmap_path = '.agent/artifacts/v5_architecture_research_roadmap.md'
if os.path.exists(roadmap_path):
    print(f"\n\n=== V5 RESEARCH ROADMAP (first 100 lines) ===")
    with open(roadmap_path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 100:
                print("  [...truncated...]")
                break
            print(f"  {line.rstrip()}")
