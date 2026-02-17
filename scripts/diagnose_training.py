"""Diagnostic script to extract training curves from WandB for profitability analysis."""
import wandb
import json

api = wandb.Api()
run = api.run('bigcan-chiwin-technology/FinRL-Pro-DS/yg8gjeey')
rows = list(run.scan_history(page_size=500))

# 1. Training curves (reward, episode length, sps)
print('=== TRAINING CURVES ===')
train_rows = [r for r in rows if 'train/reward_mean' in r and r.get('train/reward_mean') is not None]
for r in train_rows:
    step = r.get('step', r.get('_step', '?'))
    reward = r.get('train/reward_mean', 0)
    ep_len = r.get('train/len_mean', 0)
    sps = r.get('train/sps', 0)
    epoch = r.get('train/epoch', '?')
    print(f"  Step {step:>7} | epoch={epoch} | reward={reward:.4f} | ep_len={ep_len:.0f} | sps={sps:.0f}")

# 2. Agent metrics  
print('\n=== AGENT METRICS ===')
agent_rows = [r for r in rows if 'agent/entropy' in r and r.get('agent/entropy') is not None]
for r in agent_rows:
    step = r.get('step', r.get('_step', '?'))
    entropy = r.get('agent/entropy', 0)
    kl = r.get('agent/approx_kl', 0)
    clip = r.get('agent/clip_fraction', 0)
    vloss = r.get('agent/value_loss', 0)
    ploss = r.get('agent/policy_loss', 0)
    lr = r.get('agent/learning_rate', 0)
    print(f"  Step {step:>7} | entropy={entropy:.3f} | kl={kl:.5f} | clip={clip:.4f} | vloss={vloss:.5f} | ploss={ploss:.5f} | lr={lr:.6f}")

# 3. Eval debug metrics
print('\n=== EVAL DEBUG (Action Distribution & PV) ===')
eval_rows = [r for r in rows if '_debug/eval_final_pv' in r and r.get('_debug/eval_final_pv') is not None]
for r in eval_rows:
    step = r.get('step', r.get('_step', '?'))
    buy = r.get('_debug/eval_action_buy', 0) or 0
    hold = r.get('_debug/eval_action_hold', 0) or 0
    sell = r.get('_debug/eval_action_sell', 0) or 0
    pv = r.get('_debug/eval_final_pv', 0)
    ret_mean = r.get('_debug/eval_returns_mean', 0) or 0
    ep_steps = r.get('_debug/eval_steps', 0)
    print(f"  Step {step:>7} | pv={pv:>10.2f} | ret_mean={ret_mean:.6f} | steps={ep_steps} | buy={buy:.1%} hold={hold:.1%} sell={sell:.1%}")

# 4. HPO trial results
print('\n=== HPO TRIALS ===')
for i in range(15):
    prefix = f'hpo/t{i}'
    trial_rows = [r for r in rows if f'{prefix}/status' in r and r.get(f'{prefix}/status') is not None]
    if trial_rows:
        r = trial_rows[-1]  # Latest status
        status = r.get(f'{prefix}/status', '?')
        score = r.get(f'{prefix}/score', '?')
        lr = r.get(f'{prefix}/learning_rate', '?')
        ent = r.get(f'{prefix}/ent_coef', '?')
        gamma = r.get(f'{prefix}/gamma', '?')
        epochs = r.get(f'{prefix}/n_epochs', '?')
        sw = r.get(f'{prefix}/sharpe_weight', '?')
        hw = r.get(f'{prefix}/hindsight_weight', '?')
        print(f"  Trial {i:>2} | status={status:>10} | score={score} | lr={lr} | ent={ent} | gamma={gamma} | epochs={epochs} | sharpe_w={sw} | hindsight_w={hw}")

# 5. Research metrics
print('\n=== RESEARCH METRICS (sampled) ===')
research_rows = [r for r in rows if '_research/sharpe_hourly' in r and r.get('_research/sharpe_hourly') is not None]
for r in research_rows[:10]:
    step = r.get('step', r.get('_step', '?'))
    sharpe_h = r.get('_research/sharpe_hourly', 0)
    raw_ratio = r.get('_research/raw_ratio_per_step', 0) or 0
    print(f"  Step {step:>7} | sharpe_hourly={sharpe_h:.4f} | raw_ratio/step={raw_ratio:.6f}")

# 6. Summary
print('\n=== RUN SUMMARY ===')
summary = run.summary._json_dict
for k in sorted(summary.keys()):
    if not k.startswith('_') and not k.startswith('hpo/'):
        v = summary[k]
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")
