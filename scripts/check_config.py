import yaml

c = yaml.safe_load(open('configs/tier1_validation.yaml'))
print("Config loaded OK")
print(f"  data: {c['data']['file_path']}")
print(f"  encoder: {c['network']['micro_config']['encoder_type']}")
print(f"  gamma: {c['agents']['ppo']['gamma']}")
print(f"  ent_coef: {c['agents']['ppo']['ent_coef']}")
print(f"  steps: {c['training']['total_timesteps']}")
print(f"  norm_cutoff: {c['data']['norm_cutoff_date']}")
print(f"  hindsight_weight: {c['env']['reward']['hindsight_weight']}")
print(f"  augmentation: {c['env']['augmentation']['enabled']}")
print(f"  tags: {c['wandb']['tags']}")
