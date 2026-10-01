"""Train the real SACTrainer/SACAgent on a planted-signal market, then evaluate deterministically on the
held-out test window next to a linear oracle and a flat policy run through the SAME env.

Usage:
    python run_arm.py <arm_name> <parquet> <out_root> [key=value ...]

``key=value`` overrides map onto ``harness.base_config`` keyword arguments (e.g. ``seed=2``,
``episode_length=5000``, ``reward_mode=pnl``, ``use_amp=0``, ``total_timesteps=20000``).
Writes ``<out_root>/<arm>/result.json`` (+ the final checkpoint under ``<out_root>/<arm>/checkpoints``).
WandB is disabled; nothing is written inside the repository unless ``out_root`` points there.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import base_config, make_eval_env, oracle_policy, rollout, summarise  # noqa: E402

log = logging.getLogger("rl_planted_signal.run_arm")

_BOOL_KEYS = {"use_amp", "torch_compile"}


def _parse_overrides(pairs: list[str]) -> dict:
    over: dict = {}
    for kv in pairs:
        k, v = kv.split("=", 1)
        if k in _BOOL_KEYS:
            over[k] = v.lower() in ("1", "true", "yes")
            continue
        for cast in (int, float):
            try:
                over[k] = cast(v)
                break
            except ValueError:
                continue
        else:
            over[k] = v
    return over


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    arm, parquet, out_root = sys.argv[1], str(Path(sys.argv[2]).resolve()), Path(sys.argv[3]).resolve()
    over = _parse_overrides(sys.argv[4:])
    vec_sync = bool(over.pop("vec_sync", 0))           # production default: AsyncVectorEnv
    max_eval_steps = int(over.pop("max_eval_steps", 10 ** 6))   # smoke tests only
    cfg = base_config(parquet, **over)
    arm_dir = out_root / arm
    arm_dir.mkdir(parents=True, exist_ok=True)

    import wandb
    wandb.init(mode="disabled", project="rl-planted-signal")
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from sharpen.hpo.env_factory import create_vector_env
    from sharpen.training.sac_trainer import SACTrainer

    env = create_vector_env(cfg, num_envs=cfg["training"]["num_envs"], gym_shm=False, use_sync=vec_sync,
                            start_date=cfg["data"]["train_start_date"],
                            end_date=cfg["data"]["train_end_date"], seed=cfg["seed"])
    os.chdir(arm_dir)                                  # trainer checkpoints land under the arm dir
    # hpo_mode=False: production path (hpo_mode caps the replay buffer at 100k).
    trainer = SACTrainer(env, cfg, device=device, hpo_mode=False, run_name=f"rlps_{arm}")
    agent = trainer.agent

    trace: list[dict] = []
    orig = agent.train_step_mega

    def _traced(n_steps=1):
        m = orig(n_steps)
        if m is not None and (not trace or agent._train_step_count - trace[-1]["gstep"] >= 2000):
            trace.append({"gstep": int(agent._train_step_count), "alpha": float(m["alpha"]),
                          "entropy": float(m["entropy"]), "critic_loss": float(m["critic_loss"]),
                          "actor_loss": float(m["actor_loss"]), "q1": float(m["q1_mean"])})
            log.info("PROGRESS %s gstep=%d elapsed_min=%.1f alpha=%.4f entropy=%.3f q1=%.3f",
                     arm, trace[-1]["gstep"], (time.time() - t0) / 60, trace[-1]["alpha"],
                     trace[-1]["entropy"], trace[-1]["q1"])
        return m
    agent.train_step_mega = _traced

    t0 = time.time()
    trainer.train()
    train_s = time.time() - t0
    env.close()

    n_scales = len(cfg["features"]["scales"])
    dev = torch.device(device)

    def sac_policy(obs) -> float:
        st = np.stack([obs[f"scale_{i}"] for i in range(n_scales)], axis=0)
        s = torch.as_tensor(st, dtype=torch.float32).unsqueeze(0).to(dev, non_blocking=True)
        p = torch.as_tensor(obs["private"], dtype=torch.float32).unsqueeze(0).to(dev, non_blocking=True)
        return float(agent.predict(s, p, deterministic=True)[0, 0].item())

    res = {"arm": arm, "parquet": Path(parquet).name, "overrides": over, "vec_sync": vec_sync,
           "device": device, "raw_channel_norm": cfg["features"]["raw_channel_norm"],
           "train_seconds": round(train_s, 1), "gradient_steps": int(agent._train_step_count),
           "final_alpha": float(agent.alpha), "grad_skips": int(agent._nonfinite_grad_skips)}
    # Oracle and flat always run on RAW features (the oracle reads the raw return), so their numbers
    # match the raw-feature ladder. The flat pass never stops early: its feat0 is the raw planted
    # signal for every test bar, and every policy is scored against it.
    raw_cfg = copy.deepcopy(cfg)
    raw_cfg["features"]["raw_channel_norm"] = "raw"
    ref = rollout(make_eval_env(raw_cfg), lambda _o: 0.0, max_steps=max_eval_steps)
    for tag, pol, c in (("sac", sac_policy, cfg), ("oracle", oracle_policy, raw_cfg)):
        roll = rollout(make_eval_env(c), pol, max_steps=max_eval_steps)
        n = len(roll["position"])
        if not np.array_equal(roll["price_return"], ref["price_return"][:n]):
            raise RuntimeError(f"{tag}: eval env walked different bars from the reference pass")
        roll["signal"] = ref["feat0"][:n]
        res[tag] = summarise(roll)
    res["flat"] = summarise(ref)
    o = res["oracle"]["sharpe_frictionless"]
    res["sac_fraction_of_oracle"] = (res["sac"]["sharpe_frictionless"] / o) if abs(o) > 1e-9 else None
    (arm_dir / "result.json").write_text(json.dumps({"result": res, "trace": trace}, indent=2))
    log.info("RESULT %s", json.dumps(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
