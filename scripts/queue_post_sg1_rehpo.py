#!/usr/bin/env python
"""Queue GMGP1 XAUUSD + BTC + Funding-Arb DSAC re-HPOs after SG-1 completes.

Pipeline:
    1. Poll SG-1 Optuna study until target_trials completed.
    2. Refresh `agents.sac:` block of `configs/gmgp1_xauusd_ftmo_hpo.yaml` from
       SG-1 best trial params (writes `.bak` copy of original).
    3. Re-validate refreshed XAUUSD config (protocol v2).
    4. Launch `distributed_hpo_coordinator.py` for XAUUSD; wait for exit.
    5. Launch `distributed_hpo_coordinator.py` for BTC; wait for exit.
    6. Launch `distributed_hpo_coordinator.py` for Funding-Arb DSAC re-HPO.
       (S485: queued after wl7ir7ia killed — 199h/6 trials/BUG-01/monolithic.)

Usage (all default launches on GPUHub with 6 workers):
    python scripts/queue_post_sg1_rehpo.py \\
        --sg1_study sg1_xauusd_ftmo_rehpo_20260419 \\
        --db_url "$DISTRIBUTED_HPO_DB_URL"

Skip flags for partial resumption:
    --skip_wait          # SG-1 already done
    --skip_refresh       # config already seed-refreshed
    --skip_xauusd        # jump to BTC
    --skip_btc           # skip GMGP1 BTC
    --skip_funding_arb   # skip funding-arb stage (default: disabled,
                         #   audit pre-req blocks auto-launch)
    --enable_funding_arb # opt-in gate for stage 5 (requires audit first)
    --dry_run            # print planned actions, do nothing
"""
from __future__ import annotations

import argparse
import copy
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] queue — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("queue")

# HPs in SG-1 broad trial.params to copy into GMGP1 narrow-mode seed.
# Matches `finrl_pro_ds/hpo/objective.py` SAC broad path (lines 109-128).
SAC_HP_KEYS = (
    "lr_actor", "lr_critic", "lr_alpha",
    "tau", "gamma", "initial_alpha",
    "gradient_clip",
)
ENV_HP_KEYS = ("deadband_threshold",)
REWARD_HP_KEYS = ("dsr_eta",)


def wait_for_study(study_name: str, db_url: str, target_trials: int,
                   poll_interval: int) -> "optuna.study.Study":
    """Block until SG-1 study has >= target_trials in COMPLETE state."""
    import optuna
    logger.info("Polling study %r for %d completed trials (interval=%ds)",
                study_name, target_trials, poll_interval)
    while True:
        try:
            study = optuna.load_study(study_name=study_name, storage=db_url)
            trials = study.trials
            complete = sum(1 for t in trials if t.state == optuna.trial.TrialState.COMPLETE)
            running = sum(1 for t in trials if t.state == optuna.trial.TrialState.RUNNING)
            failed = sum(1 for t in trials if t.state == optuna.trial.TrialState.FAIL)
            logger.info("  progress: %d/%d complete | %d running | %d failed",
                        complete, target_trials, running, failed)
            if complete >= target_trials:
                best = study.best_trial
                logger.info("Study %r complete. Best trial #%d PF=%.4f",
                            study_name, best.number, best.value)
                return study
        except Exception as exc:
            logger.warning("poll error (will retry): %s: %s", type(exc).__name__, exc)
        time.sleep(poll_interval)


def refresh_gmgp1_xauusd_seed(
    config_path: Path, source_study: "optuna.study.Study", dry_run: bool,
) -> Path:
    """Patch GMGP1 XAUUSD config `agents.sac:` + `env.reward.dsr_eta` +
    `env.deadband_threshold` from SG-1 best trial params.

    Writes a `.bak` of the original, overwrites in place. Returns refreshed path.
    """
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    best = source_study.best_trial
    params = best.params
    missing = [k for k in (*SAC_HP_KEYS, *ENV_HP_KEYS, *REWARD_HP_KEYS)
               if k not in params]
    if missing:
        raise RuntimeError(
            f"SG-1 best trial #{best.number} missing params {missing}. "
            f"Available: {sorted(params.keys())}"
        )

    before = {
        "agents.sac": copy.deepcopy(cfg["agents"]["sac"]),
        "env.deadband_threshold": cfg["env"]["deadband_threshold"],
        "env.reward.dsr_eta": cfg["env"]["reward"]["dsr_eta"],
    }

    for k in SAC_HP_KEYS:
        cfg["agents"]["sac"][k] = float(params[k])
    cfg["env"]["deadband_threshold"] = float(params["deadband_threshold"])
    cfg["env"]["reward"]["dsr_eta"] = float(params["dsr_eta"])

    logger.info("Seed refresh (from SG-1 trial #%d, PF=%.4f):",
                best.number, best.value)
    for k in SAC_HP_KEYS:
        logger.info("  agents.sac.%s: %s -> %s",
                    k, before["agents.sac"].get(k), cfg["agents"]["sac"][k])
    logger.info("  env.deadband_threshold: %s -> %s",
                before["env.deadband_threshold"], cfg["env"]["deadband_threshold"])
    logger.info("  env.reward.dsr_eta: %s -> %s",
                before["env.reward.dsr_eta"], cfg["env"]["reward"]["dsr_eta"])

    if dry_run:
        logger.info("dry_run: skipping write of %s", config_path)
        return config_path

    bak = config_path.with_suffix(config_path.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(config_path, bak)
        logger.info("backup written: %s", bak.name)
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    logger.info("refreshed config written: %s", config_path)
    return config_path


def validate_config(config_path: Path, stage: str = "hpo") -> None:
    logger.info("validate_config.py --config %s --stage %s", config_path.name, stage)
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "validate_config.py"),
         "--config", str(config_path), "--stage", stage],
        cwd=PROJECT_ROOT, capture_output=True, text=True,
    )
    logger.info("validator stdout:\n%s", result.stdout)
    if result.returncode != 0:
        logger.error("validator stderr:\n%s", result.stderr)
        raise RuntimeError(f"validate_config FAIL for {config_path.name}")


def run_coordinator(
    config: Path, dhpo_overlay: Path, study_name: str, n_trials: int,
    n_workers: int, db_url: str, dry_run: bool,
) -> None:
    cmd = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "distributed_hpo_coordinator.py"),
        "--config", str(config),
        "--platform", "gpuhub",
        "--n_workers", str(n_workers),
        "--n_trials", str(n_trials),
        "--db_url", db_url,
        "--study_name", study_name,
        "--distributed_config", str(dhpo_overlay),
        "--resume",
    ]
    logger.info("coordinator: %s",
                " ".join(shlex_quote(_redact_db_url(p)) for p in cmd))
    if dry_run:
        logger.info("dry_run: skipping launch")
        return
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if proc.returncode != 0:
        raise RuntimeError(f"coordinator exit {proc.returncode} for {study_name}")
    logger.info("coordinator exited cleanly for %s", study_name)


def shlex_quote(s: str) -> str:
    """Cross-platform shell quoting for log visibility."""
    if not s or any(c in s for c in " \t\n\"'|&;<>()$"):
        return '"' + s.replace('"', '\\"') + '"'
    return s


def _redact_db_url(s: str) -> str:
    """Redact Postgres password in connection strings for log safety."""
    import re
    return re.sub(r"(://[^:/@]+:)[^@]+(@)", r"\1***\2", s)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sg1_study", default="sg1_xauusd_ftmo_rehpo_20260419")
    p.add_argument("--sg1_target_trials", type=int, default=50)
    p.add_argument("--xauusd_study", default="gmgp1_xauusd_ftmo_rehpo_20260419")
    p.add_argument("--xauusd_trials", type=int, default=30)
    p.add_argument("--btc_study", default="gmgp1_btc_velotrade_rehpo_20260419")
    p.add_argument("--btc_trials", type=int, default=50)
    p.add_argument("--funding_arb_study",
                   default="funding_arb_dsac_rehpo_20260419")
    p.add_argument("--funding_arb_trials", type=int, default=50)
    p.add_argument("--n_workers", type=int, default=6)
    p.add_argument("--poll_interval", type=int, default=600,
                   help="Seconds between Optuna polls (default 600 = 10 min)")
    p.add_argument("--db_url", default=os.environ.get("DISTRIBUTED_HPO_DB_URL"),
                   help="Optuna Postgres URL (defaults to $DISTRIBUTED_HPO_DB_URL)")
    p.add_argument("--skip_wait", action="store_true")
    p.add_argument("--skip_refresh", action="store_true")
    p.add_argument("--skip_xauusd", action="store_true")
    p.add_argument("--skip_btc", action="store_true")
    p.add_argument("--skip_funding_arb", action="store_true",
                   help="(default true — requires --enable_funding_arb)")
    p.add_argument("--enable_funding_arb", action="store_true",
                   help="Opt-in gate for stage 5. Requires pre-launch audit "
                        "(see configs/funding_arb_dsac_10assets_rehpo.yaml header).")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    if not args.db_url:
        logger.error("--db_url required (or set DISTRIBUTED_HPO_DB_URL)")
        return 2

    xauusd_cfg = PROJECT_ROOT / "configs" / "gmgp1_xauusd_ftmo_hpo.yaml"
    xauusd_dhpo = PROJECT_ROOT / "configs" / "dhpo_gmgp1_xauusd_ftmo.yaml"
    btc_cfg = PROJECT_ROOT / "configs" / "gmgp1_btc_velotrade_hpo.yaml"
    btc_dhpo = PROJECT_ROOT / "configs" / "dhpo_gmgp1_btc_velotrade.yaml"
    fa_cfg = PROJECT_ROOT / "configs" / "funding_arb_dsac_10assets_rehpo.yaml"
    fa_dhpo = PROJECT_ROOT / "configs" / "dhpo_funding_arb_dsac.yaml"
    for f in (xauusd_cfg, xauusd_dhpo, btc_cfg, btc_dhpo, fa_cfg, fa_dhpo):
        if not f.exists():
            logger.error("missing required config: %s", f)
            return 2

    # Stage 1: wait for SG-1
    sg1_study = None
    if not args.skip_wait:
        sg1_study = wait_for_study(
            args.sg1_study, args.db_url, args.sg1_target_trials, args.poll_interval,
        )
    else:
        logger.info("skip_wait: assuming SG-1 already complete")

    # Stage 2: seed-refresh XAUUSD
    if not args.skip_refresh:
        if sg1_study is None:
            import optuna
            sg1_study = optuna.load_study(study_name=args.sg1_study, storage=args.db_url)
        refresh_gmgp1_xauusd_seed(xauusd_cfg, sg1_study, args.dry_run)
        if not args.dry_run:
            validate_config(xauusd_cfg)
    else:
        logger.info("skip_refresh: leaving XAUUSD config untouched")

    # Stage 3: XAUUSD re-HPO
    if not args.skip_xauusd:
        run_coordinator(
            config=xauusd_cfg, dhpo_overlay=xauusd_dhpo,
            study_name=args.xauusd_study, n_trials=args.xauusd_trials,
            n_workers=args.n_workers, db_url=args.db_url, dry_run=args.dry_run,
        )
    else:
        logger.info("skip_xauusd: skipping XAUUSD coordinator")

    # Stage 4: BTC re-HPO
    if not args.skip_btc:
        run_coordinator(
            config=btc_cfg, dhpo_overlay=btc_dhpo,
            study_name=args.btc_study, n_trials=args.btc_trials,
            n_workers=args.n_workers, db_url=args.db_url, dry_run=args.dry_run,
        )
    else:
        logger.info("skip_btc: skipping BTC coordinator")

    # Stage 5: Funding-Arb DSAC re-HPO (S485). Opt-in only — default skipped
    # until pre-launch audit completes (see fa_cfg header TODO list).
    if args.enable_funding_arb and not args.skip_funding_arb:
        logger.warning("stage 5: Funding-Arb DSAC re-HPO launch requires "
                       "completed pre-launch audit — proceeding per "
                       "--enable_funding_arb flag.")
        run_coordinator(
            config=fa_cfg, dhpo_overlay=fa_dhpo,
            study_name=args.funding_arb_study, n_trials=args.funding_arb_trials,
            n_workers=args.n_workers, db_url=args.db_url, dry_run=args.dry_run,
        )
    else:
        logger.info("stage 5 skipped: Funding-Arb DSAC re-HPO (enable with "
                    "--enable_funding_arb after pre-launch audit)")

    logger.info("queue complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
