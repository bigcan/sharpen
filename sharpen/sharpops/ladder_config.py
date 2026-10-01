"""Typed access to configs/sharpops_ladder.gates.yaml: the ONLY path from those numbers to code.

A promotion runner reads its thresholds here (never literals). ``numeric_keys_read()`` lets a
test prove every numeric key in the file is consumed, so none is declared-but-unwired.
"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
LADDER_PATH = ROOT / "configs" / "sharpops_ladder.gates.yaml"


def load(path: Path = LADDER_PATH) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))["ladder"]


def rung_alpha(rung: str, cfg: dict | None = None) -> float:
    return float((cfg or load())["rungs"][rung]["alpha"])


def challenge_daily_loss_bound(cfg: dict | None = None) -> float:
    return float((cfg or load())["rungs"]["challenge"]["daily_loss_bound"])


def target_sharpe(kind: str, cfg: dict | None = None) -> float:
    return float((cfg or load())["target_sharpe"][kind])


def sharpe_ceiling(kind: str, cfg: dict | None = None) -> float:
    return float((cfg or load())["tripwires"]["sharpe_ceiling"][kind])


def placebo(cfg: dict | None = None) -> dict:
    p = (cfg or load())["tripwires"]["shuffle_placebo"]
    return {"n_perm": int(p["n_perm"]), "max_null_sharpe": float(p["max_null_sharpe"]),
            "alpha": float(p["alpha"])}


def circular_shift(cfg: dict | None = None) -> dict:
    p = (cfg or load())["tripwires"]["circular_shift_null"]
    return {"n_shifts": int(p["n_shifts"]), "min_shift": int(p["min_shift_days"]),
            "alpha": float(p["alpha"])}


def calibration(cfg: dict | None = None) -> dict:
    p = (cfg or load())["calibration"]
    return {"n_sims": int(p["n_sims"]), "max_false_pass_over_alpha": float(p["max_false_pass_over_alpha"])}


def overlay(cfg: dict | None = None) -> dict:
    p = (cfg or load())["overlay"]
    return {"margin": float(p["non_inferiority_margin_sharpe"]), "alpha": float(p["alpha"]),
            "block": int(p["bootstrap_block_days"])}


def numeric_keys_read() -> set[str]:
    """Dotted paths of every numeric the accessors above read (kept in sync by a test)."""
    keys = {f"rungs.{r}.alpha" for r in ("paper", "challenge", "live")}
    keys |= {"rungs.challenge.daily_loss_bound",
             "target_sharpe.daily_multi_asset", "target_sharpe.intraday_single_asset",
             "tripwires.sharpe_ceiling.intraday_single_asset", "tripwires.sharpe_ceiling.daily_multi_asset",
             "tripwires.shuffle_placebo.n_perm", "tripwires.shuffle_placebo.max_null_sharpe",
             "tripwires.shuffle_placebo.alpha", "tripwires.circular_shift_null.n_shifts",
             "tripwires.circular_shift_null.min_shift_days", "tripwires.circular_shift_null.alpha",
             "calibration.n_sims", "calibration.max_false_pass_over_alpha",
             "overlay.non_inferiority_margin_sharpe", "overlay.alpha", "overlay.bootstrap_block_days"}
    return keys
