"""Pre-registered evaluation gates (loaded from configs/signal_eval.gates.yaml).

NEVER hardcode a numeric gate in code or scripts (project invariant). The harness reads
every threshold from here; ``Gates.default()`` mirrors the shipped YAML for tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_DEFAULTS: dict = {
    "universe": {"id": "sp500_pit", "min_names_per_day": 50, "min_adv_usd": 5.0e6},
    "coverage": {"min_days": 1260, "max_nan_frac": 0.40, "max_ohlc_violations": 0},
    "gross_power": {
        "horizons": [1, 5, 10, 21, 63],
        "primary_horizon": 5,
        "promising_ic_ir": 0.05,
        "promising_ic_tstat": 3.0,
    },
    "deflation": {"promising_dsr": 0.90, "fdr_q_max": 0.10},
    "robustness": {"min_subperiod_ic_ir": 0.0, "recent_oos_years": 2},
    "capturability": {
        "cost_models": {"frictionless": 0.0, "standard": 0.0010, "harsh": 0.0025},
        "cost_wall_caution": 0.30,
    },
    "neutralization": {"winsor_pct": [0.01, 0.99], "controls": ["sector", "size"]},
    "promotion": {"survivorship_free_required": True, "tier2_audit_required": True},
}


def _merge(base: dict, over: dict | None) -> dict:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for k, v in (over or {}).items():
        out[k] = {**out[k], **v} if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


@dataclass(frozen=True, slots=True)
class Gates:
    """Flattened view of the gates the harness reads; ``raw`` keeps the full config."""

    horizons: tuple[int, ...]
    primary_horizon: int
    min_days: int
    min_names_per_day: int
    max_ohlc_violations: int
    promising_ic_ir: float
    promising_ic_tstat: float
    promising_dsr: float
    fdr_q_max: float
    neutralization: tuple[str, ...]
    winsor_pct: tuple[float, float]
    cost_models: dict
    cost_wall_caution: float
    survivorship_free_required: bool
    tier2_audit_required: bool
    raw: dict

    @classmethod
    def from_dict(cls, d: dict | None = None) -> "Gates":
        m = _merge(_DEFAULTS, d)
        gp, defl, neu = m["gross_power"], m["deflation"], m["neutralization"]
        return cls(
            horizons=tuple(int(h) for h in gp["horizons"]),
            primary_horizon=int(gp["primary_horizon"]),
            min_days=int(m["coverage"]["min_days"]),
            min_names_per_day=int(m["universe"]["min_names_per_day"]),
            max_ohlc_violations=int(m["coverage"]["max_ohlc_violations"]),
            promising_ic_ir=float(gp["promising_ic_ir"]),
            promising_ic_tstat=float(gp["promising_ic_tstat"]),
            promising_dsr=float(defl["promising_dsr"]),
            fdr_q_max=float(defl["fdr_q_max"]),
            neutralization=tuple(["winsor", "zscore", *neu["controls"]]),
            winsor_pct=(float(neu["winsor_pct"][0]), float(neu["winsor_pct"][1])),
            cost_models=dict(m["capturability"]["cost_models"]),
            cost_wall_caution=float(m["capturability"]["cost_wall_caution"]),
            survivorship_free_required=bool(m["promotion"]["survivorship_free_required"]),
            tier2_audit_required=bool(m["promotion"]["tier2_audit_required"]),
            raw=m,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Gates":
        import yaml

        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})

    @classmethod
    def default(cls) -> "Gates":
        return cls.from_dict(None)
