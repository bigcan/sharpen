"""Every numeric in configs/sharpops_ladder.gates.yaml is consumed through ladder_config, so no
threshold is declared-but-unwired (the audit's most common failure mode)."""
from __future__ import annotations

import numpy as np

from sharpen.sharpops import ladder as L
from sharpen.sharpops import ladder_config as C
from sharpen.sharpops import tripwires as T

# Measured values documented beside a gate, not thresholds anything should read.
INFORMATIONAL_PREFIXES = ("rungs.challenge.reachability.",)


def _numeric_leaves(node, prefix=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _numeric_leaves(v, f"{prefix}{k}.")
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        yield prefix[:-1]


def test_every_numeric_key_is_read():
    leaves = {k for k in _numeric_leaves(C.load()) if not k.startswith(INFORMATIONAL_PREFIXES)}
    assert leaves == C.numeric_keys_read()


def test_accessors_return_the_file_values():
    cfg = C.load()
    assert C.rung_alpha("paper") == cfg["rungs"]["paper"]["alpha"]
    assert C.rung_alpha("paper") > C.rung_alpha("challenge") > C.rung_alpha("live")   # stricter up the ladder
    assert C.challenge_daily_loss_bound() == 0.05
    assert C.sharpe_ceiling("intraday_single_asset") >= C.sharpe_ceiling("daily_multi_asset")


def test_registered_values_drive_the_tripwires():
    ceil = C.sharpe_ceiling("daily_multi_asset")
    assert T.plausibility_ceiling(ceil, ceil)["pass"] and not T.plausibility_ceiling(ceil + 0.01, ceil)["pass"]
    pl = C.placebo()
    r = np.random.default_rng(1).normal(0, 0.01, 600)
    out = T.shuffle_placebo(lambda x: np.sign(x) * x, r, n_perm=20,
                            max_null_sharpe=pl["max_null_sharpe"], alpha=pl["alpha"])
    assert not out["pass"]


def test_registered_bound_keeps_the_eprocess_valid():
    bound, alpha = C.challenge_daily_loss_bound(), C.rung_alpha("challenge")
    sims = np.clip(L.planted_returns(0.0, 756, 300, seed=77), -bound + 1e-9, None)
    hits = sum(L.betting_eprocess(x, bound=bound, alpha=alpha)["rejected_h0"] for x in sims)
    assert T.wilson(hits, 300)[0] <= alpha
