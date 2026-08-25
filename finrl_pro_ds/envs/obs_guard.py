"""Float16-safe observation bounds (NAN-01).

Why a magnitude clip, when every env already runs ``np.isfinite``/``np.nan_to_num``:

``training.use_amp: true`` defaults to ``amp_dtype: float16``
(``training/sac_trainer.py``, ``agents/sac/sac_agent.py``), whose largest finite value
is **65504**. A merely-huge but perfectly finite float64 feature — the shape produced by
any ratio whose denominator is guarded only against being *exactly* zero — passes
``np.isfinite`` in the env and only becomes ``+/-inf`` once autocast narrows the batch.
The encoder's first ``LayerNorm`` then turns that ``inf`` into NaN **for that row only**,
and the NaN surfaces as an invalid ``loc`` in ``torch.distributions.Normal`` inside
``SACActorNetwork.sample``. ``nan_to_num`` cannot see it; the env must bound it.

This is the crash that killed the first execution-overlay GPU run (randd_log
S553-cont-164/165) and the same latent exposure exists wherever an env divides by a
near-zero equity, portfolio value, or bar volume.

``OBS_CLIP`` is the *backstop*, not the fix: each env must also bound the quantity at
its source (a real volume floor, an equity floor tied to initial capital), so that
neither half is load-bearing on its own. 1e4 leaves >6x headroom under the float16
ceiling for the encoder's own Linear/LayerNorm activations.
"""
from __future__ import annotations

import numpy as np

# Largest finite float16. Anything above this becomes inf the instant AMP casts.
FP16_MAX = 65504.0

# Hard bound on every emitted observation feature. Every obs feature in this codebase
# is naturally O(1) (fractions, ratios, z-scores, bps), so this can only ever bind on a
# degenerate state — a wiped-out portfolio, a zero-volume bar, an empty parent order.
OBS_CLIP = 1.0e4


def sanitize_obs(buf: np.ndarray) -> np.ndarray:
    """NaN/inf -> 0, then a hard finite bound. In-place; returns ``buf``.

    Fast path: the ``isfinite`` scan is skipped for the (overwhelmingly common) all-finite
    case, matching the existing per-env guards this replaces.
    """
    if not np.isfinite(buf).all():
        np.nan_to_num(buf, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(buf, -OBS_CLIP, OBS_CLIP, out=buf)
    return buf
