"""BatchRenorm (Ioffe 2017) — the normalization layer CrossQ is built on.

Plain BatchNorm cannot be dropped into a bootstrapped Q-critic: it normalizes
with *batch* statistics while training and with *running* statistics at eval
time, and the gap between the two is exactly the instability that target
networks exist to paper over. BatchRenorm closes the gap by correcting the
batch-normalized activation toward the running statistics with two
stop-gradient terms::

    r = clip(sigma_batch / sigma_running, 1/r_max, r_max)
    d = clip((mu_batch - mu_running) / sigma_running, -d_max, d_max)
    y = (x - mu_batch) / sigma_batch * r + d

The (r_max, d_max) schedule starts at (1, 0) — where the layer is *identically*
plain BatchNorm — and relaxes only once the running estimates have had time to
settle. See `BatchRenorm1d._relaxation` for the three phases.

Reference: Ioffe, "Batch Renormalization" (arXiv:1702.03275); used by CrossQ
(Bhatt et al., ICLR 2024, arXiv:1902.05605).
"""
from typing import Optional

import torch
import torch.nn as nn


class BatchRenorm1d(nn.Module):
    """BatchRenorm over the feature dim of a (B, C) activation.

    Args:
        num_features: C.
        eps: variance floor.
        momentum: running-statistics update rate (CrossQ uses 0.01, i.e. a much
            longer memory than torch's BatchNorm default of 0.1).
        warmup_steps: training steps spent in the r=1, d=0 phase before the
            relaxation ramp begins. 0 disables the warmup entirely.
        r_max: final scale-correction clip.
        d_max: final shift-correction clip.
        affine: learn per-feature scale/shift.

    Numerics (NAN-01): statistics are always computed in float32 with autocast
    disabled. Under fp16 autocast the variance of an activation of magnitude
    ~300 already exceeds fp16's 65504 ceiling, and the resulting inf poisons
    only the rows of that batch — a partial-batch NaN that reads like a bad
    weight update rather than a bad input.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-3,
        momentum: float = 0.01,
        warmup_steps: int = 100_000,
        r_max: float = 3.0,
        d_max: float = 5.0,
        affine: bool = True,
    ):
        super().__init__()
        if num_features <= 0:
            raise ValueError(f"num_features must be positive, got {num_features}")
        self.num_features = num_features
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.warmup_steps = int(warmup_steps)
        self.r_max = float(r_max)
        self.d_max = float(d_max)

        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        # Mirrors the Python-side step counter so it survives a checkpoint
        # round-trip. Never read in the hot path — reading it would force a
        # device sync on every forward.
        self.register_buffer("num_batches_tracked", torch.zeros((), dtype=torch.long))
        self._steps = 0

        if affine:
            self.weight: Optional[nn.Parameter] = nn.Parameter(torch.ones(num_features))
            self.bias: Optional[nn.Parameter] = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def _relaxation(self) -> tuple[float, float]:
        """Current (r_max, d_max) clip bounds.

        Phase 1 — `steps < warmup_steps`: (1, 0). r and d collapse to the
            identity, so the layer is exactly BatchNorm while the running
            statistics are still too green to correct toward.
        Phase 2 — one further `warmup_steps`: linear ramp to the configured
            bounds.
        Phase 3 — full relaxation.
        """
        if self.warmup_steps <= 0:
            return self.r_max, self.d_max
        if self._steps < self.warmup_steps:
            return 1.0, 0.0
        frac = min(1.0, (self._steps - self.warmup_steps) / float(self.warmup_steps))
        return 1.0 + frac * (self.r_max - 1.0), frac * self.d_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"BatchRenorm1d expects (B, C), got shape {tuple(x.shape)}")
        if x.shape[1] != self.num_features:
            raise ValueError(
                f"BatchRenorm1d expects C={self.num_features}, got {x.shape[1]}",
            )
        if self.training and x.shape[0] < 2:
            raise ValueError(
                "BatchRenorm1d needs batch >= 2 in training mode (batch variance is "
                f"undefined for {x.shape[0]} sample(s)). Call .eval() for single-sample "
                "inference.",
            )

        orig_dtype = x.dtype
        # NAN-01: batch statistics in fp32, never in the autocast dtype.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x32 = x.float()
            if self.training:
                batch_mean = x32.mean(dim=0)
                batch_var = x32.var(dim=0, unbiased=False)
                batch_std = (batch_var + self.eps).sqrt()
                running_std = (self.running_var + self.eps).sqrt()

                r_max, d_max = self._relaxation()
                r = (batch_std.detach() / running_std).clamp(1.0 / r_max, r_max)
                d = ((batch_mean.detach() - self.running_mean) / running_std).clamp(-d_max, d_max)

                out = (x32 - batch_mean) / batch_std * r + d

                with torch.no_grad():
                    # lerp_(x, m) == (1 - m) * running + m * batch, i.e. the same
                    # EMA convention as torch's BatchNorm. The batch variance fed
                    # in is the BIASED one, matching the flax/CrossQ reference
                    # (torch's BatchNorm uses the unbiased estimate here); at the
                    # batch sizes this runs at the (n-1)/n gap is ~0.2%.
                    self.running_mean.lerp_(batch_mean.detach(), self.momentum)
                    self.running_var.lerp_(batch_var.detach(), self.momentum)
                    self._steps += 1
                    # fill_ with a Python scalar is a fire-and-forget kernel; it
                    # does not sync, unlike reading the buffer back.
                    self.num_batches_tracked.fill_(self._steps)
            else:
                out = (x32 - self.running_mean) / (self.running_var + self.eps).sqrt()

            if self.weight is not None:
                out = out * self.weight.float() + self.bias.float()

        return out.to(orig_dtype)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Restore the Python-side step counter from the checkpointed buffer."""
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        tracked = state_dict.get(prefix + "num_batches_tracked")
        if tracked is not None:
            self._steps = int(tracked.item())

    def extra_repr(self) -> str:
        return (
            f"{self.num_features}, eps={self.eps}, momentum={self.momentum}, "
            f"warmup_steps={self.warmup_steps}, r_max={self.r_max}, d_max={self.d_max}, "
            f"affine={self.weight is not None}"
        )
