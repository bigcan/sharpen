"""The clean-window seal.

Every day on or before ``SEAL_END`` belongs to the clean backward window: the project has never run its
cross-asset trend book there, so it is the one multi-decade out-of-sample window left. It may be looked at
ONCE, and only after the pre-registration is committed. Until then strategy code can read only the
development window (after ``SEAL_END``), and the sealed part of any series is visible only through
``hygiene`` (coverage and outlier counts, never a mean, volatility or return statistic).

Unsealing is not a flag: ``configs/finrlx_strategy.gates.yaml`` must name the pre-registration document
and its SHA-256, and that document must be tracked by git and identical to HEAD. So the rules that the
look is scored against are committed before the look can happen.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import TypeVar

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[3]
GATES_PATH = REPO / "configs" / "finrlx_strategy.gates.yaml"
SEAL_END = pd.Timestamp("2005-12-31")

PandasObj = TypeVar("PandasObj", pd.Series, pd.DataFrame)


class SealedError(RuntimeError):
    """Raised when code tries to read sealed (clean-window) data before the pre-registration is committed."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prereg_status(gates_path: Path = GATES_PATH) -> tuple[bool, str]:
    """Return (unsealed, reason). Unsealed only when the pre-registration is committed and unmodified."""
    if not gates_path.exists():
        return False, f"no gates file at {gates_path}"
    cfg = yaml.safe_load(gates_path.read_text(encoding="utf-8")) or {}
    pr = cfg.get("prereg") or {}
    rel, want = pr.get("path"), pr.get("sha256")
    if not rel or not want:
        return False, "gates file has no prereg.path / prereg.sha256"
    doc = REPO / rel
    if not doc.exists():
        return False, f"pre-registration {rel} does not exist"
    got = _sha256(doc)
    if got != want:
        return False, f"pre-registration sha256 {got[:12]} != gates {str(want)[:12]}"
    for args in (["ls-files", "--error-unmatch", rel], ["diff", "--quiet", "HEAD", "--", rel],
                 ["ls-files", "--error-unmatch", str(gates_path.relative_to(REPO))],
                 ["diff", "--quiet", "HEAD", "--", str(gates_path.relative_to(REPO))]):
        r = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)
        if r.returncode != 0:
            return False, f"git {' '.join(args)} failed: pre-registration or gates not committed/clean"
    return True, f"pre-registration {rel} committed at sha256 {got[:12]}"


def is_unsealed(gates_path: Path = GATES_PATH) -> bool:
    return prereg_status(gates_path)[0]


def _check_index(obj: pd.Series | pd.DataFrame) -> None:
    if not isinstance(obj.index, pd.DatetimeIndex):
        raise TypeError("seal views need a DatetimeIndex")


def dev_view(obj: PandasObj) -> PandasObj:
    """The development window: strictly after SEAL_END. Always readable."""
    _check_index(obj)
    return obj.loc[obj.index > SEAL_END]


def backward_view(obj: PandasObj, *, gates_path: Path = GATES_PATH) -> PandasObj:
    """The sealed clean window (<= SEAL_END). Raises SealedError until the pre-registration is committed."""
    _check_index(obj)
    ok, why = prereg_status(gates_path)
    if not ok:
        raise SealedError(f"clean window is sealed: {why}")
    return obj.loc[obj.index <= SEAL_END]


def guard(obj: pd.Series | pd.DataFrame, *, purpose: str, gates_path: Path = GATES_PATH) -> None:
    """Raise if ``obj`` reaches into the sealed window while it is still sealed."""
    _check_index(obj)
    if len(obj) and obj.index.min() <= SEAL_END:
        ok, why = prereg_status(gates_path)
        if not ok:
            raise SealedError(f"{purpose}: input reaches {obj.index.min().date()} <= {SEAL_END.date()} ({why})")


def hygiene(obj: pd.Series | pd.DataFrame, *, big_move: float = 0.10) -> pd.DataFrame:
    """Coverage and outlier COUNTS over the whole series, sealed part included.

    Deliberately returns no mean, volatility, drift or return statistic: counting gaps and outliers says
    nothing about how a strategy would have done. ``big_move`` counts |day-over-day change| above it.
    """
    _check_index(obj)
    df = obj.to_frame() if isinstance(obj, pd.Series) else obj
    rows = {}
    for c in df.columns:
        s = df[c]
        v = s.dropna()
        ch = v.pct_change().abs() if (v > 0).all() else v.diff().abs()
        rows[c] = {
            "first_valid": v.index.min() if len(v) else pd.NaT,
            "last_valid": v.index.max() if len(v) else pd.NaT,
            "n_obs": int(len(v)),
            "n_nan_inside": int(s.loc[v.index.min():v.index.max()].isna().sum()) if len(v) else 0,
            "n_sealed_obs": int((v.index <= SEAL_END).sum()),
            "n_nonpositive": int((v <= 0).sum()),
            "n_big_moves": int((ch > big_move).sum()),
            "n_repeats": int((v.diff() == 0).sum()),
            "max_gap_days": int(v.index.to_series().diff().dt.days.max()) if len(v) > 1 else 0,
        }
    return pd.DataFrame(rows).T


def assert_no_sealed_rows(obj: pd.Series | pd.DataFrame) -> None:
    """For tests and dev pipelines: fail loudly if a frame meant to be dev-only contains sealed rows."""
    _check_index(obj)
    if len(obj) and np.any(obj.index <= SEAL_END):
        raise SealedError(f"frame contains {int((obj.index <= SEAL_END).sum())} sealed rows")
