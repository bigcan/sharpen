"""Unit tests: SignalSpec validation/hashing and the registry."""
from __future__ import annotations

import numpy as np
import pytest

from sharpen.signals import SignalSpec, clear_registry, get_registry, register


def test_content_hash_deterministic_and_sensitive() -> None:
    a = SignalSpec(name="x", hypothesis="h", family="technical", expected_sign=1)
    b = SignalSpec(name="x", hypothesis="h", family="technical", expected_sign=1)
    c = SignalSpec(name="x", hypothesis="h", family="technical", expected_sign=-1)
    assert a.content_hash() == b.content_hash()
    assert a.content_hash() != c.content_hash()
    assert len(a.content_hash()) == 12
    # GOLDEN PIN (crucible P0/P1a reproducibility moat): a frozen literal so a future change to
    # the hash payload / _HASH_FIELDS fails HERE — the run-to-run manifest gate can't catch a
    # cross-version content_hash drift (both runs use the drifted code).
    assert a.content_hash() == "47f98a86e056"


def test_candidate_type_excluded_from_hash() -> None:
    """Crucible P1a reproducibility invariant: candidate_type must NOT enter content_hash, or
    it re-hashes every pre-P1a library spec + DslSignal genome and breaks the ledger dedup key
    and the P0 byte-identical manifest gate."""
    base = SignalSpec(name="x", hypothesis="h", family="technical", expected_sign=1)
    overlay = SignalSpec(name="x", hypothesis="h", family="technical", expected_sign=1,
                         candidate_type="overlay")
    assert base.candidate_type == "cross_sectional"          # default is the OHLCV path
    assert base.content_hash() == overlay.content_hash()     # field does NOT shift the hash
    with pytest.raises(ValueError):
        SignalSpec(name="x", hypothesis="h", family="technical", expected_sign=1,
                   candidate_type="bogus")


def test_spec_validation() -> None:
    with pytest.raises(ValueError):
        SignalSpec(name="x", hypothesis="h", family="technical", expected_sign=2)
    with pytest.raises(ValueError):
        SignalSpec(name="x", hypothesis="h", family="bogus", expected_sign=1)
    with pytest.raises(ValueError):
        SignalSpec(name="x", hypothesis="h", family="technical", expected_sign=1, horizons=())
    with pytest.raises(ValueError):
        SignalSpec(name="x", hypothesis="h", family="technical", expected_sign=1,
                   neutralization=("winsor", "nonsense"))


class _Dummy:
    def __init__(self, name: str, family: str = "technical") -> None:
        self.spec = SignalSpec(name=name, hypothesis="h", family=family, expected_sign=0)

    def compute(self, panel) -> np.ndarray:  # pragma: no cover - not exercised here
        return np.zeros((panel.T, panel.N))


def test_registry_roundtrip_and_family_filter() -> None:
    clear_registry()
    try:
        s1 = register(_Dummy("d1", "technical"))
        s2 = register(_Dummy("d2", "lob"))
        assert get_registry()["d1"] is s1
        assert set(get_registry()) == {"d1", "d2"}
        assert set(get_registry(family="technical")) == {"d1"}
        assert get_registry(family="altdata") == {}
        with pytest.raises(ValueError):
            register(_Dummy("d1"))  # duplicate name
        assert s2.spec.family == "lob"
    finally:
        clear_registry()
    assert get_registry() == {}
