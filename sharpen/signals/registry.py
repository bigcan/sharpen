"""Decorator-based registry of candidate Signals.

Instance-based: ``register`` takes a ready Signal instance (so a family generator can mint
many parameterized instances — e.g. RSI(14), RSI(28) — each with its own SignalSpec and
register each). The batch evaluator pulls ``get_registry()`` and uses ``len(...)`` as the
honest ``n_trials`` for multiple-testing deflation.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .protocol import Signal

_REGISTRY: dict[str, "Signal"] = {}


def register(sig: "Signal") -> "Signal":
    """Register a Signal instance under ``sig.spec.name``; returns it (usable as a
    decorator on a class whose instances are created elsewhere, or called directly).

    Raises on duplicate names so two candidates never silently collide (which would
    corrupt the ``n_trials`` count).
    """
    name = sig.spec.name
    if name in _REGISTRY:
        raise ValueError(f"signal {name!r} already registered")
    _REGISTRY[name] = sig
    return sig


def get_registry(family: str | None = None) -> dict[str, "Signal"]:
    """Return the registry (optionally filtered to one family), as a fresh dict."""
    if family is None:
        return dict(_REGISTRY)
    return {k: v for k, v in _REGISTRY.items() if v.spec.family == family}


def clear_registry() -> None:
    """Empty the registry (test isolation)."""
    _REGISTRY.clear()
