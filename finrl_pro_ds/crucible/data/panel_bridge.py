"""The P1b→P1a bridge (spec §4.5 / CR-9): turn connector series into Panel feature slots.

This closes the loop. A :class:`DataConnector` yields PIT observations; this module fetches them,
runs the data-quality gate, binds them to the trading calendar via the *only* sanctioned join
(:func:`quality_gate.asof_join` — release-time, never reference-period), and returns a
``{terminal: (T,) array}`` dict ready to drop into ``Panel(feature_slots=...)``. From there the
Crucible P1a grammar registry makes each series addressable as a DSL terminal (e.g. ``fred:DGS10``),
and the overlay/conditioner path scores it against the book.

The one constraint the bridge enforces up front: a feature-slot key must be a **DSL-legal terminal**
(``[A-Za-z_]\\w*(?::[A-Za-z_]\\w*)?`` — one ``source:series`` segment, letter-leading). FRED ids like
``DGS10`` are already legal; a COT ``088691:comm_net`` is not (digit-leading, two colons), so the
caller supplies an explicit alias (e.g. ``cot:gold_comm_net``). Enforcing this here fails fast rather
than at parse time deep inside a search.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..catalog import DataCatalog
from .connector import DataConnector, SeriesRef
from .quality_gate import asof_join, assert_asof_join_causal, register_series, validate_series

logger = logging.getLogger(__name__)

# Must match the P1a DSL name-token grammar (library/_alpha_dsl.py:29): one ``source:series`` segment.
_TERMINAL_RE = re.compile(r"[A-Za-z_]\w*(?::[A-Za-z_]\w*)?\Z")


def is_valid_terminal(name: str) -> bool:
    """True iff ``name`` lexes as a single DSL feature-slot terminal (P1a grammar)."""
    return bool(_TERMINAL_RE.match(name))


@dataclass(frozen=True, slots=True)
class SlotRequest:
    """One series to bind into a feature slot.

    ``terminal`` is the DSL-legal slot key the series is addressable as. It defaults to
    ``ref.terminal`` (``source:series_id``) — correct for FRED, but a caller MUST override it for
    ids that aren't DSL-legal (e.g. a COT ``088691:comm_net`` → ``cot:gold_comm_net``).
    """

    connector: DataConnector
    ref: SeriesRef
    terminal: str | None = None

    def resolved_terminal(self) -> str:
        return self.terminal if self.terminal is not None else self.ref.terminal


def build_feature_slots(
    requests: Sequence[SlotRequest],
    bar_dates: np.ndarray,
    *,
    start: np.datetime64 | str,
    end: np.datetime64 | str,
    catalog: DataCatalog | None = None,
    require_valid: bool = False,
    freshness: str | None = None,
) -> dict[str, np.ndarray]:
    """Fetch → validate → as-of-join each request onto ``bar_dates`` → ``{terminal: (T,) array}``.

    For each series: (1) the terminal must be DSL-legal; (2) :func:`validate_series` runs (a failure
    warns, or raises if ``require_valid``); (3) the release-time :func:`asof_join` binds it to the
    calendar; (4) the PIT leak gate :func:`assert_asof_join_causal` RAISES on any look-ahead — this is
    non-negotiable (CR-4/LEAK-2); (5) it is optionally registered into ``catalog`` (Stage-1 ACQUIRE).
    Returns a dict directly usable as ``Panel(feature_slots=...)``.

    :func:`quality_gate.assert_asof_join_causal` is the FULL causal+truncation gate (it certifies the
    join equals the release-time per-bar as-of value at every bar, subsuming the weaker availability-
    only check). It is O(M log M + T log M) — cheap enough to run in the hot path per slot (it was an
    O(T²) prefix sweep, reserved for CI, until the per-bar-reference rewrite).
    """
    bars = np.asarray(bar_dates, dtype="datetime64[ns]")
    slots: dict[str, np.ndarray] = {}
    for req in requests:
        terminal = req.resolved_terminal()
        if not is_valid_terminal(terminal):
            raise ValueError(
                f"feature-slot terminal {terminal!r} is not a DSL-legal name "
                f"([A-Za-z_]\\w* with one optional ':segment'); pass an explicit `terminal` alias")
        if terminal in slots:
            raise ValueError(f"duplicate feature-slot terminal {terminal!r}")

        data = req.connector.fetch(req.ref, start, end)
        report = validate_series(data)
        if not report.passed:
            msg = (f"{terminal}: data-quality gate flagged {report.reasons} "
                   f"(n_obs={report.n_obs}, max_gap={report.max_gap_days:.0f}d)")
            if require_valid:
                raise ValueError(msg)
            logger.warning(msg)

        joined = asof_join(data, bars)
        assert_asof_join_causal(data, bars)            # HARD PIT gate (full causal+truncation, O(T log M))
        slots[terminal] = joined
        if catalog is not None:
            register_series(catalog, data, freshness=freshness)
        logger.info("feature slot %s: %d/%d bars populated", terminal,
                    int(np.isfinite(joined).sum()), bars.shape[0])
    return slots
