"""Forward runner for linear-core paper books (TAILWIND Tier-2 roadmap X2).

The batch executor (``TwoSleeveExecutor.sim_oracle``) books each month-end rebalance at the
month-end close under ``execution.decision_lead_bars: 1``. Nothing could run that book forward:
the month-end flags came from the data window, whose last row is never a confirmed month-end,
and the documented live target ``linear_core_weights(window)[-1]`` filled one or two closes late
(audit T2-01, T1-04, T6-04). This module is the forward path.

**The forward read.** At the close of session ``s`` (the as-of session) the runner decides the
weights to fill at the close of the next session ``d``:
1. It truncates the prices at ``s``. Any later bar is an in-progress bar and is dropped, and the
   data must reach ``s`` (freshness: data at most one session older than the fill).
2. It appends the next ``SESSIONS_AHEAD = 2`` sessions from the ex-ante NYSE calendar
   (``sharpen.data.trading_calendar``) as empty rows.
3. It recomputes every signal from those prices (``prepare_two_sleeve_payload``: nothing is
   sliced from a precomputed history) and drives the unchanged executor over the whole window.
4. It reads row ``-2`` of the combined weights: the step that fills at ``d``.

Two appended sessions, not one, because the drive treats a window's LAST row as unconfirmed.
The lead's final-row hold and the alpha's monthly flag both key on it. With ``d`` and the
session after it in the window, every flag the read step uses is computed with the next session
present. That flag is therefore exactly the calendar's month-end flag, for conviction and for
alpha alike, and the final row only touches the unread last step. Row ``d``'s conviction and vol
need data ``<= s`` only (their builders declare cutoff lag 1, tripwire-verified), so the
appended rows never feed the read step with anything unknown at ``s``.

**Daily cycle** (:meth:`ForwardRunner.run_session`): decide, then settle every session since the
book's last mark (normally one) at its close. Settling fills the target logged for that session
through ``SimFillEngine`` into the persisted ``PaperState``. A session with no logged target
holds, and is recorded as missed. Then the risk kills are checked, and the new target (or a
flatten) is appended to the hash-chained target log. State lives in ``state_dir``:
- ``targets.jsonl``: append-only, each record hash-chained to the previous one;
- ``fills.jsonl``: one record per booked session;
- ``paper_state.parquet`` and ``runner_state.json``: the book and its marks.

**Parity is incremental, not tautological** (T2-02). Each logged target was computed from prices
truncated at its own as-of session. Every run recomputes the whole history in one batch on
today's data and compares the book's positions with it, session by session, under the
``paper_soak.parity`` gates. A signal that read the future would make the batch differ from what
was logged.

**Vendor re-adjustment** (T1-10). Adjusted prices are re-scaled on every refetch. The book stores
its last marks, and on load re-bases each entry price by ``new_mark / old_mark``, so a dividend
re-adjustment is not booked as a loss.

**Kills** (fail closed, terminal). Any of these triggers a flatten at the next session and stops
the book:
- a kill file (``safety.kill_file`` or ``<state_dir>/KILL``);
- drawdown beyond ``paper_soak.risk.max_drawdown_kill_pct``;
- a session loss beyond ``daily_loss_halt_pct``;
- a target above ``max_gross_exposure``;
- a per-sleeve action-drift CRIT when ``gates.safe_mode.crit_triggers_flatten``.
Whether a challenge phase should use terminal kills at all is still the operator's decision
(audit R3).

**Exit codes** (roadmap N3's map): PASS 0, REVIEW 3, FAIL 1. A soak whose only gap is its
horizon exits 0 with ``overall_status`` UNKNOWN. A hard group that cannot be evaluated, a kill,
stale data or a broken log exits 1.
"""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from sharpen.data import trading_calendar as tc
from sharpen.data.cross_asset_loader import build_two_sleeve_arrays, prepare_two_sleeve_payload
from sharpen.envs.allocator_factory import (
    decision_lead_bars,
    execution_stamp,
    monthly_rebal_conviction,
)
from sharpen.paper.paper_state import LiveTrajectory, PaperState, generate_orders
from sharpen.paper.parity_harness import ParityHarness, ParityReport
from sharpen.paper.soak_metrics import FAIL, UNKNOWN, evaluate_paper_soak_gates
from sharpen.paper.two_sleeve import TwoSleeveExecutor

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
SESSIONS_AHEAD = 2
EXIT_PASS, EXIT_FAIL, EXIT_REVIEW = 0, 1, 3
TARGET_LOG, FILL_LOG, RUNNER_STATE = "targets.jsonl", "fills.jsonl", "runner_state.json"
_GENESIS = "0" * 64
_REPEAT_TOL_L1 = 1e-9           # a re-run of the same as-of must reproduce the logged weights


class ForwardRunError(RuntimeError):
    """A condition under which the runner must not trade (fail closed)."""


class StaleDataError(ForwardRunError):
    """The data does not reach the as-of session."""


class TargetLogError(ForwardRunError):
    """The target log's hash chain is broken or a record is out of order."""


def _retry_io(fn, *, attempts: int = 12):
    """Run a file write that Windows can refuse transiently (an antivirus or indexer holding a
    file that was just written). Retries on PermissionError with backoff (about 4 s in all)."""
    import time

    for k in range(attempts):
        try:
            return fn()
        except PermissionError:
            if k == attempts - 1:
                raise
            time.sleep(min(0.05 * 2 ** k, 1.0))


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def frame_sha16(*frames: pd.DataFrame) -> str:
    """Content hash of the price/volume frames a target was computed from."""
    h = hashlib.sha256()
    for f in frames:
        h.update(",".join(map(str, f.columns)).encode("utf-8"))
        h.update(pd.util.hash_pandas_object(f, index=True).values.tobytes())
    return h.hexdigest()[:16]


def _iso(ts) -> str:
    return pd.Timestamp(ts).date().isoformat()


def _session_dates(timestamps) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(np.asarray(timestamps, dtype=np.int64), unit="s")).normalize()


# --------------------------------------------------------------------------- #
# Data checks and the forward read
# --------------------------------------------------------------------------- #
def truncate_to_session(close: pd.DataFrame, volume: pd.DataFrame,
                        as_of) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rows up to and including ``as_of``. A later row is an in-progress (or future) bar and is
    dropped. Fails closed when:
    - the data does not reach ``as_of`` (freshness);
    - its sessions disagree with the ex-ante calendar;
    - a listed asset has no close on ``as_of``."""
    as_of = pd.Timestamp(as_of).normalize()
    if not tc.is_session(as_of):
        raise ForwardRunError(f"{as_of.date()} is not an NYSE session")
    c, v = close.loc[close.index <= as_of], volume.loc[volume.index <= as_of]
    dropped = int((close.index > as_of).sum())
    if dropped:
        log.info("forward: dropped %d bar(s) after the as-of session %s (in-progress)",
                 dropped, as_of.date())
    if len(c) == 0 or c.index[-1] != as_of:
        last = c.index[-1].date() if len(c) else None
        raise StaleDataError(f"data ends {last}; the as-of session is {as_of.date()}")
    expected = tc.sessions(c.index[0], as_of)
    if not c.index.equals(expected):
        extra = c.index.difference(expected)[:5].date.tolist()
        missing = expected.difference(c.index)[:5].date.tolist()
        raise ForwardRunError(f"data sessions disagree with the NYSE calendar: data-only {extra}, "
                              f"calendar-only {missing}")
    listed = c.notna().any()
    unmarked = [a for a in c.columns if listed[a] and not np.isfinite(c[a].iloc[-1])]
    if unmarked:
        raise StaleDataError(f"no close on {as_of.date()} for {unmarked}")
    return c, v


def extend_with_next_sessions(close: pd.DataFrame, volume: pd.DataFrame,
                              n: int = SESSIONS_AHEAD) -> tuple[pd.DataFrame, pd.DataFrame,
                                                                pd.DatetimeIndex]:
    """Append the next ``n`` calendar sessions as empty rows. The array builders forward-fill
    prices, so the placeholders never make an asset unavailable, and no signal row the read
    step uses depends on them."""
    ahead = tc.next_sessions(close.index[-1], n)
    empty = pd.DataFrame(np.nan, index=ahead, columns=close.columns)
    return (pd.concat([close, empty]),
            pd.concat([volume, empty.reindex(columns=volume.columns)]), ahead)


@dataclass
class ForwardDecision:
    """One forward read: the target for ``fill_session`` from data through ``as_of``, plus the
    batch (every step on this window) it was read from."""

    as_of: pd.Timestamp
    fill_session: pd.Timestamp
    assets: list[str]
    weights: np.ndarray
    alphas: dict[str, float]
    sleeve_conviction: dict[str, list[float]]
    sleeve_assets: dict[str, list[str]]
    month_end_fill: bool
    data_sha16: str
    batch_weights: np.ndarray            # (T-1, N): row k fills at the close of union bar k+1
    union: dict = field(repr=False)
    # Each sleeve's batch weights, alphas and assets over the window of ``batch_weights``, so
    # ``evaluate`` can attribute the book per sleeve (audit T4-12b).
    sleeve_batch: dict = field(default_factory=dict, repr=False)


def compute_forward_decision(
    config: Mapping,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    *,
    as_of,
    curve: Mapping | None = None,
    curve_manifest: Mapping | None = None,
    assert_causal: bool = True,
) -> ForwardDecision:
    """The weights to fill at the close of the session after ``as_of``, from data through
    ``as_of`` only (see the module docstring for the read rule)."""
    c, v = truncate_to_session(close, volume, as_of)
    c_ext, v_ext, ahead = extend_with_next_sessions(c, v)
    payload = prepare_two_sleeve_payload(
        config, c_ext, v_ext, manifest={"status": "forward", "date_max": _iso(c.index[-1])},
        curve=curve, curve_manifest=(copy.deepcopy(dict(curve_manifest))
                                     if curve_manifest is not None else None),
        assert_causal=assert_causal)
    bundle = build_two_sleeve_arrays(payload, c_ext.index[0], c_ext.index[-1])
    executor = TwoSleeveExecutor(config)
    _, detail = executor.sim_oracle(bundle)
    W = np.asarray(detail["combined_w"], dtype=np.float64)
    union = dict(bundle["union"])
    dates = _session_dates(union["timestamps"])
    if len(dates) != len(W) + 1 or dates[-SESSIONS_AHEAD] != ahead[0]:
        raise ForwardRunError("forward window misaligned with the appended sessions")
    lead = decision_lead_bars(config)
    conviction, sleeve_assets = {}, {}
    for s in executor.sleeve_names:
        cm = monthly_rebal_conviction(bundle[s]["timestamps"], bundle[s]["conviction_ary"])
        conviction[s] = [float(x) for x in cm[len(cm) - SESSIONS_AHEAD - 1 + lead]]
        sleeve_assets[s] = list(bundle[s]["assets"])
    sleeve_batch = {
        "weights": {s: np.asarray(detail["sleeve_traj"][s]["weights"], dtype=np.float64)
                    for s in executor.sleeve_names},
        "alphas": {s: np.asarray(detail["alphas"][s], dtype=np.float64)
                   for s in executor.sleeve_names},
        "assets": sleeve_assets,
    }
    return ForwardDecision(
        as_of=c.index[-1], fill_session=ahead[0], assets=list(union["assets"]),
        weights=W[-SESSIONS_AHEAD].copy(),
        alphas={s: float(np.asarray(a)[-SESSIONS_AHEAD]) for s, a in detail["alphas"].items()},
        sleeve_conviction=conviction, sleeve_assets=sleeve_assets,
        month_end_fill=tc.is_month_end(ahead[0]), data_sha16=frame_sha16(c, v),
        batch_weights=W, union=union, sleeve_batch=sleeve_batch)


def target_record(decision: ForwardDecision, config: Mapping, *, config_sha16: str = "") -> dict:
    """The log record of a forward decision (deterministic fields only; the log adds sequence,
    time and hashes)."""
    return {
        "kind": "target", "as_of": _iso(decision.as_of),
        "fill_session": _iso(decision.fill_session), "assets": list(decision.assets),
        "weights": [float(x) for x in decision.weights],
        "month_end_fill": bool(decision.month_end_fill), "alphas": decision.alphas,
        "sleeve_conviction": decision.sleeve_conviction,
        "sleeve_assets": decision.sleeve_assets,
        "execution_stamp": execution_stamp(config),
        "data_sha256_16": decision.data_sha16, "config_sha256_16": config_sha16,
    }


def flatten_record(as_of, assets: list[str], reason: str, config: Mapping,
                   *, config_sha16: str = "") -> dict:
    fill = tc.next_sessions(as_of, 1)[0]
    return {"kind": "flatten", "as_of": _iso(as_of), "fill_session": _iso(fill),
            "assets": list(assets), "weights": [0.0] * len(assets),
            "month_end_fill": bool(tc.is_month_end(fill)), "reason": reason,
            "execution_stamp": execution_stamp(config), "config_sha256_16": config_sha16}


# --------------------------------------------------------------------------- #
# The append-only, hash-chained target log
# --------------------------------------------------------------------------- #
class TargetLog:
    """Append-only JSONL of daily targets, each record hash-chained to the previous one.

    A record is never rewritten. Re-logging a fill session that already has a record is
    idempotent when the weights agree within ``_REPEAT_TOL_L1``. When they disagree, the first
    record stands and the difference is reported (the parity gate then judges it). The one
    exception is a ``flatten`` (a kill), which supersedes a target for the same session: the
    later record wins in :meth:`for_session`. The chain is verified when the file is first read
    (``verify`` re-reads it from disk); this object is the log's only writer during a run."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._recs: list[dict] | None = None
        self._by_session: dict[str, dict] = {}

    @staticmethod
    def _read(path: Path) -> list[dict]:
        if not path.exists():
            return []
        out, prev = [], _GENESIS
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            rec = json.loads(line)
            digest = rec.pop("hash", None)
            if rec.get("prev_hash") != prev or _sha(_canonical(rec)) != digest:
                raise TargetLogError(f"{path.name}: hash chain broken at line {n}")
            rec["hash"] = digest
            out.append(rec)
            prev = digest
        return out

    def _load(self) -> list[dict]:
        if self._recs is None:
            self._recs = self._read(self.path)
            self._by_session = {r["fill_session"]: r for r in self._recs}
        return self._recs

    def verify(self) -> bool:
        """Re-read the file and check the whole hash chain."""
        try:
            self._read(self.path)
            return True
        except TargetLogError:
            return False

    def records(self) -> list[dict]:
        return list(self._load())

    def for_session(self, session) -> dict | None:
        self._load()
        return self._by_session.get(_iso(session))

    def append(self, record: Mapping) -> tuple[dict, float | None]:
        """Append ``record``. Returns ``(stored_record, repeat_l1)``. ``repeat_l1`` is None for
        a new session, or the L1 gap to the existing record when the fill session was already
        logged (the existing record is kept, unless the new one is a flatten)."""
        recs = self._load()
        body = {k: v for k, v in record.items()
                if k not in ("seq", "logged_utc", "prev_hash", "hash")}
        old = self._by_session.get(body["fill_session"])
        if old is not None:
            gap = float(np.abs(np.asarray(old["weights"]) - np.asarray(body["weights"])).sum())
            if not (body["kind"] == "flatten" and old["kind"] != "flatten"):
                if gap > _REPEAT_TOL_L1:
                    log.warning("forward: %s re-decided with L1 %.3g off the logged target; the "
                                "logged record stands", body["fill_session"], gap)
                return old, gap
        elif recs and body["fill_session"] <= recs[-1]["fill_session"]:
            raise TargetLogError(f"target for {body['fill_session']} is older than the last "
                                 f"logged session {recs[-1]['fill_session']}")
        rec = {**body, "seq": len(recs),
               "logged_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               "prev_hash": recs[-1]["hash"] if recs else _GENESIS}
        rec["hash"] = _sha(_canonical(rec))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = _canonical(rec) + "\n"

        def _write() -> None:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line)
        _retry_io(_write)
        recs.append(rec)
        self._by_session[rec["fill_session"]] = rec
        return rec, None


# --------------------------------------------------------------------------- #
# The runner
# --------------------------------------------------------------------------- #
@dataclass
class RunnerState:
    first_as_of: str | None = None
    as_of: str | None = None
    marks: dict[str, float] = field(default_factory=dict)
    killed: dict | None = None
    max_rebase_abs: float = 0.0


class ForwardRunner:
    """Persisted forward book for one linear-core config (see the module docstring)."""

    def __init__(self, config: Mapping, gates_cfg: Mapping, state_dir: str | Path, *,
                 config_sha16: str = "") -> None:
        self.config, self.gates_cfg = config, gates_cfg
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.targets = TargetLog(self.state_dir / TARGET_LOG)
        self.config_sha16 = config_sha16
        self.harness = ParityHarness(config)
        self.engine = self.harness.default_fill_engine()
        risk = dict(dict(gates_cfg["paper_soak"])["risk"])
        self.max_dd_pct = float(risk["max_drawdown_kill_pct"])
        self.daily_halt_pct = float(risk["daily_loss_halt_pct"])
        self.max_gross = float(risk["max_gross_exposure"])
        kill = dict(config.get("safety") or {}).get("kill_file")
        self.kill_files = [Path(p) for p in (kill, self.state_dir / "KILL") if p]
        self.sleeve_names = TwoSleeveExecutor(config).sleeve_names
        self._drift: dict | None = None        # per-sleeve trackers, built lazily from the log
        self.last_live: LiveTrajectory | None = None     # the last evaluate(), for PaperMetrics
        self.last_parity: ParityReport | None = None

    # ---- persistence -------------------------------------------------------
    def _load(self, n_assets: int, assets: list[str]) -> tuple[RunnerState, PaperState]:
        path = self.state_dir / RUNNER_STATE
        state = RunnerState(**json.loads(path.read_text(encoding="utf-8"))) if path.exists() \
            else RunnerState()
        if (self.state_dir / "paper_state.parquet").exists():
            book = PaperState.load(self.state_dir)
            if book.assets != assets:
                raise ForwardRunError(f"persisted book assets {book.assets} != config {assets}")
        else:
            book = PaperState(n_assets=n_assets, initial_capital=self.harness.initial_capital,
                              assets=assets)
        return state, book

    def _save(self, state: RunnerState, book: PaperState) -> None:
        _retry_io(lambda: book.save(self.state_dir))
        tmp = self.state_dir / (RUNNER_STATE + ".tmp")
        _retry_io(lambda: tmp.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8"))
        _retry_io(lambda: tmp.replace(self.state_dir / RUNNER_STATE))

    def fills(self) -> list[dict]:
        path = self.state_dir / FILL_LOG
        if not path.exists():
            return []
        return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]

    def _append_fill(self, rec: Mapping) -> None:
        line = _canonical(rec) + "\n"

        def _write() -> None:
            with (self.state_dir / FILL_LOG).open("a", encoding="utf-8") as fh:
                fh.write(line)
        _retry_io(_write)

    # ---- the daily cycle ---------------------------------------------------
    def run_session(self, close: pd.DataFrame, volume: pd.DataFrame, *, as_of,
                    curve: Mapping | None = None, curve_manifest: Mapping | None = None,
                    assert_causal: bool = True) -> dict:
        """Decide, settle, check the kills, log, and score. Returns the session report, with
        ``exit_code`` set from the verdict."""
        decision = compute_forward_decision(self.config, close, volume, as_of=as_of, curve=curve,
                                            curve_manifest=curve_manifest,
                                            assert_causal=assert_causal)
        record = target_record(decision, self.config, config_sha16=self.config_sha16)
        report = self.step(as_of=decision.as_of, union=decision.union, record=record)
        report["verdict"] = self.evaluate(union=decision.union,
                                          batch_weights=decision.batch_weights,
                                          sleeve_batch=decision.sleeve_batch)
        report["exit_code"] = exit_code(report)
        return report

    def step(self, *, as_of, union: Mapping, record: Mapping | None) -> dict:
        """Settle every session since the last mark through ``as_of`` on ``union`` (prices,
        volumes and financing over a window that covers them), apply the kills, then log
        ``record`` (the next session's target) or a flatten."""
        as_of = pd.Timestamp(as_of).normalize()
        assets = list(union["assets"])
        dates = _session_dates(union["timestamps"])
        state, book = self._load(len(assets), assets)
        if state.as_of is not None and pd.Timestamp(state.as_of) > as_of:
            raise ForwardRunError(f"book is marked to {state.as_of}, after the as-of {as_of.date()}")
        report: dict = {"as_of": _iso(as_of), "settled": [], "kill": None, "logged": None,
                        "repeat_l1": None}
        if state.as_of is not None and pd.Timestamp(state.as_of) < as_of:
            self._rebase(book, state, union, dates)
            for session in dates[(dates > pd.Timestamp(state.as_of)) & (dates <= as_of)]:
                report["settled"].append(self._settle(book, session, union, dates))
        elif state.as_of is None:
            state.first_as_of = _iso(as_of)
        i = int(dates.get_loc(as_of))
        state.as_of = _iso(as_of)
        state.marks = {a: float(p) for a, p in zip(assets, union["price_ary"][i])}

        reason = state.killed["reason"] if state.killed else self._kill_reason(book, report, record)
        if reason and not state.killed:
            state.killed = {"reason": reason, "session": _iso(as_of)}
            log.error("forward: KILL at %s (%s): flattening at the next session", as_of.date(), reason)
        report["kill"] = state.killed
        if state.killed:
            if np.abs(book.positions).sum() > 0 or not self.targets.for_session(
                    tc.next_sessions(as_of, 1)[0]):
                rec, gap = self.targets.append(flatten_record(
                    as_of, assets, state.killed["reason"], self.config,
                    config_sha16=self.config_sha16))
                report["logged"] = {"seq": rec["seq"], "kind": rec["kind"],
                                    "fill_session": rec["fill_session"]}
        elif record is not None:
            if record["as_of"] != _iso(as_of) or record["assets"] != assets:
                raise ForwardRunError("target record does not belong to this session/universe")
            rec, gap = self.targets.append(record)
            report["logged"] = {"seq": rec["seq"], "kind": rec["kind"],
                                "fill_session": rec["fill_session"]}
            report["repeat_l1"] = gap
            if gap is None:
                self._observe_drift(rec)
        self._save(state, book)
        report["equity"] = float(book.portfolio_value(union["price_ary"][i]))
        return report

    def _rebase(self, book: PaperState, state: RunnerState, union: Mapping,
                dates: pd.DatetimeIndex) -> None:
        """Re-base entry prices when a refetch re-adjusted history (T1-10)."""
        last = pd.Timestamp(state.as_of)
        if last not in dates:
            raise ForwardRunError(f"the book's last session {state.as_of} is missing from the data")
        now = union["price_ary"][int(dates.get_loc(last))]
        for j, a in enumerate(book.assets):
            old = float(state.marks.get(a, 0.0))
            if old > 0.0 and now[j] > 0.0 and abs(now[j] / old - 1.0) > 1e-12:
                ratio = float(now[j] / old)
                book.entry_prices[j] *= ratio
                state.max_rebase_abs = max(state.max_rebase_abs, abs(ratio - 1.0))

    def _settle(self, book: PaperState, session: pd.Timestamp, union: Mapping,
                dates: pd.DatetimeIndex) -> dict:
        """Book one session at its close: the target logged for it, else a hold (missed)."""
        i = int(dates.get_loc(session))
        rec = self.targets.for_session(session)
        prev_price, price_now = union["price_ary"][i - 1], union["price_ary"][i]
        pv_before = book.pv_before(prev_price)
        if rec is None:
            delta = np.zeros(book.n_assets)
        else:
            delta = generate_orders(np.asarray(rec["weights"], dtype=np.float64), book.positions,
                                    min_trade_pct=0.0)
        fill = self.engine.fill(delta_weights=delta, ref_prices=price_now, pv_before=pv_before,
                                dollar_volume=union["volume_ary"][i - 1])
        borrow = union.get("borrow_ary")
        info = book.step_bar(delta_weights=delta, fill=fill, prev_price=prev_price,
                             price_now=price_now, carry_rates=union["carry_ary"][i],
                             pv_before=pv_before, as_of_ts=int(union["timestamps"][i]),
                             borrow_rates=None if borrow is None else borrow[i])
        out = {"session": _iso(session), "target_seq": None if rec is None else rec["seq"],
               "kind": "missed" if rec is None else rec["kind"],
               "month_end": bool(tc.is_month_end(session)), "pv_before": float(pv_before),
               "pv": float(info["portfolio_value"]), "step_return": float(info["step_return"]),
               "cost": float(info["cost"]), "carry": float(info["carry"]),
               "turnover": float(info["turnover"]), "gross": float(info["gross_exposure"]),
               "net": float(info["net_exposure"]),
               "notional_gross": float(info["notional_gross"]),
               "positions": [float(x) for x in book.positions],
               "cumulative_fees": float(book.cumulative_fees)}
        self._append_fill(out)
        return out

    def _kill_reason(self, book: PaperState, report: Mapping, record: Mapping | None) -> str | None:
        for path in self.kill_files:
            if path.exists():
                return f"kill_file {path}"
        settled = report["settled"]
        if settled:
            last = settled[-1]
            dd_pct = (1.0 - last["pv"] / max(book.peak_equity, 1e-12)) * 100.0
            if dd_pct >= self.max_dd_pct:
                return f"drawdown {dd_pct:.2f}% >= {self.max_dd_pct}%"
            loss_pct = -min(s["step_return"] for s in settled) * 100.0
            if loss_pct >= self.daily_halt_pct:
                return f"session loss {loss_pct:.2f}% >= {self.daily_halt_pct}%"
        if record is not None:
            gross = float(np.abs(record["weights"]).sum())
            if gross > self.max_gross:
                return f"target gross {gross:.3f} > {self.max_gross}"
        crit = [s for s, r in (self.drift_reports() or {}).items()
                if isinstance(r, dict) and r.get("status") == "CRIT"]
        safe = dict(dict(self.config.get("gates") or {}).get("safe_mode") or {})
        if crit and safe.get("crit_triggers_flatten"):
            return f"action drift CRIT on {crit} (gates.safe_mode.crit_triggers_flatten)"
        return None

    # ---- drift -------------------------------------------------------------
    def _build_drift(self) -> dict:
        """One ``ActionDriftTracker`` per sleeve, fed the held conviction of every logged
        target, when the config declares ``drift.enabled`` (thresholds from ``gates.drift``,
        as the RL live engine resolves them)."""
        dcfg = dict(self.config.get("drift") or {})
        if not dcfg.get("enabled"):
            return {"reports": None}
        path = Path(dcfg.get("baseline_path", ""))
        path = path if path.is_absolute() else ROOT / path
        if not path.exists():
            return {"reports": {"_status": UNKNOWN, "_reason": f"drift baseline missing: {path}"}}
        from sharpen.monitoring.action_drift import ActionDriftTracker

        by_sleeve = json.loads(path.read_text(encoding="utf-8"))["eval_distribution_by_sleeve"]
        gd = dict(dict(self.config.get("gates") or {}).get("drift") or {})

        def pick(key, default):
            return gd.get(key, dcfg.get(key, default))

        trackers, keys = {}, {}
        for s in self.sleeve_names:
            base = by_sleeve.get(s)
            if base is None:
                return {"reports": {"_status": UNKNOWN, "_reason": f"baseline has no sleeve {s!r}"}}
            keys[s] = list(base.get("by_asset", {}).keys())
            trackers[s] = ActionDriftTracker(
                baseline=base, window_bars=int(pick("window_bars", 252)),
                min_bars_before_check=int(pick("min_bars_before_check", 63)),
                deadband_warn=float(pick("deadband_frac_warn", 0.15)),
                deadband_crit=float(pick("deadband_frac_crit", 0.30)),
                saturation_warn=float(pick("saturation_frac_warn", 0.15)),
                saturation_crit=float(pick("saturation_frac_crit", 0.30)),
                action_kl_warn=float(pick("action_kl_warn", 0.5)),
                action_kl_crit=float(pick("action_kl_crit", 1.0)),
                asset_keys=keys[s] or None)
        self._drift = {"trackers": trackers, "keys": keys, "reports": {}}
        for rec in self.targets.records():
            self._observe_drift(rec)
        return self._drift

    def _observe_drift(self, rec: Mapping) -> None:
        state = self._drift
        if not state or not state.get("trackers") or rec.get("kind") != "target":
            return
        for s, vec in rec["sleeve_conviction"].items():
            keys = state["keys"][s]
            if keys:
                by_asset = dict(zip(rec["sleeve_assets"][s], vec))
                vec = [by_asset[a] for a in keys]
            state["reports"][s] = state["trackers"][s].observe(vec, None).to_dict()

    def drift_reports(self) -> dict | None:
        if self._drift is None:
            self._drift = self._build_drift()
        return self._drift["reports"]

    # ---- verdict -----------------------------------------------------------
    def evaluate(self, *, union: Mapping, batch_weights: np.ndarray,
                 sleeve_batch: Mapping | None = None) -> dict:
        """Score the booked sessions: incremental parity against today's batch
        recomputation, then the pre-registered ``paper_soak`` gates. ``sleeve_batch``
        (``ForwardDecision.sleeve_batch``) supplies the per-sleeve attribution that the
        soak's sleeve_attribution check requires; without it that check fails closed."""
        fills = self.fills()
        chain_ok = self.targets.verify()
        if not fills:
            return {"overall_status": UNKNOWN, "reason": "no booked session yet",
                    "target_log_chain_ok": chain_ok}
        dates = _session_dates(union["timestamps"])
        idx = np.array([dates.get_loc(pd.Timestamp(f["session"])) for f in fills])
        if not np.array_equal(idx, np.arange(idx[0], idx[0] + len(idx))):
            raise ForwardRunError("booked sessions are not contiguous in the data window")
        i0, i1 = int(idx[0]) - 1, int(idx[-1])
        W_book = np.array([f["positions"] for f in fills], dtype=np.float64)
        W_batch = np.asarray(batch_weights, dtype=np.float64)[i0:i1]
        window = {k: (np.asarray(v)[i0:i1 + 1] if k in ("price_ary", "volume_ary", "carry_ary",
                                                       "borrow_ary", "timestamps") else v)
                  for k, v in union.items() if v is not None}
        sim = self.harness._replay(window, W_batch, fill_engine=None)
        r_book = np.array([f["step_return"] for f in fills])
        l1 = np.abs(W_book - W_batch).sum(axis=1)
        te = np.abs(r_book - sim.step_returns) * 1e4
        sim_cost = float(sim.cumulative_fees[-1]) if sim.n_steps else 0.0
        book_cost = float(fills[-1]["cumulative_fees"])
        cost_ratio = (book_cost / sim_cost if sim_cost > 1e-12
                      else (1.0 if book_cost <= 1e-12 else float("inf")))
        missed = sum(1 for f in fills if f["month_end"] and f["kind"] != "target")
        parity = ParityReport(
            weight_l1_drift_max=float(l1.max()), weight_l1_drift_mean=float(l1.mean()),
            daily_return_te_bps_mean=float(te.mean()), daily_return_te_bps_max=float(te.max()),
            missed_rebalances=int(missed), cost_drift_ratio=float(cost_ratio), n_steps=len(fills))
        assets, asset_class = self.harness._asset_meta(window, W_book.shape[1])
        class_pnl, spy = self.harness._attribution(W_book, window["price_ary"], assets, asset_class)
        # Per-sleeve P&L of the batch the book is parity-checked against. It decomposes the
        # booked book exactly while incremental parity holds (audit T4-12b).
        sleeve_pnl = None
        if sleeve_batch:
            sleeve_pnl = TwoSleeveExecutor._sleeve_attribution(
                {s: np.asarray(w)[i0:i1] for s, w in sleeve_batch["weights"].items()},
                sleeve_batch["assets"],
                {s: np.asarray(a)[i0:i1] for s, a in sleeve_batch["alphas"].items()},
                assets, window["price_ary"])
        # Actual notional gross (audit T4-12a). Fills booked before it was recorded lack it; the
        # gross kill is then read on the labels, and the verdict says so.
        ng = ([f["notional_gross"] for f in fills]
              if all("notional_gross" in f for f in fills) else None)
        pv = np.array([fills[0]["pv_before"]] + [f["pv"] for f in fills])
        live = LiveTrajectory(
            weights=W_book, equity_curve=pv, step_returns=r_book,
            turnovers=np.array([f["turnover"] for f in fills]),
            cumulative_fees=np.array([f["cumulative_fees"] for f in fills]),
            gross_exposure=np.array([f["gross"] for f in fills]),
            net_exposure=np.array([f["net"] for f in fills]),
            notional_gross=None if ng is None else np.array(ng, dtype=np.float64),
            timestamps=np.asarray(window["timestamps"][1:], dtype=np.int64),
            class_pnl=class_pnl, assets=assets, asset_class=asset_class,
            initial_capital=float(pv[0]), spy_returns=spy, sleeve_pnl=sleeve_pnl)
        try:
            verdict = evaluate_paper_soak_gates(live, parity, self.gates_cfg,
                                                executor_sleeves=list(self.sleeve_names))
        except (KeyError, TypeError, ValueError) as e:
            # A gates file the soak evaluator cannot read is a FAIL, never a silent pass. The
            # challenge gates lack paper_soak.performance keys (audit T6-05). Parity and drift
            # are still reported below.
            verdict = {"overall_status": FAIL, "groups": {},
                       "error": f"paper_soak gates not evaluable: {type(e).__name__}: {e}"}
        self.last_live, self.last_parity = live, parity
        drift = self.drift_reports()
        verdict["forward"] = {
            "target_log_chain_ok": chain_ok,
            "booked_sessions": len(fills), "first_session": fills[0]["session"],
            "last_session": fills[-1]["session"],
            "missed_sessions": [f["session"] for f in fills if f["kind"] == "missed"],
            "action_drift": drift,
            "incremental_parity": parity.as_dict(),
        }
        return verdict


def exit_code(report: Mapping) -> int:
    """PASS 0, REVIEW 3, FAIL 1 (roadmap N3). A too-short soak with every hard group evaluable
    exits 0. Anything that stops the book (kill, broken log) exits 1."""
    verdict = report.get("verdict") or {}
    if report.get("kill") or not (verdict.get("forward") or {}).get("target_log_chain_ok", True):
        return EXIT_FAIL
    status = verdict.get("overall_status")
    if status == FAIL:
        return EXIT_FAIL
    groups = verdict.get("groups") or {}
    if any(groups.get(g, {}).get("status") == UNKNOWN for g in ("parity", "risk")):
        return EXIT_FAIL
    drift = (verdict.get("forward") or {}).get("action_drift") or {}
    drift_bad = drift.get("_status") == UNKNOWN or any(
        isinstance(r, dict) and r.get("status") in ("WARN", "CRIT") for r in drift.values())
    if status == "REVIEW" or drift_bad:
        return EXIT_REVIEW
    return EXIT_PASS
