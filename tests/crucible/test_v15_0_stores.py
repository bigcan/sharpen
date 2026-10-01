"""crucible-v15.0 tripwires for the poll-and-accumulate data stores (deep audit 2026-09-30).

* TAIFEX / T86: an unreadable store was treated as EMPTY and the next save overwrote it — poll-only
  history (TAIFEX cannot be re-downloaded) destroyed by one torn write. TAIFEX's save was also a
  non-atomic ``write_text`` (truncate-then-write). Now: atomic saves, and a corrupt file is moved aside.
* T86: any ``stat != OK`` reply was persisted as a permanent NON-TRADING day — including "not yet
  published" for today (the backfill's default end) — and never re-polled. The live store held 58 real
  trading days recorded empty. Now: recent no-data replies are not persisted; a repair switch re-polls
  weekday empties.
"""
from __future__ import annotations

import json

import numpy as np

from sharpen.crucible.data.taifex_positioning import TaifexPositioningConnector
from sharpen.crucible.data.twse_institutional import TwseInstitutionalConnector

from test_taifex_positioning import _snapshot, _transport_for as _taifex_transport  # noqa: E402
from test_twse_institutional import _row, _t86_payload  # noqa: E402


def test_taifex_corrupt_store_is_quarantined_not_overwritten(tmp_path) -> None:
    store = tmp_path / "store.json"
    torn = "{\"TX|20260601\": {\"large_trader_net_pct\": 1.0}, \"TX|2026"   # torn mid-write
    store.write_text(torn, encoding="utf-8")
    conn = TaifexPositioningConnector(transport=_taifex_transport(_snapshot("20260703")),
                                      contracts=("TX",), store_path=store)
    data = conn.fetch(conn.discover()[0], "2026-06-01", "2026-07-15")      # the tick still proceeds
    assert data.n_obs == 1
    quarantined = list(tmp_path.glob("store.json.corrupt-*"))
    assert len(quarantined) == 1 and quarantined[0].read_text(encoding="utf-8") == torn
    json.loads(store.read_text(encoding="utf-8"))                         # the new store is valid
    assert not list(tmp_path.glob("*.tmp"))                               # atomic: no stray temp


def _t86(tmp_path, days: dict, *, today: str, repoll: bool = False) -> TwseInstitutionalConnector:
    calls: list[str] = []

    def transport(url: str) -> dict:
        d = url.split("date=")[1].split("&")[0]
        calls.append(d)
        return days.get(d, {"stat": "很抱歉，沒有符合條件的資料!"})        # TWSE's no-data reply
    conn = TwseInstitutionalConnector(transport=transport, tickers=("2330",),
                                      store_path=tmp_path / "t86.json", repoll_weekday_empty=repoll,
                                      today=lambda: np.datetime64(today, "D"), max_lookback_days=60)
    conn.calls = calls   # type: ignore[attr-defined]
    return conn


def test_t86_recent_no_data_reply_is_not_a_permanent_holiday(tmp_path) -> None:
    ok = {"20260910": _t86_payload("20260910", [_row("2330", "x", "1", "2", "3", "6")])}
    conn = _t86(tmp_path, ok, today="2026-09-11")
    ref = next(r for r in conn.discover() if r.series_id == "2330:foreign_net")
    conn.fetch(ref, "2026-09-09", "2026-09-11")
    stored = json.loads((tmp_path / "t86.json").read_text(encoding="utf-8"))
    assert "20260911" not in stored            # today: "no data yet" is NOT recorded as a holiday
    assert "20260910" in stored and stored["20260910"]
    # an OLD no-data day is still recorded as polled (a genuine non-trading day)
    conn2 = _t86(tmp_path, {}, today="2026-10-30")
    conn2.fetch(ref, "2026-09-05", "2026-09-06")                          # a weekend, long ago
    stored = json.loads((tmp_path / "t86.json").read_text(encoding="utf-8"))
    assert stored.get("20260905") == {} and stored.get("20260906") == {}


def test_t86_repair_switch_repolls_weekday_empties_only(tmp_path) -> None:
    store = tmp_path / "t86.json"
    store.write_text(json.dumps({"20260902": {}, "20260905": {}}), encoding="utf-8")   # Wed, Sat
    ok = {"20260902": _t86_payload("20260902", [_row("2330", "x", "5", "0", "0", "5")])}
    conn = _t86(tmp_path, ok, today="2026-10-30", repoll=True)
    ref = next(r for r in conn.discover() if r.series_id == "2330:foreign_net")
    data = conn.fetch(ref, "2026-09-02", "2026-09-05")
    assert "20260902" in conn.calls and "20260905" not in conn.calls     # weekday re-polled, weekend kept
    assert data.n_obs == 1
