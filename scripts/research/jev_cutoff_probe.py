"""Phase-0 C1 — empirical knowledge cutoff of Jev (ATL x Jev plan §3.2).

WHY. Jev's training cutoff is undisclosed and it was released 2026-09-15. If it remembers what stocks
did in a month, any backtest over that month can be answered from memory instead of from the input —
the LLM form of LEAK-2. This probe measures the month where that memory stops (T_c); only data after
T_c + embargo can certify anything (plan leg P4).

HOW. For every (security, month) in [start_month, end_month] Jev gets one ``noul`` question: "did X
close HIGHER (or LOWER) at the end of month m than at the end of m-1?". Polarity is drawn at random per
item, so a model that always says "yes" scores exactly 0.5. Truth comes from split-adjusted Alpaca
monthly bars. A two-segment binomial changepoint on item accuracy gives T_c; resampling SECURITIES with
replacement gives its CI. Knowledge counts as detected only if the pre-T_c accuracy's 95% Wilson lower
bound clears 0.5 + ``known_margin``.

LIMITATION (audit 2026-09-23, found after the run). Polarity does NOT neutralize a constant DRIFT
prior: a model that only "knows stocks usually rise" answers "higher" yes and "lower" no, so it scores
the period's share of up-months (0.42-0.69 by year here), and the accuracy changepoint can then fire on
the market's own up/down years with no memory at all. The exploratory month-level statistic below —
correlation of Jev's mean implied P(up) with the realized share of risers — is immune to any constant
prior (zero correlation by construction) and is the statistic the adopted T_c rests on.

All thresholds come from ``configs/atl_jev.gates.yaml`` (sha256 stamped into the artifact). Answers
are cached (``results/atl_jev/jev_cache.sqlite``), so a re-run is free and exactly reproducible.

    python scripts/research/jev_cutoff_probe.py [--limit-tickers N] [--dry-run]
"""
from __future__ import annotations

import argparse
import calendar
import csv
import hashlib
import json
import logging
import math
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sharpen.jev import JevClient  # noqa: E402

GATES = ROOT / "configs" / "atl_jev.gates.yaml"
OUT_DIR = ROOT / "results" / "atl_jev" / "phase0"
CACHE = ROOT / "results" / "atl_jev" / "jev_cache.sqlite"
EQ = ROOT / "data" / "raw" / "equity_panel"
# ATL's environment universe (dashboard/backend/infrastructure/llm/validator.py::DJIA_30, 2026-09-23).
DJIA_30 = ("AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "GOOGL", "GS", "HD",
           "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT", "NKE", "NVDA", "PG", "SHW",
           "TRV", "UNH", "V", "WMT")
STATE = ("You are answering factual questions about historical U.S. stock-market prices from your own "
         "knowledge. Each question names one security and one calendar month; prices are split-adjusted. "
         "If you do not know the answer, give a probability near 0.5.")

log = logging.getLogger("jev_cutoff_probe")


def _months(start: str, end: str) -> list[str]:
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    out = []
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def _prev(month: str) -> str:
    y, m = map(int, month.split("-"))
    return f"{y - 1:04d}-12" if m == 1 else f"{y:04d}-{m - 1:02d}"


def _label(month: str) -> str:
    y, m = map(int, month.split("-"))
    return f"{calendar.month_name[m]} {y}"


def universe(extra: int) -> list[tuple[str, str]]:
    names: dict[str, str] = {}
    with open(EQ / "sp500_constituents.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            names[r["Symbol"].replace(".", "-")] = r["Security"]
    out = [("SPY", "SPDR S&P 500 ETF Trust")] + [(t, names.get(t, t)) for t in DJIA_30]
    seen_names = {n for _, n in out}
    with open(EQ / "sp500_mktcap_rank.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            t = r["ticker"]
            n = names.get(t)
            if n is None or n in seen_names or t in dict(out):
                continue                               # GOOG duplicates GOOGL, etc.
            out.append((t, n))
            seen_names.add(n)
            if len(out) >= 1 + len(DJIA_30) + extra:
                break
    return out


def monthly_closes(tickers: list[str], start: str, end: str) -> dict[tuple[str, str], float]:
    """Split-adjusted month-end closes from Alpaca 1Month SIP bars: {(ticker, 'YYYY-MM'): close}."""
    hdr = {"APCA-API-KEY-ID": os.environ["ALPACA_API_KEY"],
           "APCA-API-SECRET-KEY": os.environ["ALPACA_SECRET_KEY"]}
    alp = {t: t.replace("-", ".") for t in tickers}
    back = {v: k for k, v in alp.items()}
    out: dict[tuple[str, str], float] = {}
    token = None
    while True:
        q = {"symbols": ",".join(alp.values()), "timeframe": "1Month", "start": f"{start}-01",
             "end": f"{end}-28", "adjustment": "split", "feed": "sip", "limit": 10000}
        if token:
            q["page_token"] = token
        url = "https://data.alpaca.markets/v2/stocks/bars?" + urllib.parse.urlencode(q)
        with urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=60) as r:
            d = json.loads(r.read().decode())
        for sym, bars in (d.get("bars") or {}).items():
            for b in bars:
                out[(back.get(sym, sym), b["t"][:7])] = float(b["c"])
        token = d.get("next_page_token")
        if not token:
            return out


def build_items(univ, months, closes, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    items = []
    for ticker, name in univ:
        for m in months:
            c1, c0 = closes.get((ticker, m)), closes.get((ticker, _prev(m)))
            pol = "higher" if rng.random() < 0.5 else "lower"   # drawn for EVERY cell: stable per seed
            if c1 is None or c0 is None or c1 == c0:
                continue
            up = c1 > c0
            items.append({
                "ticker": ticker, "month": m, "polarity": pol, "ret": c1 / c0 - 1.0,
                "truth": bool(up if pol == "higher" else not up),
                "question": {"type": "noul", "instructions": (
                    f"Did shares of {name} ({ticker}) close {pol} on the last trading day of "
                    f"{_label(m)} than on the last trading day of {_label(_prev(m))}?")}})
    return items


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def changepoint(month_idx: np.ndarray, correct: np.ndarray, n_months: int, min_seg: int) -> dict:
    """Two-segment binomial MLE over item-level accuracy; split s = first month of the late segment."""
    k = np.bincount(month_idx, weights=correct, minlength=n_months)
    n = np.bincount(month_idx, minlength=n_months).astype(float)

    def ll(kk, nn):
        if nn <= 0:
            return 0.0
        a = min(max(kk / nn, 1e-9), 1 - 1e-9)
        return kk * math.log(a) + (nn - kk) * math.log(1 - a)

    best = None
    for s in range(min_seg, n_months - min_seg + 1):
        k1, n1, k2, n2 = k[:s].sum(), n[:s].sum(), k[s:].sum(), n[s:].sum()
        v = ll(k1, n1) + ll(k2, n2)
        if best is None or v > best[0]:
            best = (v, s, k1, n1, k2, n2)
    _, s, k1, n1, k2, n2 = best
    return {"split": int(s), "acc_early": k1 / n1, "n_early": int(n1), "k_early": float(k1),
            "acc_late": k2 / n2, "n_late": int(n2), "k_late": float(k2)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--limit-tickers", type=int, default=None, help="debug: first N securities only")
    ap.add_argument("--dry-run", action="store_true", help="build items and stop before calling Jev")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv(ROOT / ".env")

    graw = GATES.read_bytes()
    gates = yaml.safe_load(graw)
    g, jcfg = gates["phase0"]["cutoff_probe"], gates["jev"]
    if g["changepoint"] != "two_segment_binomial_mle":       # the declaration binds: no silent method swap
        raise SystemExit(f"unsupported changepoint method {g['changepoint']!r}")
    months = _months(g["start_month"], g["end_month"])
    univ = universe(int(g["extra_top_names"]))
    if args.limit_tickers:
        univ = univ[: args.limit_tickers]
    closes = monthly_closes([t for t, _ in univ], _prev(months[0]), months[-1])
    items = build_items(univ, months, closes, int(g["polarity_seed"]))
    have = {t for t, _ in univ if any((t, m) in closes for m in months)}
    log.info("%d securities (%d with bars), %d months, %d items", len(univ), len(have), len(months), len(items))
    if args.dry_run:
        return 0

    client = JevClient(model=jcfg["model"], cache_path=CACHE,
                       max_questions_per_request=int(jcfg["max_questions_per_request"]),
                       max_retries=int(jcfg["max_retries"]))
    step = int(jcfg["max_questions_per_request"])
    jobs = [(STATE, {f"i{j}": items[j]["question"] for j in range(s, min(s + step, len(items)))})
            for s in range(0, len(items), step)]
    t0 = time.monotonic()
    results = client.ask_many(jobs, concurrency=int(jcfg["concurrency"]))
    wall = time.monotonic() - t0
    for res in results:
        for qid, ans in res.items():
            items[int(qid[1:])]["p_yes"] = float(ans.value)

    p = np.array([it["p_yes"] for it in items])
    truth = np.array([it["truth"] for it in items])
    correct = np.where(p == 0.5, 0.5, ((p > 0.5) == truth).astype(float))   # a tie is half-right
    midx = np.array([months.index(it["month"]) for it in items])
    tick = np.array([it["ticker"] for it in items])

    cp = changepoint(midx, correct, len(months), int(g["min_segment_months"]))
    lo_early, hi_early = wilson(cp["k_early"], cp["n_early"])
    lo_late, hi_late = wilson(cp["k_late"], cp["n_late"])
    known = bool(lo_early > 0.5 + float(g["known_margin"]))

    rng = np.random.default_rng(int(g["polarity_seed"]) + 1)
    uniq = np.unique(tick)
    by_t = {t: np.flatnonzero(tick == t) for t in uniq}
    boot = []
    for _ in range(int(g["bootstrap_reps"])):
        idx = np.concatenate([by_t[t] for t in rng.choice(uniq, size=uniq.size, replace=True)])
        boot.append(changepoint(midx[idx], correct[idx], len(months), int(g["min_segment_months"]))["split"])
    b_lo, b_hi = (int(np.percentile(boot, q)) for q in (5, 95))

    t_c = months[cp["split"]] if known else None
    clean_from = None
    if t_c is not None:
        y, m = map(int, t_c.split("-"))
        m += int(g["embargo_months"])
        y, m = y + (m - 1) // 12, (m - 1) % 12 + 1
        clean_from = f"{y:04d}-{m:02d}"

    per_month = []
    for i, mo in enumerate(months):
        sel = midx == i
        if not sel.any():
            continue
        big = sel & (np.abs([it["ret"] for it in items]) > 0.05)
        per_month.append({"month": mo, "n": int(sel.sum()), "acc": round(float(correct[sel].mean()), 4),
                          "acc_big_moves": None if not big.any() else round(float(correct[big].mean()), 4),
                          "mean_conf": round(float(np.abs(p[sel] - 0.5).mean() * 2), 4),
                          "brier": round(float(((p[sel] - truth[sel]) ** 2).mean()), 4)})
    per_year = {}
    for y in sorted({mo[:4] for mo in months}):
        sel = np.array([months[i].startswith(y) for i in midx])
        k = float(correct[sel].sum())
        per_year[y] = {"n": int(sel.sum()), "acc": round(k / sel.sum(), 4),
                       "wilson95": [round(x, 4) for x in wilson(k, int(sel.sum()))]}

    # EXPLORATORY (added after the first run, NOT a pre-registered rule): item accuracy is diluted by
    # months where the cross-section split ~50/50, so it under-detects memory of MARKET REGIMES. The
    # sharper statistic is month-level: does Jev's implied P(up), averaged over securities, move WITH
    # the realized share of securities that rose? Reported by year; the adopted T_c is decided in the
    # Phase-0 summary with this block as evidence, and only ever in the conservative direction.
    p_up = np.where(np.array([it["polarity"] for it in items]) == "higher", p, 1 - p)
    up = np.array([it["ret"] > 0 for it in items], dtype=float)
    m_imp = np.bincount(midx, weights=p_up, minlength=len(months)) / np.maximum(np.bincount(midx, minlength=len(months)), 1)
    m_real = np.bincount(midx, weights=up, minlength=len(months)) / np.maximum(np.bincount(midx, minlength=len(months)), 1)
    regime = {}
    for y in sorted({mo[:4] for mo in months}):
        sel = np.array([mo.startswith(y) for mo in months])
        directional = sel & ((m_real <= 0.30) | (m_real >= 0.70))
        agree = np.sign(m_imp[directional] - 0.5) == np.sign(m_real[directional] - 0.5)
        regime[y] = {"corr_implied_vs_realized": round(float(np.corrcoef(m_imp[sel], m_real[sel])[0, 1]), 3),
                     "mean_abs_lean": round(float(np.abs(m_imp[sel] - 0.5).mean()), 4),
                     "directional_months": int(directional.sum()), "lean_agrees": int(agree.sum())}

    record = {
        "probe": "C1 knowledge cutoff", "gates_file": str(GATES.relative_to(ROOT)),
        "gates_sha256": hashlib.sha256(graw).hexdigest(), "gates": g,
        "n_securities": len(univ), "n_months": len(months), "n_items": len(items),
        "overall_acc": round(float(correct.mean()), 4), "wall_s": round(wall, 1),
        "jev_usage": client.usage.to_json(float(jcfg["price_usd_per_m_input"])),
        "changepoint": {**{k: (round(v, 4) if isinstance(v, float) else v) for k, v in cp.items()},
                        "split_month": months[cp["split"]], "wilson95_early": [round(lo_early, 4), round(hi_early, 4)],
                        "wilson95_late": [round(lo_late, 4), round(hi_late, 4)],
                        "bootstrap_split_month_p05_p95": [months[b_lo], months[b_hi]]},
        "decision": {"knowledge_detected": known, "T_c": t_c, "clean_window_from": clean_from},
        "per_year": per_year, "exploratory_regime_memory_by_year": regime, "per_month": per_month,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "cutoff_probe.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    rates = OUT_DIR / "jev_rates_cutoff.json"
    prev = json.loads(rates.read_text(encoding="utf-8"))["requests"] if rates.exists() else 0
    if client.usage.requests > prev:     # keep the LARGEST live measurement; cached re-runs never overwrite
        rates.write_text(json.dumps(
            {"source": "live run", "at": time.strftime("%Y-%m-%dT%H:%M:%S"), **record["jev_usage"]},
            indent=1), encoding="utf-8")
    with open(OUT_DIR / "cutoff_probe_items.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ticker", "month", "polarity", "ret", "truth", "p_yes"])
        for it in items:
            w.writerow([it["ticker"], it["month"], it["polarity"], f"{it['ret']:.5f}", int(it["truth"]),
                        f"{it['p_yes']:.4f}"])
    log.info("overall acc %.3f | split %s early %.3f (n=%d) late %.3f (n=%d) | known=%s T_c=%s | %s",
             record["overall_acc"], months[cp["split"]], cp["acc_early"], cp["n_early"], cp["acc_late"],
             cp["n_late"], known, t_c, json.dumps(record["jev_usage"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
