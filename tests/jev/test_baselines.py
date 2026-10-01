"""P3's non-LLM baselines (architecture §5, ADR-9): the pinned Loughran–McDonald loader fails closed, net tone and
prior-release similarity follow their definitions on earnings releases only, and the daily baseline signal sits on
exactly the release rows and hold window the Jev signal uses."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from sharpen.jev.baselines import (
    Lexicon,
    baseline_scores,
    baseline_signal,
    cosine,
    lm_net_tone,
    load_lm_lexicon,
    prior_similarity,
)
from sharpen.jev.release import load_release_rule
from sharpen.signals.features import make_synthetic_panel
from sharpen.signals.library.jev_filings import Construction, FilingScores, JevFilingSignal

ROOT = Path(__file__).resolve().parents[2]
PHASE1 = yaml.safe_load((ROOT / "configs" / "atl_jev.gates.yaml").read_text(encoding="utf-8"))["phase1"]
_LM = ("Word,Seq_num,Negative,Positive,Uncertainty\n"
       "LOSS,1,2009,0,0\nDECLINE,2,2009,0,0\nGAIN,3,0,2009,0\nSTRONG,4,0,2009,0\n"
       "ABANDONED,5,-2020,0,0\nNEUTRAL,6,0,0,0\n")


def _lexfile(tmp_path, text=_LM):
    p = tmp_path / "lm.csv"
    p.write_bytes(text.encode("utf-8"))
    return p, hashlib.sha256(p.read_bytes()).hexdigest()


def test_lexicon_loader_fails_closed(tmp_path):
    p, sha = _lexfile(tmp_path)
    with pytest.raises(ValueError, match="not pinned"):
        load_lm_lexicon(p, sha256=None)
    with pytest.raises(FileNotFoundError, match="operator-supplied"):
        load_lm_lexicon(tmp_path / "absent.csv", sha256=sha)
    with pytest.raises(ValueError, match="not the registered"):
        load_lm_lexicon(p, sha256="0" * 64)
    lex = load_lm_lexicon(p, sha256=sha)
    assert lex.positive == {"GAIN", "STRONG"}
    assert lex.negative == {"LOSS", "DECLINE"}, "a removed word (negative year) must not count"


def test_the_repo_pin_is_unset_so_p3_refuses_until_the_operator_file_is_pinned():
    cfg = yaml.safe_load((ROOT / "configs" / "atl_jev_baselines.yaml").read_text(encoding="utf-8"))["lm_lexicon"]
    assert cfg["release"] == "2026-03" and cfg["path"].startswith("data/lexicons/")
    if cfg["sha256"] is None:
        with pytest.raises(ValueError, match="not pinned"):
            load_lm_lexicon(ROOT / cfg["path"], sha256=cfg["sha256"])


def test_net_tone_is_positive_minus_negative_over_words():
    lex = Lexicon(frozenset({"GAIN", "STRONG"}), frozenset({"LOSS"}), "x")
    # 7 words: STRONG GAIN SMALL LOSS IN COMPANY UNITS
    assert lm_net_tone("Strong gain, small loss in [COMPANY] units.", lex) == pytest.approx((2 - 1) / 7)
    assert math.isnan(lm_net_tone("  12 % ", lex))


def test_cosine_limits():
    assert cosine({"A": 2, "B": 1}, {"A": 2, "B": 1}) == pytest.approx(1.0)
    assert cosine({"A": 1}, {"B": 3}) == 0.0
    assert math.isnan(cosine({}, {"A": 1}))


def _corpus():
    rows = [("a1", 1, "2019-01-30 21:00", True, "revenue grew strongly"),
            ("e1", 1, "2019-02-10 14:00", False, "Item 1.02 termination"),
            ("a2", 1, "2019-04-30 21:00", True, "revenue grew strongly again"),
            ("b1", 2, "2019-02-01 21:00", True, "revenue grew strongly"),      # another CIK: must not chain to a1
            ("b2", 2, "2019-05-01 21:00", True, "costs fell sharply")]
    return pd.DataFrame({"accession": [r[0] for r in rows], "cik": [r[1] for r in rows],
                         "tickers": ["AAA", "AAA", "AAA", "BBB", "BBB"],
                         "accepted_utc": pd.to_datetime([r[2] for r in rows]),
                         "earnings": [r[3] for r in rows], "text": [r[4] for r in rows]})


def test_prior_similarity_chains_earnings_releases_within_a_company_only():
    c = _corpus()
    s = prior_similarity(c)
    assert math.isnan(s[0]) and math.isnan(s[3]), "a company's first release has no prior"
    assert math.isnan(s[1]), "event filings carry no similarity"
    assert s[2] == pytest.approx(3 / math.sqrt(3 * 4)), "a2 vs a1, skipping the event filing"
    assert s[4] == 0.0, "b2 vs b1, not vs a CIK-1 release"


def test_baselines_are_nan_on_event_filings():
    lex = Lexicon(frozenset({"STRONGLY"}), frozenset({"FELL"}), "x")
    b = baseline_scores(_corpus(), lex)
    assert math.isnan(b.lm_tone[1]) and math.isnan(b.similarity[1])
    assert b.lm_tone[0] == pytest.approx(1 / 3) and b.lm_tone[4] == pytest.approx(-1 / 3)


def test_the_baseline_signal_sits_on_the_jev_signals_release_rows_and_window():
    panel = make_synthetic_panel(T=200, N=3, seed=1)
    cal = panel.dates.astype("datetime64[D]")
    rule, cons = load_release_rule(PHASE1), Construction.from_phase1(PHASE1)
    # accepted at 21:00Z (after the 15:30 ET cutoff in winter and summer) on rows 10 and 80 -> released rows 11, 81
    acc = [np.datetime64(str(cal[r]) + "T21:00", "ns") for r in (10, 80)]
    scores = pd.DataFrame({"accession": ["x1", "x2"], "cik": [1, 1], "tickers": [panel.tickers[0]] * 2,
                           "accepted_utc": acc, "lm_tone": [0.02, -0.01], "similarity": [np.nan, 0.9]})
    base = baseline_signal("lm", scores, "lm_tone", 21, cons, cal, rule).compute(panel)
    jev = JevFilingSignal("jev", ("surprise",), 21,
                          FilingScores(np.array([panel.tickers[0]] * 2), np.array(acc), np.array([0.5, -0.5]),
                                       np.full(2, np.nan), np.full(2, np.nan)), cons, cal, rule).compute(panel)
    assert np.array_equal(np.isfinite(base[:, 0]), np.isfinite(jev[:, 0])), "baseline and Jev rows differ"
    assert np.isnan(base[10, 0]) and base[11, 0] == 0.02 and base[31, 0] == 0.02 and np.isnan(base[32, 0])
    with pytest.raises(ValueError, match="unknown baseline"):
        baseline_signal("x", scores, "returns", 21, cons, cal, rule)
