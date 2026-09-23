"""Anonymizer tripwires (plan C2): every identifier class the module promises to remove must be gone,
and ordinary financial text (amounts, percentages, short-ticker words) must survive."""
from __future__ import annotations

from sharpen.jev.anonymize import anonymize, name_variants, residual_identifiers, strip_cover_page

_COVER = """UNITED STATES SECURITIES AND EXCHANGE COMMISSION
FORM 8-K
Apple Inc.
(Exact name of registrant as specified in its charter)
California 001-36743 94-2404110
One Apple Park Way Cupertino, California 95014
Common Stock AAPL The Nasdaq Stock Market LLC
"""
_BODY = """Item 2.02 Results of Operations and Financial Condition.
On May 2, 2024, Apple Inc. issued a press release. APPLE's revenue was $90.8 billion, up 12% for fiscal 2024.
CUPERTINO, CALIFORNIA — Apple today announced results for Q2 2024 (Nasdaq: AAPL). FY24 guidance.
Contact investor_relations@apple.com or (408) 996-1010, see www.apple.com/investor. Dated 2024-05-02, 5/2/2024,
2 May 2024 and March 2023.
"""


def test_cover_page_is_dropped():
    out = strip_cover_page(_COVER + _BODY)
    assert out.startswith("Item 2.02")
    for token in ("94-2404110", "001-36743", "Apple Park Way", "95014", "Exact name of registrant"):
        assert token not in anonymize(_COVER + _BODY, legal_name="Apple Inc.", tickers=["AAPL"])


def test_names_tickers_dates_and_contacts_are_masked():
    out = anonymize(_COVER + _BODY, legal_name="Apple Inc.", tickers=["AAPL"])
    assert residual_identifiers(out, legal_name="Apple Inc.", tickers=["AAPL"]) == []
    for placeholder in ("[COMPANY]", "[TICKER]", "[DATE]", "[YEAR]", "[URL]", "[EMAIL]", "[PHONE]", "[DATELINE]"):
        assert placeholder in out, placeholder
    assert "Cupertino" not in out and "CUPERTINO" not in out
    assert "$90.8 billion" in out and "12%" in out            # the financial content survives


def test_name_variants_strip_parentheticals_suffixes_and_leading_the():
    assert "Alphabet" in name_variants("Alphabet Inc. (Class A)")
    v = name_variants("The Coca-Cola Company")
    assert "Coca-Cola" in v and "The Coca-Cola Company" in v
    out = anonymize("Item 8.01 The Coca-Cola Company and Coca-Cola's bottlers.", legal_name="The Coca-Cola Company")
    assert "Coca-Cola" not in out


def test_short_tickers_masked_only_in_exchange_prefixed_form():
    out = anonymize("Item 8.01 Class V shares (NYSE: V) and vitamin V.", legal_name="Visa Inc.", tickers=["V"])
    assert "(NYSE: V)" not in out and "[TICKER]" in out
    assert "Class V shares" in out and "vitamin V" in out    # bare 1-letter ticker left alone


def test_share_class_ticker_both_spellings():
    out = anonymize("Item 8.01 NYSE: BRK.B and BRK-B holders", legal_name="Berkshire Hathaway Inc.",
                    tickers=["BRK-B"])
    assert "BRK" not in out


def test_inline_datelines_are_masked_after_html_flattening():
    flat = ("Item 2.02 Exhibit 99.1 Results MOUNTAIN VIEW, Calif. – February 1, 2022 – The company said. "
            "SEATTLE—(BUSINESS WIRE) October 28, 2021—Results. NEW YORK, May 2, 2024 /PRNewswire/ -- Text.")
    out = anonymize(flat, legal_name="Alphabet Inc.")
    for city in ("MOUNTAIN VIEW", "SEATTLE", "NEW YORK", "BUSINESS WIRE"):
        assert city not in out, city
    assert "Results" in out and "The company said" in out


def test_names_mask_whole_words_only():
    out = anonymize("Item 8.01 Target Corporation targeted growth.", legal_name="Target Corporation")
    assert "targeted" in out and "Target Corporation" not in out
