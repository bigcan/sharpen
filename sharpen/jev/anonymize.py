"""Entity/date masking for filing text sent to Jev (ATL x Jev plan item B3, contamination step C2).

The point is to stop Jev answering from MEMORY of a known company and period instead of from the text.
Masking is necessarily partial — product names, segment names and distinctive numbers survive — which
is exactly what the C3 identification probe measures, so this module aims to be predictable rather than
clever. It removes:

* the 8-K COVER PAGE (registrant name, address, CIK, IRS/commission numbers, ticker table): everything
  before the first ``Item N.NN`` heading;
* press-release DATELINES ("CUPERTINO, CALIFORNIA — …"), whose city names identify the filer;
* the registrant's names — legal name, the name without its corporate suffix, supplied aliases and
  possessives — case-insensitively, as whole words;
* tickers in exchange-prefixed form (``NYSE: V``, ``(Nasdaq: AAPL)``) always, and bare tickers of
  length >= 3 as whole upper-case tokens (short tickers such as ``V``/``BA`` collide with ordinary text);
* calendar dates, fiscal-year tags and four-digit years 1990-2039;
* URLs, e-mail addresses and phone numbers.

Tokens are replaced with fixed placeholders (``[COMPANY]``, ``[TICKER]``, ``[DATELINE]``, ``[DATE]``,
``[YEAR]``, ``[URL]``, ``[EMAIL]``, ``[PHONE]``) so the text stays readable. Name masking is whole-word
and case-insensitive, so a name that is also a common word ("Target") masks that word too — harmless
for identification, and preferable to leaking the name in lower case.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

_SUFFIXES = re.compile(
    r"[,\s]+(incorporated|inc\.?|corporation|corp\.?|company|co\.?|holdings?|group|plc|ltd\.?|llc|"
    r"l\.p\.|lp|n\.v\.|s\.a\.|ag|limited|the)$", re.I)
_EXCHANGES = r"(?:NYSE(?:\s+American|\s+Arca)?|NASDAQ(?:\s*GS|\s*GM|\s*CM)?|Nasdaq(?:\s+Global\s+Select)?|Cboe|NYSEArca|AMEX)"
_MONTHS = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
           r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?")
_DATE_PATTERNS = (
    re.compile(rf"\b{_MONTHS}\s+\d{{1,2}},?\s+(?:19|20)\d{{2}}\b", re.I),     # May 2, 2024
    re.compile(rf"\b\d{{1,2}}\s+{_MONTHS}\s+(?:19|20)\d{{2}}\b", re.I),       # 2 May 2024
    re.compile(rf"\b{_MONTHS}\s+(?:19|20)\d{{2}}\b", re.I),                   # May 2024
    re.compile(r"\b(?:19|20)\d{2}-\d{1,2}-\d{1,2}\b"),                         # 2024-05-02
    re.compile(r"\b\d{1,2}/\d{1,2}/(?:19|20)?\d{2}\b"),                        # 5/2/2024, 5/2/24
)
_FISCAL = re.compile(r"\b(?:FY|F|Q[1-4]\s*(?:FY|F)?)\s*'?(?:19|20)?\d{2}\b", re.I)
_YEAR = re.compile(r"\b(?:199\d|20[0-3]\d)\b")
_URL = re.compile(r"\b(?:https?://|www\.)\S+", re.I)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE = re.compile(r"(?:\+?1[\s.-]?)?\(?\b\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b")
_ITEM = re.compile(r"\bItem\s+\d{1,2}\.\d{2}\b", re.I)
# Press-release dateline at the start of a line: "CUPERTINO, CALIFORNIA — …" / "NEW YORK, May 2 (…) --".
_DATELINE = re.compile(r"^[A-Z][A-Z .'’&-]{2,40},[^\n]{0,80}?(?:[-–—]{1,2}|/[A-Za-z ]+/)", re.M)
# ...and anywhere once dates are masked, because html_to_text flattens block boundaries into spaces:
# "MOUNTAIN VIEW, Calif. – [DATE] –", "SEATTLE—(BUSINESS WIRE) [DATE]—", "NEW YORK, [DATE] /PRNewswire/".
_DATELINE_INLINE = re.compile(
    r"\b[A-Z][A-Z'’&.-]+(?: [A-Z][A-Z'’&.-]+){0,3}(?:,\s*[A-Z][A-Za-z.]{1,15}\.?)?\s*[-–—,]{1,2}\s*"
    r"(?:\([A-Z][A-Za-z ]{2,30}\)\s*)?(?=\[DATE\])")


def strip_cover_page(text: str) -> str:
    """Drop everything before the first ``Item N.NN`` heading (the 8-K cover page); no heading =>
    text unchanged."""
    m = _ITEM.search(text)
    return text[m.start():] if m else text


def name_variants(legal_name: str, aliases: Iterable[str] = ()) -> list[str]:
    """Legal name, the name without its corporate suffix(es) and any aliases, longest first."""
    full = re.sub(r"\s*\([^)]*\)", "", legal_name).strip()          # "Alphabet Inc. (Class A)"
    names = {legal_name.strip(), full}
    base = re.sub(r"^the\s+", "", full, flags=re.I)                 # "The Coca-Cola Company"
    names.add(base)
    while True:
        stripped = _SUFFIXES.sub("", base).strip(" ,.")
        if stripped == base or not stripped:
            break
        base = stripped
    names.add(base)
    names.update(a.strip() for a in aliases if a and a.strip())
    return sorted((n for n in names if len(n) >= 2), key=len, reverse=True)


def anonymize(text: str, *, legal_name: str, tickers: Iterable[str] = (), aliases: Iterable[str] = (),
              drop_cover_page: bool = True) -> str:
    """Mask ``text`` (see module docstring). ``drop_cover_page`` is for an 8-K PRIMARY document: pass
    ``False`` for exhibits (EX-99.x), where an incidental "Item 2.02" mention would otherwise make
    :func:`strip_cover_page` silently discard everything before it."""
    out = strip_cover_page(text) if drop_cover_page else text
    out = _DATELINE.sub("[DATELINE] —", out)
    out = _URL.sub("[URL]", out)
    out = _EMAIL.sub("[EMAIL]", out)
    for n in name_variants(legal_name, aliases):
        out = re.sub(rf"(?<![\w-]){re.escape(n)}(?:['’]s)?(?![\w-])", "[COMPANY]", out, flags=re.I)
    for t in {t.upper() for t in tickers if t}:
        variants = {t, t.replace("-", "."), t.replace(".", "-")}
        for v in variants:
            out = re.sub(rf"{_EXCHANGES}\s*[:：]\s*{re.escape(v)}\b", "[TICKER]", out, flags=re.I)
            if len(v) >= 3:
                out = re.sub(rf"(?<![\w-]){re.escape(v)}(?![\w-])", "[TICKER]", out)
    for pat in _DATE_PATTERNS:
        out = pat.sub("[DATE]", out)
    out = _DATELINE_INLINE.sub("[DATELINE] – ", out)
    out = _FISCAL.sub("[YEAR]", out)
    out = _YEAR.sub("[YEAR]", out)
    return _PHONE.sub("[PHONE]", out)


def residual_identifiers(text: str, *, legal_name: str, tickers: Iterable[str] = (),
                         aliases: Iterable[str] = ()) -> list[str]:
    """Tripwire helper: which name/ticker(len>=3)/year tokens survived masking (should be empty)."""
    hits = [n for n in name_variants(legal_name, aliases)
            if re.search(rf"(?<![\w-]){re.escape(n)}(?![\w-])", text, flags=re.I)]
    hits += [t for t in tickers if len(t) >= 3 and re.search(rf"(?<![\w-]){re.escape(t)}(?![\w-])", text)]
    hits += _YEAR.findall(text)
    return hits
