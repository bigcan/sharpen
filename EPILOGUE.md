# Epilogue

*This is opinion. Everything else in this repository tries to stay inside what the evidence
supports; this page deliberately goes beyond it. Read it as one person's conclusion, not as a result.*

---

I started this project to build trading strategies that make money after costs, and to prove it by
passing prop-firm challenges. Nine months later, nothing here has been traded with real capital, and
the single edge that survived did not clear its own deployment gate.

The part I did not expect was how often I was fooled. Three times the project produced a result that
looked like skill and was actually a bug: a feature that could see minutes into the future, a
gate that read the bar it was about to trade, and a seed that never reached the environments. Each
passed code review. The first one was the whole edge of a strategy I was already running on paper.
They are written up in [docs/LEAKS_FOUND.md](docs/LEAKS_FOUND.md). If you take one thing from this
repository, take those.

After that, most of the work was killing ideas: 76 of them, listed in
[NEGATIVE_RESULTS.md](NEGATIVE_RESULTS.md). Options selling turned out to be contaminated data and
then a lucky subsample. Arbitrage was real on paper and unharvestable in practice. Retail patterns
from books and videos lost to costs, or lost to randomly re-timed copies of themselves. Automated
search produced nothing that a search over pure noise could not also produce.

## What I now believe

**For someone like me, short-horizon trading is not a sustainable edge.** Not "impossible", which
this repository cannot show, but not sustainable for an individual without an information,
execution or cost advantage. The effects that survive are small, the costs are not, and the tools
that measure them honestly need more data than a retail trader has.

**Long-term investing in good businesses, held patiently, is the approach I am choosing instead.**
Keep it simple.

## What this repository does not show

I want to be precise here, because this page is the one most likely to be quoted.

- **This repository never tested fundamental value investing.** It never analysed a business, a
  balance sheet or a price paid against intrinsic value, and it never held anything for years.
- **The one value-shaped test it ran was negative.** A systematic cross-asset *value factor*, long and
  short, rebalanced, on ETF prices, lost money net of cost (pooled net Sharpe −0.364) and made the
  momentum book worse when combined. That is a trading strategy with "value" in its name. It is
  neither evidence for nor against buying good companies and holding them.
- **The surviving edge is a trading strategy.** Cross-asset momentum, the one thing that held up, is
  exactly the kind of systematic trading this page says I am stepping away from. Its edge was real
  and too small to clear a strict deployment bar.

So my conclusion is a judgement about where *my* time and capital are best spent, reached after
watching my own results fall apart under honest testing. It is not a finding of this code.

If you use this repository to keep searching, I hope the validation machinery saves you the months it
cost me, and that it tells you "no" early.

— Keng Lee, September 2026
