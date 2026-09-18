# Disclaimer

**This is research code. It is not investment advice, and it is not a trading product.**

## Not financial advice

Nothing in this repository — code, configuration, documentation, research notes,
result artifacts, or the R&D log — constitutes investment advice, financial
advice, trading advice, or a recommendation to buy, sell, or hold any security,
derivative, currency, or other financial instrument. The author is not a
licensed financial adviser, broker, or dealer. Consult a qualified professional
before making any investment decision.

## The results here are overwhelmingly negative

This repository is published as a **negative-results archive**. Seventy-six
strategies and probes across eight families were tested and closed. One survived
weakly and still failed its own pre-registered deployment gates. **No strategy in
this repository is validated for live trading, and none was ever deployed with
real capital.**

Any figure quoted in this repository — Sharpe ratio, profit factor, drawdown,
return — is the output of a **historical simulation or paper-trading run**, not a
record of realised trading profit. Simulated results are subject to look-ahead
bias, survivorship bias, transaction-cost misestimation, overfitting, and
selection effects. This project found and documents multiple instances of exactly
such defects in its own prior results; see `docs/LEAKS_FOUND.md`. Some published
numbers in the history of this project were later shown to be invalid.

**Past performance, whether real or simulated, does not indicate future results.**

## No warranty

This software is provided under the Apache License, Version 2.0, **"AS IS",
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND**, either express or implied. See
the LICENSE file.

Trading leveraged instruments, derivatives, foreign exchange, and
cryptocurrencies carries a high risk of loss, including loss exceeding deposited
funds. To the maximum extent permitted by law, the author accepts no liability
for any loss or damage arising from use of this software or reliance on anything
in this repository.

## Your responsibility

If you run any part of this code against a live or funded account, you do so
entirely at your own risk and are solely responsible for the outcome, for
compliance with the terms of your broker and any prop firm, and for compliance
with the laws and regulations of your jurisdiction.
