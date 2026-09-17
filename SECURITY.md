# Security policy

## Reporting a vulnerability

Please **do not open a public issue.** Use GitHub's private vulnerability reporting: the
**Security** tab of this repository → **Report a vulnerability**.

Include what you found, where (file and line, or commit), and how to reproduce it. You will get an
acknowledgement when the report is read. This is a research repository maintained by one person, so
there is no guaranteed response time.

## In scope

- **Credentials or private identifiers** in the code, configuration, documentation or git history.
  The published history was filtered and scanned before release; if something was missed, this is
  the most important kind of report.
- **Code that could place real orders unintentionally**, for example a paper or testnet path that
  can reach a live account, or a kill switch that does not stop trading.
- Unsafe deserialization, command injection or path traversal in scripts that read configs,
  checkpoints or downloaded data.
- Dependency issues that affect code actually used here.

## Out of scope

- Trading losses from running any strategy in this repository. See [DISCLAIMER.md](DISCLAIMER.md).
- Vulnerabilities in third-party brokers, exchanges or data providers themselves.
- Issues that require an attacker to already control your machine or your broker credentials.

## If you run the live-trading code

The execution stack was only ever run against paper and demo accounts. Keep credentials in
environment variables or a local `.env` that is never committed, use testnet or paper endpoints, and
set up the kill file before connecting anything to real capital.
