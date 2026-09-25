# Forward lockbox registration — `keel-v1` and `keel-lo` (2026-09-25)

Registered before any forward day exists (registration close 2026-09-25; the latest close on disk is 2026-09-24).
Code: `scripts/research/finrlx_strategy/lockbox.py` (scores only days strictly after the registration close; tested).

| Book | Cell | Why it is here |
|---|---|---|
| `keel-v1` | `cov_sv03_static_sc0.5` | The pre-registered H1 book. It FAILED the sealed 1973–2005 window narrowly (alpha t 1.996 vs 2.0; deflated alpha 0.63 vs 0.95). Tracked forward as a monitor. |
| `keel-lo` | `linear_sv03_static_sc0` | **New forward-only hypothesis.** The long-only variant (50% SPY + long-or-flat trend sleeve, no shorts), which runs natively through FinRL-X's backtester and Alpaca executor. On the sealed window the long-only cells scored best (alpha t 3.1–3.2), but that window was used to notice it, so **no past data counts as evidence for `keel-lo`**; only days after 2026-09-25 do. |

- **Data:** the 18 ETFs' total-return closes from yfinance (`auto_adjust=True`), cash = the 13-week bill (^IRX); costs and
  account model exactly as `configs/finrlx_strategy.yaml` (no proxies, so no expense charge: ETF prices are net of fees).
- **Criterion (house, `configs/crucible_lockbox.gates.yaml`):** at least 63 forward bars AND forward excess Sharpe ≥ 0.30.
  Also reported every time: forward Sharpe vs SPY's, return vs SPY's, drawdown.
- **Power, stated now:** at IR 0.35–0.55 the forward alpha t after one year is about 0.35–0.55; reaching t ≈ 2 takes
  13–30 years. The lockbox can falsify (a book that loses badly to SPY or turns negative) far sooner than it can certify.
  Passing the house incubation criterion is not evidence that either book beats SPY.
- **No paper or live trading** follows from this lockbox without the operator's approval and a passed Tier-2 audit.
- **Run:** `cd scripts && python -m research.finrlx_strategy.lockbox` → `results/finrlx_strategy/lockbox_status.json`.
