# Can risk management rescue a losing strategy? 53 rules on one trading bot

> **Source:** a study from the **KISI** YouTube channel (Keep It Simple Investing), by Keng. Channel: [youtube.com/channel/UCPyTohwSp1URsZL9EX-NJ2A](https://www.youtube.com/channel/UCPyTohwSp1URsZL9EX-NJ2A) · Code: [github.com/bigcan/sharpen](https://github.com/bigcan/sharpen) · Support: [ko-fi.com/bigcan](https://ko-fi.com/bigcan).
> Video: link added when Ep07 is published.

People say risk management can rescue any strategy. We tested that on one strategy that was losing money: the GMGP1 reinforcement-learning agent trading Bitcoin on 15-minute bars ([`studies/gmgp1/`](../gmgp1/), 20 runs = 5 seeds x 4 walk-forward test months, realistic costs). We replayed its recorded trades under 53 overlay arms: stop-losses (on price, on equity, with a cooldown), take-profits, time stops, daily loss limits, volatility targets, drawdown brakes, and 6 controls that simply trade a smaller size.

**Verdict:** on this strategy, no. None of the 53 arms lifts the median profit factor (PF) of the 20 runs to 1.0. Every stop-loss arm is below the no-rule baseline (median PF 0.9057). The take-profits at best equal it, and time stops are the worst. Drawdown brakes and volatility targets come closest (best 0.9841), mainly by being in the market less: the best arm's median exposure is 0.29 against 1.08 with no rule, and it makes about a sixth of the trades. The rules did cut the damage: the median maximum drawdown falls from -29.2% to -6.5% for the best arm. Risk rules shrank the losses; they did not create an edge.

59 of the 1,060 single run-and-arm results do reach PF 1.0, and 42 of those come from one test month. Picking one of them after the fact is how a losing strategy gets mistaken for a winning one.

This is one strategy, one asset and four months, replayed rather than retrained. It is evidence about this setup, not a claim that stop-losses or risk management are useless.

## Rerun it (CPU, no private data, no keys)

```bash
git clone https://github.com/bigcan/sharpen && cd sharpen
pip install -e ".[dev]"
python studies/gmgp1/fetch_btc_1min.py      # public Bybit 1-minute candles -> data/ (a few minutes)
python studies/gmgp1/rerun.py               # step 2 replays the saved runs under the 53 overlay arms
python studies/risk-overlays/summarize.py   # the tables below
```

Expected output: [`expected/summarize-output.txt`](expected/summarize-output.txt).

## Results by rule family

Median = the median over the 20 runs for one arm.

| Rule family | Arms | Best median PF | Worst median PF | Arms above baseline | Runs with PF >= 1 |
|---|---|---|---|---|---|
| Stop-loss on price | 6 | 0.9026 | 0.7026 | 0 of 6 | 1 |
| Stop-loss on equity | 7 | 0.8949 | 0.7535 | 0 of 7 | 0 |
| Stop-loss with cooldown | 3 | 0.8625 | 0.8330 | 0 of 3 | 1 |
| Take-profit on price | 5 | 0.9057 | 0.7861 | 0 of 5 | 0 |
| Take-profit on equity | 5 | 0.9057 | 0.8579 | 0 of 5 | 0 |
| Time stop | 6 | 0.8883 | 0.2125 | 0 of 6 | 0 |
| Daily loss limit | 4 | 0.8796 | 0.8090 | 0 of 4 | 7 |
| Volatility target | 6 | 0.9367 | 0.9017 | 5 of 6 | 11 |
| Drawdown brake | 5 | 0.9841 | 0.9226 | 5 of 5 | 31 |
| Control: just trade smaller | 6 | 0.9303 | 0.8988 | 3 of 6 | 8 |

Baseline (no rule): median PF 0.9057, return -24.3%, maximum drawdown -29.2%, exposure 1.08, 206 trades.

| Arm | Median PF | Median return | Median max drawdown | Median exposure | Median trades |
|---|---|---|---|---|---|
| DDTHROTTLE_0.02-0.06_fl0.0 | 0.9841 | -1.2% | -6.5% | 0.29 | 34 |
| DDTHROTTLE_0.03-0.08_fl0.0 | 0.9723 | -2.2% | -8.9% | 0.36 | 51 |
| DDTHROTTLE_0.01-0.04_fl0.0 | 0.9573 | -2.0% | -5.1% | 0.17 | 21 |
| DDTHROTTLE_0.05-0.1_fl0.25 | 0.9399 | -6.1% | -12.2% | 0.42 | 70 |
| VOLTARGET_0.05 | 0.9367 | -6.2% | -9.3% | 0.38 | 56 |
| baseline (no rule) | 0.9057 | -24.3% | -29.2% | 1.08 | 206 |

## Sources in this repo
- [`studies/gmgp1/`](../gmgp1/): the strategy, its 20 saved runs, data fetch and `rerun.py`
- [`scripts/research/risk_overlay_lab.py`](../../scripts/research/risk_overlay_lab.py) (`--substrate gmgp1`): the overlay definitions and replay
