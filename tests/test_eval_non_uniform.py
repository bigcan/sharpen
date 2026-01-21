from pathlib import Path

from finrl_pro_ds.eval.base import EvaluationContext
from finrl_pro_ds.eval.benchmark_catalog import BenchmarkCatalog
from finrl_pro_ds.eval.walk_forward import WalkForwardEvaluator


def test_walk_forward_metrics_vary_by_fingerprint(tmp_path: Path) -> None:
    catalog = BenchmarkCatalog(Path("finrl_pro_ds/configs/benchmarks.yaml"))
    catalog.load()
    wf = WalkForwardEvaluator(catalog)

    ctx_a = EvaluationContext(fingerprint_id="fp-A", benchmark_id="sp500_rolling_1y", walk_forward_splits=3)
    ctx_b = EvaluationContext(fingerprint_id="fp-B", benchmark_id="sp500_rolling_1y", walk_forward_splits=3)

    res_a = wf.evaluate(ctx_a)
    res_b = wf.evaluate(ctx_b)

    ma = res_a.evaluated_metrics
    mb = res_b.evaluated_metrics

    # Expect at least one key metric to differ due to per-fingerprint RNG seeding
    assert (
        ma.get("sharpe_ratio") != mb.get("sharpe_ratio")
        or ma.get("volatility") != mb.get("volatility")
        or ma.get("max_drawdown") != mb.get("max_drawdown")
    )

