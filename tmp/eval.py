import json
from pathlib import Path
from finrl_pro.eval.benchmark_catalog import BenchmarkCatalog
from finrl_pro.eval.walk_forward import WalkForwardEvaluator
from finrl_pro.eval.base import EvaluationContext

data = json.loads(Path('tmp/fingerprints_full.json').read_text(encoding='utf-8-sig'))
catalog = BenchmarkCatalog(Path('finrl_pro/configs/benchmarks.yaml'))
catalog.load()
wf = WalkForwardEvaluator(catalog)
report = []
for fp in data:
    ref = fp.get('baseline_reference','')
    bench_id = ref.split(':',1)[1] if ':' in ref else ref
    ctx = EvaluationContext(fingerprint_id=fp['fingerprint_id'], benchmark_id=bench_id, walk_forward_splits=5)
    res = wf.evaluate(ctx)
    report.append({
        'fingerprint_id': fp['fingerprint_id'],
        'mlflow_run_id': fp['mlflow_run_id'],
        'benchmark_id': bench_id,
        'evaluated_metrics': res.evaluated_metrics,
        'variance_vs_baseline': res.variance_vs_baseline,
        'shap_summary': res.shap_summary,
        'walk_forward_splits': res.walk_forward_splits,
    })
Path('tmp/eval_report.json').write_text(json.dumps(report, indent=2, sort_keys=True), encoding='utf-8-sig')
print(Path('tmp/eval_report.json').read_text(encoding='utf-8-sig'))
