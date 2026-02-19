"""Dump all runs from the local metrics DB."""
import sqlite3, sys

db = sqlite3.connect('results/metrics.db')
db.row_factory = sqlite3.Row
rows = db.execute('SELECT * FROM runs ORDER BY created_at DESC').fetchall()
print(f'Total runs in DB: {len(rows)}\n')

header = f'{"Run ID":<12} {"Name":<45} {"State":<10} {"Test Sharpe":>12} {"Test PF":>10} {"Test Return":>12} {"Steps":>10}'
print(header)
print('-' * len(header))
for r in rows:
    d = dict(r)
    rid = (d.get('run_id', '?') or '?')[:10]
    name = (d.get('run_name', '?') or '?')[:44]
    state = (d.get('state', '?') or '?')[:9]
    ts = float(d.get('test_sharpe', 0) or 0)
    tpf = float(d.get('test_profit_factor', 0) or 0)
    tr = float(d.get('test_return', 0) or 0) * 100
    steps = int(d.get('total_steps', 0) or 0)
    print(f'{rid:<12} {name:<45} {state:<10} {ts:>12.3f} {tpf:>10.3f} {tr:>11.2f}% {steps:>10}')
db.close()
