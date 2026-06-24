"""Signal libraries — registered candidate alphas grouped by family.

Each library module exposes a module-level ``SIGNALS: list[Signal]`` that the CLI
(``scripts/research/eval_signals.py``) collects into a batch. ``demo`` ships a few classic
causal technical signals; ``alphas101`` / ``tradingview`` (P8) extend the set.
"""
