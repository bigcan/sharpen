"""Backtest analytics: pyfolio tear-sheet analysis and the WandB evaluator.

Carries no re-exports by design — import the submodules directly. This file
exists so `find_packages()` ships the subpackage: without it, `sharpen.analytics`
is silently omitted from the distribution while still importing fine in-tree.
"""
