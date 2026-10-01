"""Load local pickle caches written before the FinRL-Pro-DS → Sharpen package rename.

The rename (2026-08-31, ``feca38ac``) moved every module from ``finrl_pro_ds.*`` to ``sharpen.*``. A pickle
stores the defining module path of each class, so every cache pickled before that date — including the
PIT S&P 500 union behind the ``us_equity`` substrate (``data/raw/equity_panel/_pit_union_2007.pkl``) —
raised ``ModuleNotFoundError: No module named 'finrl_pro_ds'`` on load: the only adequately-powered
Crucible substrate could not be built, and its recorded verdicts could not be reproduced (deep audit
2026-09-30). This unpickler maps the old package path onto the new one and changes nothing else.

Only for TRUSTED local caches this project wrote itself — the same trust boundary ``pickle.load`` had.
"""
from __future__ import annotations

import pickle
from typing import IO, Any

#: (old top-level package, new top-level package)
_RENAMED_PACKAGES: tuple[tuple[str, str], ...] = (("finrl_pro_ds", "sharpen"),)


class _RenameUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        for old, new in _RENAMED_PACKAGES:
            if module == old or module.startswith(old + "."):
                module = new + module[len(old):]
                break
        return super().find_class(module, name)


def load_compat(fh: IO[bytes]) -> Any:
    """``pickle.load`` that resolves classes pickled under a pre-rename package path."""
    return _RenameUnpickler(fh).load()
