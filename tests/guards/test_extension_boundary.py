"""Guards verifying the FinRL Pro extension boundary is enforced."""

from __future__ import annotations

from pathlib import Path

import pytest

from finrl_pro.utils.module_registry import FinRLProModule
from finrl_pro.utils.module_scaffolder import ModuleScaffolder


def _package_root(tmp_path: Path) -> Path:
    root = tmp_path / "finrl_pro"
    root.mkdir()
    (root / "__init__.py").write_text("", encoding="utf-8")
    return root


def test_finrl_pro_module_requires_namespace() -> None:
    """Modules outside the finrl_pro namespace must be rejected."""
    with pytest.raises(ValueError):
        FinRLProModule("external.module")


def test_finrl_pro_module_accepts_valid_namespace() -> None:
    """Valid modules within finrl_pro namespace should instantiate."""
    module = FinRLProModule("finrl_pro.agents.strategy")
    assert module.dotted_path == "finrl_pro.agents.strategy"


def test_scaffolder_cannot_escape_namespace(tmp_path: Path) -> None:
    """Scaffolder must refuse to generate modules beyond finrl_pro."""
    scaffolder = ModuleScaffolder(package_root=_package_root(tmp_path))
    with pytest.raises(ValueError):
        scaffolder.scaffold("not_finrl.module")
