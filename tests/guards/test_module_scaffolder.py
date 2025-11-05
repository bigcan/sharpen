"""Guards ensuring module scaffolding respects the finrl_pro namespace."""

from __future__ import annotations

from pathlib import Path

import pytest

from finrl_pro.utils.module_scaffolder import ModuleScaffolder


def _create_package_root(tmp_path: Path) -> Path:
    package_root = tmp_path / "finrl_pro"
    package_root.mkdir()
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    return package_root


def test_scaffolder_rejects_non_finrl_modules(tmp_path: Path) -> None:
    """Scaffolder must refuse to write outside the FinRL Pro namespace."""
    package_root = _create_package_root(tmp_path)
    scaffolder = ModuleScaffolder(package_root=package_root)

    with pytest.raises(ValueError):
        scaffolder.scaffold("external.module")


def test_scaffolder_creates_module_with_init_files(tmp_path: Path) -> None:
    """Scaffolder should build module skeletons and ensure packages exist."""
    package_root = _create_package_root(tmp_path)
    scaffolder = ModuleScaffolder(package_root=package_root)

    target = scaffolder.scaffold(
        "finrl_pro.alpha.beta",
        docstring="Example scaffolded module.",
    )

    assert target.exists()
    assert target.read_text(encoding="utf-8").startswith('"""Example scaffolded module.')
    assert (package_root / "alpha" / "__init__.py").exists()
