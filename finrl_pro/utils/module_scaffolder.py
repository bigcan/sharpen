"""Utilities for scaffolding FinRL Pro modules."""

from __future__ import annotations

from pathlib import Path


class ModuleScaffolder:
    """Scaffold new modules within the finrl_pro namespace."""

    def __init__(self, package_root: Path | None = None) -> None:
        self._package_root = package_root or Path(__file__).resolve().parents[1]

    def scaffold(self, dotted_path: str, *, docstring: str | None = None) -> Path:
        """Create a new module ensuring it resides within finrl_pro."""
        if not dotted_path.startswith("finrl_pro."):
            raise ValueError("Modules must be created under the finrl_pro namespace.")

        relative_parts = dotted_path.split(".")[1:]
        if not relative_parts:
            raise ValueError("A concrete module path is required.")

        module_path = self._package_root.joinpath(*relative_parts)
        target_file = module_path.with_suffix(".py")

        if target_file.exists():
            raise FileExistsError(f"Module already exists: {target_file}")

        target_file.parent.mkdir(parents=True, exist_ok=True)
        for parent in self._iter_package_parents(target_file.parent):
            init_file = parent / "__init__.py"
            init_file.touch(exist_ok=True)

        header = docstring or f"Module scaffold for {dotted_path}."
        target_file.write_text(
            f'"""{header}"""\n\nfrom __future__ import annotations\n',
            encoding="utf-8",
        )
        return target_file

    @staticmethod
    def _iter_package_parents(path: Path) -> list[Path]:
        """Return all parents down to the finrl_pro package."""
        packages: list[Path] = []
        current = path
        while current.name and current.name != "finrl_pro":
            packages.append(current)
            current = current.parent
        packages.append(current)
        return packages
