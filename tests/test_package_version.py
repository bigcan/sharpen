"""The package version is stated in three places and they must agree.

Until 2026-10-01 nothing tracked it: ``pyproject.toml`` said ``0.1.0`` two weeks after the public
``v1.0.0`` release, and ``sharpen`` exported no ``__version__``. A version nothing checks drifts, so
this pins ``pyproject.toml`` == ``sharpen.__version__`` == the top entry of ``CHANGELOG.md``.

The release tag (``vX.Y.Z``) is created on the published repository, which this checkout cannot see,
so it is not asserted here; the changelog entry is the record a bump cannot skip."""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

import sharpen

ROOT = Path(__file__).resolve().parents[1]
_SEMVER = re.compile(r"\d+\.\d+\.\d+")
_ENTRY = re.compile(r"^## \[(\d+\.\d+\.\d+)\] — (unreleased|\d{4}-\d{2}-\d{2})$", re.MULTILINE)


def _key(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in v.split("."))


def _pyproject_version() -> str:
    with open(ROOT / "pyproject.toml", "rb") as fh:
        return str(tomllib.load(fh)["project"]["version"])


def _changelog_entries() -> list[tuple[str, str]]:
    return _ENTRY.findall((ROOT / "CHANGELOG.md").read_text(encoding="utf-8"))


def test_version_is_semver_and_the_same_in_pyproject_and_package() -> None:
    assert _SEMVER.fullmatch(sharpen.__version__)
    assert _pyproject_version() == sharpen.__version__


def test_changelog_top_entry_is_the_current_version() -> None:
    entries = _changelog_entries()
    assert entries, "CHANGELOG.md has no '## [X.Y.Z] — <date|unreleased>' entry"
    assert entries[0][0] == sharpen.__version__


def test_changelog_is_newest_first_with_one_entry_per_version() -> None:
    versions = [v for v, _ in _changelog_entries()]
    assert versions == sorted(set(versions), key=_key, reverse=True)
    assert "1.0.0" in versions                                   # the 2026-09-18 public launch
    # only the newest entry may still be unreleased
    assert all(when != "unreleased" for _, when in _changelog_entries()[1:])
