"""Module: sharpen
Purpose: Expose the FinRL Pro namespace for high-level package imports."""

# The package version (semantic versioning). Must equal `version` in pyproject.toml and the top
# entry of CHANGELOG.md; tests/test_package_version.py fails when the three drift apart. Sharpen
# launched publicly as v1.0.0 on 2026-09-18.
__version__ = "1.1.2"

__all__ = [
    "__version__",
    "data",
    "envs",
    "agents",
    "training",
    "analytics",
    "utils",
    "crypto",
]
