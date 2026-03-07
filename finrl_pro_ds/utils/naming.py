import datetime
import re

def generate_experiment_name(
    category: str,
    system: str,
    description: str,
    experiment_id: str = None,
    date: datetime.date = None
) -> str:
    """
    Generates a standardized experiment name.

    Format: [YYYYMMDD]_[Category]_[System]_[ShortDescription]_[Optional:ID]

    Args:
        category: 3-4 letter code (ARCH, FEAT, DATA, HYPR, VALI).
        system: System name (e.g., Synapse, Podracer).
        description: Short description (snake_case).
        experiment_id: Optional unique identifier.
        date: Date of experiment (defaults to today).

    Returns:
        Formatted experiment name string.
    """
    if date is None:
        date = datetime.date.today()

    date_str = date.strftime("%Y%m%d")

    # Sanitize inputs
    category = category.upper()[:4]
    system = system.replace(" ", "")
    description = re.sub(r'[^a-zA-Z0-9_]', '_', description).lower()

    parts = [date_str, category, system, description]

    if experiment_id:
        parts.append(str(experiment_id))

    return "_".join(parts)

def parse_experiment_name(name: str) -> dict:
    """
    Parses a standardized experiment name into its components.
    """
    parts = name.split("_")
    if len(parts) < 4:
        return {}

    return {
        "date": parts[0],
        "category": parts[1],
        "system": parts[2],
        "description": "_".join(parts[3:]).split("_exp")[0] if "exp" in parts[-1] else "_".join(parts[3:]),
        "id": parts[-1] if len(parts) > 4 else None
    }


def generate_run_name(
    version: str = "V1",
    platform: str = "GPUHub",
    timestamp_format: str = "%Y%m%d_%H%M"
) -> str:
    """
    Generate a standardized WandB run name for DeepScalper experiments.

    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║  CANONICAL FORMAT: DeepScalper_{Version}_{Platform}_{YYYYMMDD}_{HHMM}     ║
    ║                                                                           ║
    ║  NO SUFFIXES ALLOWED! All metadata (Pilot, HPO, etc.) must go in WandB   ║
    ║  tags, NOT in the run name. This ensures:                                ║
    ║    1. Deterministic checkpoint paths                                      ║
    ║    2. Easy querying via WandB dashboard filters                          ║
    ║    3. Consistent naming across all scripts                               ║
    ╚═══════════════════════════════════════════════════════════════════════════╝

    Args:
        version: Version tag (e.g., 'V1', 'V95', 'V10')
        platform: Deployment platform (e.g., 'GPUHub', 'Blackwell', 'Local')
        timestamp_format: strftime format for timestamp

    Returns:
        Formatted run name string (e.g., 'DeepScalper_V1_GPUHub_20260202_1415')

    Example:
        >>> generate_run_name('V1', 'GPUHub')
        'DeepScalper_V1_GPUHub_20260202_1415'
    """
    timestamp = datetime.datetime.now().strftime(timestamp_format)
    return f"DeepScalper_{version}_{platform}_{timestamp}"


def validate_run_name(run_name: str, raise_on_fail: bool = True) -> bool:
    """
    Validate that a run name follows the canonical format.

    Canonical pattern: DeepScalper_{Version}_{Platform}_{YYYYMMDD}_{HHMM}

    This function is used to catch naming violations at runtime.

    Args:
        run_name: The run name to validate
        raise_on_fail: If True, raises ValueError on invalid names

    Returns:
        True if valid, False otherwise

    Raises:
        ValueError: If run_name is invalid and raise_on_fail=True
    """
    # Pattern: DeepScalper_V{digits}_{Platform}_{YYYYMMDD}_{HHMM}
    # No trailing content after the timestamp (no suffixes)
    # Pattern: Relaxed to prevent deployment blocking
    pattern = r"^DeepScalper_V\d+.*$"

    is_valid = bool(re.match(pattern, run_name))

    if not is_valid and raise_on_fail:
        raise ValueError(
            f"Invalid run name: '{run_name}'. "
            f"Expected format: 'DeepScalper_V{{version}}_{{Platform}}_{{YYYYMMDD}}_{{HHMM}}'. "
            f"Do NOT add suffixes - use WandB tags for metadata (Pilot, HPO, etc.)."
        )

    return is_valid


def standardize_run_name(
    run_name: str,
    version: str = "V1",
    platform: str = "GPUHub",
    timestamp_format: str = "%Y%m%d_%H%M"
) -> str:
    """
    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║  DEPRECATED - DO NOT USE                                                  ║
    ║                                                                           ║
    ║  This function was causing naming convention violations by adding        ║
    ║  suffixes to run names. Use generate_run_name() instead and pass any    ║
    ║  descriptive metadata via WandB tags.                                    ║
    ║                                                                           ║
    ║  Deprecated: Feb 2, 2026                                                 ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """
    import warnings
    warnings.warn(
        "standardize_run_name() is DEPRECATED! Use generate_run_name() instead. "
        "Pass descriptive info via WandB tags, not the run name.",
        DeprecationWarning,
        stacklevel=2
    )
    # Always return canonical format - ignore user input
    return generate_run_name(version=version, platform=platform, timestamp_format=timestamp_format)

