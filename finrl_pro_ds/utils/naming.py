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
    config_path: str,
    timestamp_format: str = "%Y%m%d_%H%M%S"
) -> str:
    """
    Generate a standardized WandB run name from the config filename.

    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║  CANONICAL FORMAT: {descriptive_id}_{YYYYMMDD}_{HHMMSS}                  ║
    ║                                                                           ║
    ║  descriptive_id is derived from the config filename with underscores     ║
    ║  replaced by hyphens. All metadata goes in WandB tags, NOT the name.    ║
    ╚═══════════════════════════════════════════════════════════════════════════╝

    Args:
        config_path: Path to the YAML config file.
        timestamp_format: strftime format for timestamp.

    Returns:
        Formatted run name string.

    Examples:
        >>> generate_run_name('configs/funding_arb_sac_5assets_hpo.yaml')
        'funding-arb-sac-5assets-hpo_20260319_080300'
        >>> generate_run_name('configs/phase_r21v2_bdq_gc_3min.yaml')
        'phase-r21v2-bdq-gc-3min_20260319_080300'
        >>> generate_run_name('configs/gmgp1_sac_gc_15min.yaml')
        'gmgp1-sac-gc-15min_20260319_080300'
    """
    import os
    stem = os.path.splitext(os.path.basename(config_path))[0]
    descriptive_id = stem.replace("_", "-")
    timestamp = datetime.datetime.now().strftime(timestamp_format)
    return f"{descriptive_id}_{timestamp}"


def validate_run_name(run_name: str, raise_on_fail: bool = True) -> bool:
    """
    Validate that a run name follows the canonical format.

    Canonical pattern: {descriptive-id}_{YYYYMMDD}_{HHMMSS}
    Also accepts legacy: DeepScalper_V{digits}_...

    Args:
        run_name: The run name to validate
        raise_on_fail: If True, raises ValueError on invalid names

    Returns:
        True if valid, False otherwise

    Raises:
        ValueError: If run_name is invalid and raise_on_fail=True
    """
    # New format: {descriptive-id}_{YYYYMMDD}_{HHMMSS}
    # Legacy format: DeepScalper_V{digits}_...
    pattern = r"^([a-z0-9-]+_\d{8}_\d{6}|DeepScalper_V\d+.*)$"

    is_valid = bool(re.match(pattern, run_name))

    if not is_valid and raise_on_fail:
        raise ValueError(
            f"Invalid run name: '{run_name}'. "
            f"Expected format: '{{descriptive-id}}_{{YYYYMMDD}}_{{HHMMSS}}'. "
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
    ║  suffixes to run names. Use generate_run_name(config_path) instead.    ║
    ║  Pass descriptive metadata via WandB tags.                              ║
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
    return generate_run_name(config_path="legacy", timestamp_format=timestamp_format)

