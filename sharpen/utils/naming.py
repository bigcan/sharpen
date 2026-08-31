from __future__ import annotations

import datetime
import re


def generate_experiment_name(
    category: str,
    system: str,
    description: str,
    experiment_id: str | None = None,
    date: datetime.date | None = None,
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


def generate_run_name(
    config_path: str,
    timestamp_format: str = "%Y%m%d_%H%M%S",
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
            f"Do NOT add suffixes - use WandB tags for metadata (Pilot, HPO, etc.).",
        )

    return is_valid



