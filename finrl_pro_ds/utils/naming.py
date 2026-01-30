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
    suffix: str = None,
    timestamp_format: str = "%Y%m%d_%H%M"
) -> str:
    """
    Generate a standardized WandB run name for DeepScalper experiments.
    
    Canonical format: DeepScalper_{Version}_{Platform}_{YYYYMMDD}_{HHMM}[_{Suffix}]
    
    Args:
        version: Version tag (e.g., 'V1', 'V95', 'V10')
        platform: Deployment platform (e.g., 'GPUHub', 'Blackwell', 'Local')
        suffix: Optional suffix (e.g., 'HPO', 'Backtest')
        timestamp_format: strftime format for timestamp
        
    Returns:
        Formatted run name string
        
    Example:
        >>> generate_run_name('V95', 'Blackwell')
        'DeepScalper_V95_Blackwell_20260129_1942'
        >>> generate_run_name('V95', 'GPUHub', suffix='HPO')
        'DeepScalper_V95_GPUHub_20260129_1942_HPO'
    """
    timestamp = datetime.datetime.now().strftime(timestamp_format)
    base = f"DeepScalper_{version}_{platform}_{timestamp}"
    return f"{base}_{suffix}" if suffix else base


def standardize_run_name(
    run_name: str,
    version: str = "V1",
    platform: str = "GPUHub",
    timestamp_format: str = "%Y%m%d_%H%M"
) -> str:
    """
    Standardize a manually-provided run name to fit the canonical format.
    
    If name already fits the pattern DeepScalper_V.*_Platform_TIMESTAMP_..., returns it as is.
    Otherwise, treats the manual name as the 'suffix' and rebuilds the standard name.
    """
    import re
    # Pattern: DeepScalper_..._YYYYMMDD_HHMM
    ts_pattern = r"\d{8}_\d{4}"
    if re.search(f"DeepScalper_.*_.*_{ts_pattern}", run_name):
        return run_name
        
    # Extract suffix from manual name
    # Remove prefix if present
    clean_suffix = run_name
    if clean_suffix.startswith("DeepScalper_"):
        clean_suffix = clean_suffix[len("DeepScalper_"):]
    
    # Clean up underscores
    clean_suffix = clean_suffix.strip('_')
        
    return generate_run_name(version=version, platform=platform, suffix=clean_suffix, timestamp_format=timestamp_format)
