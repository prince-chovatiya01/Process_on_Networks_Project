"""
Configuration loader supporting root resolution and YAML parsing.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml


def find_project_root() -> Path:
    """Resolve project root directory by searching for marker files."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists() or (parent / "config").exists():
            return parent
    return Path.cwd()


def load_yaml(config_path: Path) -> Dict[str, Any]:
    """Load and parse a YAML file."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def get_config(config_name: str = "default.yaml", config_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from the config directory."""
    if config_dir is None:
        root = find_project_root()
        config_dir = root / "config"
    return load_yaml(config_dir / config_name)
