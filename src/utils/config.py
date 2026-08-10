"""Configuration loading and validation."""
from pathlib import Path
from typing import Any
import yaml

ROOT = Path(__file__).resolve().parents[2]

def load_config(path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    """Load YAML config and resolve project-relative directory values."""
    config_path = Path(path)
    if not config_path.is_absolute(): config_path = ROOT / config_path
    if not config_path.is_file(): raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open(encoding="utf-8") as file: config = yaml.safe_load(file)
    required = {"project", "dataset", "hand", "features", "training", "model", "export"}
    missing = required - set(config or {})
    if missing: raise ValueError(f"Configuration missing sections: {sorted(missing)}")
    ratios = config["dataset"]
    if abs(sum(ratios[k] for k in ("train_ratio", "validation_ratio", "test_ratio")) - 1.0) > 1e-6:
        raise ValueError("Dataset split ratios must sum to 1.0")
    for section, keys in (("dataset", ("raw_dir", "processed_dir", "train_dir", "validation_dir", "test_dir")), ("export", ("onnx_path", "labels_path"))):
        for key in keys: config[section][key] = str((ROOT / config[section][key]).resolve())
    return config

