"""Reproducible class-stratified dataset splitting with recording group support."""
from __future__ import annotations
import argparse, shutil
from collections import defaultdict
from pathlib import Path
import numpy as np
from src.utils.config import load_config

def _group_key(path: Path) -> str:
    """Treat name before '__' as recording/person group; single samples remain unique."""
    return path.stem.split("__", 1)[0]

def split_dataset(config_path: str = "configs/config.yaml") -> dict[str, int]:
    config = load_config(config_path); source = Path(config["dataset"]["processed_dir"])
    files = sorted(source.glob("*/*.npz"))
    if not files: raise ValueError(f"No processed .npz samples found in {source}")
    rng = np.random.default_rng(config["project"]["seed"]); ratios = config["dataset"]
    destinations = {"train": Path(ratios["train_dir"]), "validation": Path(ratios["validation_dir"]), "test": Path(ratios["test_dir"])}
    for dest in destinations.values():
        dest.mkdir(parents=True, exist_ok=True)
        for old in dest.rglob("*.npz"): old.unlink()
    grouped: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for path in files: grouped[path.parent.name][_group_key(path)].append(path)
    counts = defaultdict(int)
    for label, groups in grouped.items():
        keys = list(groups); rng.shuffle(keys); n = len(keys)
        n_train, n_val = int(n * ratios["train_ratio"]), int(n * ratios["validation_ratio"])
        assignments = {"train": keys[:n_train], "validation": keys[n_train:n_train+n_val], "test": keys[n_train+n_val:]}
        for split, selected in assignments.items():
            out = destinations[split] / label; out.mkdir(exist_ok=True)
            for key in selected:
                for item in groups[key]: shutil.copy2(item, out / item.name); counts[split] += 1
    if not counts["train"] or not counts["validation"] or not counts["test"]: raise ValueError("Split produced an empty partition; collect more recording groups per class")
    return dict(counts)

if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="configs/config.yaml"); args=parser.parse_args(); print(split_dataset(args.config))

