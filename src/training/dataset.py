"""PyTorch dataset for stored feature vectors."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

def discover_labels(directory: str | Path) -> dict[int, str]:
    labels = sorted(p.name for p in Path(directory).iterdir() if p.is_dir())
    if not labels: raise ValueError(f"No class directories found in {directory}")
    return dict(enumerate(labels))

class MudraDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, directory: str | Path, labels: dict[int, str] | None = None) -> None:
        self.directory = Path(directory); self.labels = labels or discover_labels(self.directory)
        self.class_to_id = {name: idx for idx, name in self.labels.items()}
        self.samples = [(p, self.class_to_id[p.parent.name]) for p in sorted(self.directory.glob("*/*.npz")) if p.parent.name in self.class_to_id]
        if not self.samples: raise ValueError(f"No .npz feature samples found in {self.directory}")
    def __len__(self) -> int: return len(self.samples)
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[index]
        with np.load(path, allow_pickle=False) as item: features = item["features"].astype(np.float32)
        if features.ndim != 1 or not np.isfinite(features).all(): raise ValueError(f"Invalid feature vector: {path}")
        return torch.from_numpy(features), torch.tensor(label, dtype=torch.long)

def save_labels(labels: dict[int, str], path: str | Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({str(k): v for k,v in labels.items()}, indent=2), encoding="utf-8")

