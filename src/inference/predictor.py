"""Model predictor for PyTorch checkpoints and exported ONNX models."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from src.training.model import MudraClassifier

class MudraPredictor:
    def __init__(self, model_path: str | Path = "models/best_model/mudra_classifier.pt", labels_path: str | Path = "models/exported/labels.json") -> None:
        labels_file=Path(labels_path); model_file=Path(model_path)
        if not labels_file.is_file(): raise FileNotFoundError(f"Labels file not found: {labels_file}. Train and export first.")
        if not model_file.is_file(): raise FileNotFoundError(f"Model file not found: {model_file}. Train first.")
        self.labels={int(k):v for k,v in json.loads(labels_file.read_text(encoding="utf-8")).items()}; saved=torch.load(model_file,map_location="cpu",weights_only=False); meta=saved["metadata"]
        self.model=MudraClassifier(meta["input_size"],meta["hidden_sizes"],meta["dropout"],len(self.labels)); self.model.load_state_dict(saved["model_state"]); self.model.eval()
    @torch.no_grad()
    def predict(self, features: np.ndarray) -> dict:
        array=np.asarray(features,dtype=np.float32)
        expected=self.model.network[0].in_features
        if array.ndim != 1 or len(array)!=expected or not np.isfinite(array).all(): raise ValueError(f"Expected {expected} finite normalized features")
        probabilities=torch.softmax(self.model(torch.from_numpy(array).unsqueeze(0)),dim=1)[0].numpy(); index=int(probabilities.argmax())
        return {"class":self.labels[index],"confidence":float(probabilities[index]),"probabilities":{self.labels[i]:float(probabilities[i]) for i in range(len(self.labels))}}

