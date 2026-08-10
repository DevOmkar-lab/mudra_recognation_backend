"""Validation metrics for multiclass classification."""
from __future__ import annotations
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support

@torch.no_grad()
def validate(model: torch.nn.Module, loader: torch.utils.data.DataLoader, criterion: torch.nn.Module, device: torch.device) -> dict[str, float]:
    model.eval(); losses=[]; actual=[]; predicted=[]
    for features, targets in loader:
        features, targets = features.to(device), targets.to(device); logits = model(features)
        losses.append(criterion(logits, targets).item()); actual.extend(targets.cpu().tolist()); predicted.extend(logits.argmax(1).cpu().tolist())
    precision, recall, f1, _ = precision_recall_fscore_support(actual, predicted, average="weighted", zero_division=0)
    return {"loss": float(np.mean(losses)), "accuracy": float(np.mean(np.asarray(actual)==np.asarray(predicted))), "precision": float(precision), "recall": float(recall), "f1": float(f1)}

