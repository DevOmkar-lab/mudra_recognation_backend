"""ONNX export for the trained landmark classifier."""
from __future__ import annotations
from pathlib import Path
import torch
from .dataset import save_labels

def export_onnx(model: torch.nn.Module, labels: dict[int,str], config: dict) -> Path:
    destination=Path(config["export"]["onnx_path"]); destination.parent.mkdir(parents=True,exist_ok=True); model.eval().cpu()
    torch.onnx.export(model,torch.zeros(1,config["model"]["input_size"],dtype=torch.float32),destination,input_names=["features"],output_names=["logits"],dynamic_axes={"features":{0:"batch"},"logits":{0:"batch"}},opset_version=17)
    save_labels(labels,config["export"]["labels_path"]); return destination

