"""Held-out test evaluation and confusion-matrix report generation."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: torch.utils.data.DataLoader, labels: dict[int,str], output_dir: str | Path = "evaluation") -> dict:
    device=next(model.parameters()).device; model.eval(); truth=[]; predictions=[]
    for features,targets in loader: predictions.extend(model(features.to(device)).argmax(1).cpu().tolist()); truth.extend(targets.tolist())
    names=[labels[i] for i in sorted(labels)]; report=classification_report(truth,predictions,labels=sorted(labels),target_names=names,output_dict=True,zero_division=0); matrix=confusion_matrix(truth,predictions,labels=sorted(labels))
    output=Path(output_dir); output.mkdir(parents=True,exist_ok=True); (output/"metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    fig,ax=plt.subplots(figsize=(max(6,len(names)),max(5,len(names)))); image=ax.imshow(matrix,cmap="Blues"); fig.colorbar(image); ax.set(xticks=range(len(names)),yticks=range(len(names)),xticklabels=names,yticklabels=names,xlabel="Predicted",ylabel="Actual",title="Mudra confusion matrix"); plt.setp(ax.get_xticklabels(),rotation=45,ha="right"); fig.tight_layout(); fig.savefig(output/"confusion_matrix.png",dpi=160); plt.close(fig)
    return report

