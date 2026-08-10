"""Reusable PyTorch training loop with checkpoints, scheduler, and early stopping."""
from __future__ import annotations
import json, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .validate import validate
from src.utils.logger import get_logger
LOG=get_logger(__name__)

def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def train_model(model: torch.nn.Module, train_loader: DataLoader, validation_loader: DataLoader, config: dict, labels: dict[int,str]) -> tuple[torch.nn.Module, dict]:
    set_seed(config["project"]["seed"]); setting=config["training"]; device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device)
    criterion=torch.nn.CrossEntropyLoss(); optimizer=torch.optim.AdamW(model.parameters(), lr=setting["learning_rate"], weight_decay=setting["weight_decay"])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=setting["scheduler_patience"], factor=setting["scheduler_factor"])
    checkpoint_dir=Path("models/checkpoints"); best_dir=Path("models/best_model"); checkpoint_dir.mkdir(parents=True,exist_ok=True); best_dir.mkdir(parents=True,exist_ok=True)
    best_loss=float("inf"); patience=0; metadata={"labels":labels,"input_size":config["model"]["input_size"],"hidden_sizes":config["model"]["hidden_sizes"],"dropout":setting["dropout"],"device":str(device),"history":[]}
    for epoch in range(1, setting["epochs"]+1):
        model.train(); losses=[]; correct=total=0
        for features, targets in train_loader:
            features,targets=features.to(device),targets.to(device); optimizer.zero_grad(); logits=model(features); loss=criterion(logits,targets); loss.backward(); optimizer.step()
            losses.append(loss.item()); correct += (logits.argmax(1)==targets).sum().item(); total += len(targets)
        metrics=validate(model,validation_loader,criterion,device); metrics.update(epoch=epoch,train_loss=float(np.mean(losses)),train_accuracy=correct/max(total,1)); metadata["history"].append(metrics); scheduler.step(metrics["loss"])
        torch.save({"model_state":model.state_dict(),"metadata":metadata}, checkpoint_dir / f"epoch_{epoch:03d}.pt")
        LOG.info("epoch %s train_loss=%.4f val_loss=%.4f val_acc=%.3f",epoch,metrics["train_loss"],metrics["loss"],metrics["accuracy"])
        if metrics["loss"] < best_loss:
            best_loss=metrics["loss"]; patience=0; torch.save({"model_state":model.state_dict(),"metadata":metadata}, best_dir / "mudra_classifier.pt"); (best_dir/"metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
        else:
            patience += 1
            if patience >= setting["early_stopping_patience"]: LOG.info("Early stopping at epoch %s",epoch); break
    saved=torch.load(best_dir/"mudra_classifier.pt",map_location=device,weights_only=False); model.load_state_dict(saved["model_state"]); return model, saved["metadata"]

