"""Complete preprocessing-to-export training entry point."""
from __future__ import annotations
import argparse
from torch.utils.data import DataLoader
from src.data.preprocess import preprocess
from src.data.split import split_dataset
from src.training.dataset import MudraDataset
from src.training.evaluate import evaluate
from src.training.export import export_onnx
from src.training.model import MudraClassifier
from src.training.train import train_model
from src.utils.config import load_config

def main(config_path: str) -> None:
    config=load_config(config_path); preprocess(config_path); split_dataset(config_path)
    labels=MudraDataset(config["dataset"]["train_dir"]).labels
    train_data=MudraDataset(config["dataset"]["train_dir"],labels); validation_data=MudraDataset(config["dataset"]["validation_dir"],labels); test_data=MudraDataset(config["dataset"]["test_dir"],labels)
    input_size=train_data[0][0].numel(); config["model"]["input_size"]=input_size
    batch=config["training"]["batch_size"]; workers=config["training"]["num_workers"]
    model=MudraClassifier(input_size,config["model"]["hidden_sizes"],config["training"]["dropout"],len(labels))
    # BatchNorm needs batches larger than one; only discard a lone final training sample.
    drop_last = len(train_data) > 1 and len(train_data) % batch == 1
    model,_=train_model(model,DataLoader(train_data,batch_size=batch,shuffle=True,num_workers=workers,drop_last=drop_last),DataLoader(validation_data,batch_size=batch,num_workers=workers),config,labels)
    report=evaluate(model,DataLoader(test_data,batch_size=batch,num_workers=workers),labels); export_onnx(model,labels,config); print(f"Test accuracy: {report['accuracy']:.3f}; ONNX exported to {config['export']['onnx_path']}")

if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--config",default="configs/config.yaml");args=parser.parse_args();main(args.config)
