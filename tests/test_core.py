import json
from pathlib import Path
import numpy as np
import torch
import mediapipe as mp
from src.hand.normalization import landmark_features, normalize_landmarks
from src.training.dataset import MudraDataset, save_labels
from src.training.model import MudraClassifier
from src.inference.predictor import MudraPredictor
from src.utils.config import load_config

def points() -> np.ndarray:
    values=np.arange(63,dtype=np.float32).reshape(21,3); values[5]=[1,0,0]; values[17]=[0,1,0]; return values

def test_config_loads(): assert load_config()["project"]["name"] == "mudra-recognition"
def test_mediapipe_tasks_api(): assert hasattr(mp, "tasks") and hasattr(mp.tasks.vision, "HandLandmarker")
def test_normalization_and_feature_shape():
    normalized=normalize_landmarks(points()); assert normalized.shape==(21,3); assert np.allclose(normalized[0],0); assert landmark_features(points()).shape==(63,)
def test_dataset_and_labels(tmp_path: Path):
    for label in ("alpha","beta"):
        folder=tmp_path/label;folder.mkdir();np.savez_compressed(folder/"one.npz",features=np.ones(63,dtype=np.float32),label=label)
    dataset=MudraDataset(tmp_path); features,target=dataset[0]; assert features.shape==(63,) and target.item() in (0,1)
def test_model_forward(): assert MudraClassifier(63,[16],.1,3)(torch.zeros(2,63)).shape == (2,3)
def test_predictor_output(tmp_path: Path):
    model=MudraClassifier(63,[8],.1,2); checkpoint=tmp_path/"model.pt"; torch.save({"model_state":model.state_dict(),"metadata":{"input_size":63,"hidden_sizes":[8],"dropout":.1}},checkpoint); labels=tmp_path/"labels.json";save_labels({0:"a",1:"b"},labels)
    result=MudraPredictor(checkpoint,labels).predict(np.zeros(63,dtype=np.float32)); assert set(result)=={"class","confidence","probabilities"}
