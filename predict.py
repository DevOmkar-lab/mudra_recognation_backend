"""CLI for image or webcam Mudra prediction."""
from __future__ import annotations
import argparse, cv2
from src.hand.detector import HandDetector
from src.hand.landmarks import first_hand_landmarks
from src.hand.normalization import landmark_features
from src.inference.predictor import MudraPredictor
from src.inference.realtime import run
from src.utils.config import load_config

def image_prediction(image_path: str, config_path: str) -> None:
    config=load_config(config_path); image=cv2.imread(image_path)
    if image is None: raise FileNotFoundError(f"Could not read image: {image_path}")
    with HandDetector(**config["hand"]) as detector: points=first_hand_landmarks(detector.process_rgb(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)))
    if points is None: raise ValueError("No hand detected in image")
    print(MudraPredictor(labels_path=config["export"]["labels_path"]).predict(landmark_features(points,**config["features"])))

if __name__ == "__main__":
    parser=argparse.ArgumentParser(); group=parser.add_mutually_exclusive_group(required=True);group.add_argument("--image");group.add_argument("--camera",action="store_true");parser.add_argument("--camera-source",help="webcam index or mobile stream URL; config camera.source is used by default");parser.add_argument("--config",default="configs/config.yaml");args=parser.parse_args()
    run(args.camera_source,args.config) if args.camera else image_prediction(args.image,args.config)
