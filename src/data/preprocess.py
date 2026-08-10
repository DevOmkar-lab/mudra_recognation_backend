"""Turn raw landmark files or images into validated, normalized feature samples."""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np
from src.hand.detector import HandDetector
from src.hand.landmarks import first_hand_landmarks
from src.hand.normalization import landmark_features
from src.utils.config import load_config
from src.utils.logger import get_logger
LOG = get_logger(__name__)

def _load_raw(path: Path, detector: HandDetector) -> np.ndarray | None:
    if path.suffix.lower() == ".npy": return np.load(path, allow_pickle=False)
    image = cv2.imread(str(path))
    if image is None: return None
    return first_hand_landmarks(detector.process_rgb(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))

def preprocess(config_path: str = "configs/config.yaml") -> int:
    """Process all raw samples. Invalid/corrupt samples are logged and skipped."""
    config = load_config(config_path); raw, processed = Path(config["dataset"]["raw_dir"]), Path(config["dataset"]["processed_dir"])
    if not raw.is_dir(): raise FileNotFoundError(f"Raw dataset directory not found: {raw}")
    classes = sorted(p for p in raw.iterdir() if p.is_dir())
    if not classes: raise ValueError(f"No class directories found in {raw}")
    count = 0; valid_extensions = {".npy", ".jpg", ".jpeg", ".png", ".bmp"}
    with HandDetector(**config["hand"]) as detector:
        for class_dir in classes:
            out = processed / class_dir.name; out.mkdir(parents=True, exist_ok=True)
            for source in class_dir.iterdir():
                if source.suffix.lower() not in valid_extensions: continue
                try:
                    landmarks = _load_raw(source, detector)
                    if landmarks is None: raise ValueError("No 21-landmark hand detected")
                    features = landmark_features(landmarks, **config["features"])
                    np.savez_compressed(out / f"{source.stem}.npz", features=features, label=class_dir.name)
                    count += 1
                except (ValueError, OSError) as error: LOG.warning("Skipping %s: %s", source, error)
    LOG.info("Processed %s samples", count); return count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/config.yaml"); args = parser.parse_args(); preprocess(args.config)
