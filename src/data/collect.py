"""Webcam collector that stores only raw hand landmarks, never full frames."""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
from src.hand.detector import HandDetector
from src.hand.detector import draw_landmarks
from src.hand.landmarks import first_hand_landmarks
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.camera import open_camera, orient_frame

LOG = get_logger(__name__)

def collect(class_name: str, samples: int, camera: int | str | None = None, config_path: str = "configs/config.yaml") -> None:
    """Collect `samples` valid landmarks; SPACE toggles recording, Q quits."""
    config = load_config(config_path)
    destination = Path(config["dataset"]["raw_dir"]) / class_name
    destination.mkdir(parents=True, exist_ok=True)
    camera_cfg = config.get("camera", {}); source = camera if camera is not None else camera_cfg.get("source", 0)
    capture = open_camera(source, camera_cfg.get("width"), camera_cfg.get("height"))
    recording, saved = False, 0
    hand = config["hand"]
    with HandDetector(**hand) as detector:
        while saved < samples:
            ok, frame = capture.read()
            if not ok: raise RuntimeError("Webcam frame capture failed")
            frame = orient_frame(frame, rotation=camera_cfg.get("rotation", 0), flip_horizontal=camera_cfg.get("flip_horizontal", False))
            result = detector.process_rgb(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            points = first_hand_landmarks(result)
            draw_landmarks(frame, result, detector)
            text = f"{class_name}: {saved}/{samples} | {'RECORDING' if recording else 'SPACE to start'}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 0), 2)
            cv2.imshow("Mudra collector (SPACE start/stop, Q quit)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"): break
            if key == ord(" "): recording = not recording
            if recording and points is not None:
                import numpy as np
                np.save(destination / f"webcam_{saved:05d}.npy", points)
                saved += 1
    capture.release(); cv2.destroyAllWindows()
    LOG.info("Saved %s valid landmark samples to %s", saved, destination)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("class_name"); parser.add_argument("--samples", type=int, default=200); parser.add_argument("--camera", help="webcam index or mobile stream URL; config camera.source is used by default"); parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args(); collect(args.class_name, args.samples, args.camera, args.config)
