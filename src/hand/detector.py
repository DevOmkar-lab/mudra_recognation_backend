"""MediaPipe 0.10.x Tasks Hand Landmarker wrapper.

The Tasks API requires an official ``.task`` model asset.  If it is absent,
the official Google-hosted asset is downloaded to the configured project path.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
from urllib.request import urlopen
import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError as error:  # pragma: no cover - exercised only without dependency
    raise ImportError("MediaPipe is required. Install mediapipe==0.10.35.") from error

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
ROOT = Path(__file__).resolve().parents[2]

def ensure_model_asset(model_path: str | Path, *, download: bool = True) -> Path:
    path = Path(model_path)
    if not path.is_absolute(): path = ROOT / path
    if path.is_file(): return path
    if not download:
        raise FileNotFoundError(f"Hand Landmarker model not found. Expected: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(MODEL_URL, timeout=30) as response: path.write_bytes(response.read())
    except Exception as error:
        raise FileNotFoundError(f"Hand Landmarker model not found. Expected: {path}. Download the official asset from {MODEL_URL}") from error
    return path

class HandDetector:
    """Context-managed Hand Landmarker using synchronous frame-by-frame VIDEO mode."""
    def __init__(self, model_path: str | Path = "models/hand_landmarker/hand_landmarker.task", max_num_hands: int = 1, min_hand_detection_confidence: float = .5, min_hand_presence_confidence: float = .5, min_tracking_confidence: float = .5, running_mode: str = "VIDEO", auto_download_model: bool = True, **legacy: Any) -> None:
        # Keep compatibility with the old config key while using Tasks names.
        min_hand_detection_confidence = legacy.get("min_detection_confidence", min_hand_detection_confidence)
        asset = ensure_model_asset(model_path, download=auto_download_model)
        vision = mp.tasks.vision
        mode = getattr(vision.RunningMode, running_mode.upper(), vision.RunningMode.VIDEO)
        options = vision.HandLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(asset)), running_mode=mode, num_hands=max_num_hands, min_hand_detection_confidence=min_hand_detection_confidence, min_hand_presence_confidence=min_hand_presence_confidence, min_tracking_confidence=min_tracking_confidence)
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._timestamp_ms = 0
        self.connections = ((0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17))

    def process_rgb(self, image_rgb: np.ndarray) -> Any:
        """Convert RGB ndarray to ``mp.Image`` and detect at an increasing timestamp."""
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3: raise ValueError("Expected RGB image with shape (H, W, 3)")
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image_rgb))
        self._timestamp_ms += 1
        return self._landmarker.detect_for_video(image, self._timestamp_ms)

    def close(self) -> None: self._landmarker.close()
    def __enter__(self) -> "HandDetector": return self
    def __exit__(self, *_: object) -> None: self.close()

def draw_landmarks(image_bgr: np.ndarray, result: Any, detector: HandDetector) -> None:
    """Draw Tasks landmarks with OpenCV."""
    height, width = image_bgr.shape[:2]
    for hand in getattr(result, "hand_landmarks", []) or []:
        points = [(int(p.x * width), int(p.y * height)) for p in hand]
        for start, end in detector.connections: cv2.line(image_bgr, points[start], points[end], (0, 255, 0), 2)
        for point in points: cv2.circle(image_bgr, point, 3, (0, 0, 255), -1)
