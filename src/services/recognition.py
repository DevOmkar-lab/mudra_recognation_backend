"""Low-latency, process-lifetime Mudra recognition service."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from src.hand.detector import HandDetector
from src.hand.landmarks import first_hand_landmarks
from src.hand.normalization import landmark_features


class RecognitionError(ValueError):
    """A client-correctable recognition failure."""


class RecognitionService:
    """Keep ONNX Runtime and MediaPipe warm instead of loading per request."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._detector: HandDetector | None = None
        self._detector_lock = threading.Lock()
        self._session: ort.InferenceSession | None = None
        self._labels: dict[int, str] = {}
        self.load_error: str | None = None
        self._load()

    @property
    def ready(self) -> bool:
        return self._session is not None and self._detector is not None

    def _load(self) -> None:
        try:
            export = self.config["export"]
            model_path = Path(export["onnx_path"])
            labels_path = Path(export["labels_path"])
            if not model_path.is_file():
                raise FileNotFoundError(f"ONNX classifier not found: {model_path}. Run python train.py first.")
            if not labels_path.is_file():
                raise FileNotFoundError(f"Label mapping not found: {labels_path}. Run python train.py first.")

            labels = json.loads(labels_path.read_text(encoding="utf-8"))
            self._labels = {int(index): str(name) for index, name in labels.items()}
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.intra_op_num_threads = int(self.config.get("api", {}).get("onnx_intra_op_threads", 1))
            self._session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
            self._detector = HandDetector(**self.config["hand"])
        except Exception as error:
            self.close()
            self.load_error = str(error)

    def close(self) -> None:
        if self._detector is not None:
            self._detector.close()
        self._detector = None
        self._session = None

    def recognize_image(self, image_bytes: bytes) -> dict[str, Any]:
        """Decode one image, extract one hand, and predict its Mudra."""
        if not self.ready:
            raise RuntimeError(self.load_error or "Recognition service is unavailable")
        max_bytes = int(self.config.get("api", {}).get("max_image_bytes", 5_000_000))
        if not image_bytes:
            raise RecognitionError("Image upload is empty")
        if len(image_bytes) > max_bytes:
            raise RecognitionError(f"Image exceeds the {max_bytes:,}-byte limit")
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RecognitionError("Unable to decode image; upload a valid JPEG, PNG, or WebP image")
        max_pixels = int(self.config.get("api", {}).get("max_image_pixels", 16_000_000))
        if image.shape[0] * image.shape[1] > max_pixels:
            raise RecognitionError(f"Image exceeds the {max_pixels:,}-pixel limit")

        # VIDEO mode has state/timestamps, therefore its use is serialized.
        assert self._detector is not None
        with self._detector_lock:
            result = self._detector.process_rgb(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        landmarks = first_hand_landmarks(result)
        if landmarks is None:
            raise RecognitionError("No hand detected. Keep one hand visible and well lit.")
        features = landmark_features(landmarks, **self.config["features"])
        prediction = self.recognize_features(features)
        prediction["landmarks_detected"] = 21
        return prediction

    def recognize_features(self, features: np.ndarray) -> dict[str, Any]:
        """Classify precomputed normalized features using the loaded ONNX model."""
        if not self.ready:
            raise RuntimeError(self.load_error or "Recognition service is unavailable")
        array = np.asarray(features, dtype=np.float32)
        assert self._session is not None
        expected = self._session.get_inputs()[0].shape[-1]
        if array.ndim != 1 or array.size != expected or not np.isfinite(array).all():
            raise RecognitionError(f"Expected {expected} finite normalized feature values")
        logits = self._session.run(None, {self._session.get_inputs()[0].name: array.reshape(1, -1)})[0][0]
        logits = logits - np.max(logits)
        probabilities = np.exp(logits) / np.exp(logits).sum()
        index = int(np.argmax(probabilities))
        confidence = float(probabilities[index])
        threshold = float(self.config.get("api", {}).get("minimum_confidence", 0.60))
        return {
            "class": self._labels[index],
            "confidence": confidence,
            "matched": confidence >= threshold,
            "probabilities": {self._labels[i]: float(probabilities[i]) for i in range(len(self._labels))},
        }
