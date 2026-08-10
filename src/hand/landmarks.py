"""Stable NumPy representation for MediaPipe Tasks results."""
from __future__ import annotations
import numpy as np
from .normalization import validate_landmarks

def first_hand_landmarks(result: object) -> np.ndarray | None:
    """Extract the first Tasks hand as exactly ``(21, 3)``, or return None."""
    hands = getattr(result, "hand_landmarks", None)
    if not hands: return None
    points = hands[0]
    return validate_landmarks(np.asarray([[p.x, p.y, p.z] for p in points], dtype=np.float32))

def draw_landmarks(image: object, result: object, detector: object) -> None:
    """Backward-compatible forwarding helper for callers of the old module."""
    from .detector import draw_landmarks as _draw
    _draw(image, result, detector)
