"""OpenCV camera/stream connection helpers for Windows and mobile feeds."""
from __future__ import annotations
from typing import Any
import cv2

def camera_source(value: int | str) -> int | str:
    """Convert numeric camera IDs while preserving HTTP/RTSP stream URLs."""
    text = str(value)
    return int(text) if text.isdigit() else text

def open_camera(value: int | str, width: int | None = None, height: int | None = None) -> cv2.VideoCapture:
    """Open a local/OBS camera with DirectShow, or a network URL normally."""
    source = camera_source(value)
    if isinstance(source, int):
        # OBS commonly appears through either DirectShow or the Windows
        # default (usually Media Foundation) backend, depending on version.
        capture = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(source)
    else:
        capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to open camera source {source}. Start OBS Virtual Camera, check its index, and close other apps using the camera.")
    if width: capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height: capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return capture

def orient_frame(frame: Any, *, rotation: int = 0, flip_horizontal: bool = False) -> Any:
    """Apply configured camera orientation before landmark extraction/display."""
    rotation = rotation % 360
    rotations = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    if rotation not in rotations: raise ValueError("camera.rotation must be 0, 90, 180, or 270")
    if rotations[rotation] is not None: frame = cv2.rotate(frame, rotations[rotation])
    if flip_horizontal: frame = cv2.flip(frame, 1)
    return frame
