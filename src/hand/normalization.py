"""Shared landmark normalization and extensible feature construction."""
from __future__ import annotations
import numpy as np

WRIST, INDEX_MCP, PINKY_MCP = 0, 5, 17

def validate_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Validate and coerce a single MediaPipe hand to a (21, 3) float array."""
    array = np.asarray(landmarks, dtype=np.float32)
    if array.shape != (21, 3): raise ValueError(f"Expected 21x3 landmarks, received {array.shape}")
    if not np.isfinite(array).all(): raise ValueError("Landmarks contain NaN or infinite values")
    return array

def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Center at wrist and scale by palm width while retaining 3-D hand geometry."""
    points = validate_landmarks(landmarks).copy()
    points -= points[WRIST]
    scale = np.linalg.norm(points[INDEX_MCP] - points[PINKY_MCP])
    if scale < 1e-6:
        scale = np.max(np.linalg.norm(points, axis=1))
    if scale < 1e-6: raise ValueError("Cannot normalize degenerate hand landmarks")
    return points / scale

def joint_angles(points: np.ndarray) -> np.ndarray:
    """Return cosine angles for consecutive bones; optional future feature block."""
    points = validate_landmarks(points)
    triplets = [(0,1,2),(1,2,3),(2,3,4),(0,5,6),(5,6,7),(6,7,8),(0,9,10),(9,10,11),(10,11,12),(0,13,14),(13,14,15),(14,15,16),(0,17,18),(17,18,19),(18,19,20)]
    values = []
    for a,b,c in triplets:
        left, right = points[a]-points[b], points[c]-points[b]
        values.append(float(np.dot(left,right) / max(np.linalg.norm(left)*np.linalg.norm(right), 1e-8)))
    return np.asarray(values, dtype=np.float32)

def landmark_features(landmarks: np.ndarray, *, use_xyz: bool = True, use_angles: bool = False, use_distances: bool = False) -> np.ndarray:
    """Build a deterministic feature vector from normalized landmarks.

    All callers must use this function so training and inference are identical.
    """
    normalized = normalize_landmarks(landmarks)
    blocks: list[np.ndarray] = []
    if use_xyz: blocks.append(normalized.reshape(-1))
    if use_angles: blocks.append(joint_angles(normalized))
    if use_distances: blocks.append(np.linalg.norm(normalized[1:] - normalized[0], axis=1))
    if not blocks: raise ValueError("Enable at least one feature type")
    features = np.concatenate(blocks).astype(np.float32)
    if not np.isfinite(features).all(): raise ValueError("Generated invalid features")
    return features

