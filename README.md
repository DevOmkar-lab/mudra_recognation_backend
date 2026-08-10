# Mudra Recognition

A Python 3.11.9, landmark-first ML foundation for classifying hand Mudras. It deliberately does not train on faces, clothing, body, background, or full images. MediaPipe 0.10.35 Tasks Hand Landmarker detects one hand and produces 21 XYZ landmarks; the classifier consumes normalized numeric features.

`models/hand_landmarker/hand_landmarker.task` is MediaPipe's official hand-landmark model asset. It is separate from `models/exported/mudra_model.onnx`, which is your trained Mudra classifier. The detector downloads the official Google asset to that project-relative path when missing; if download is disabled or unavailable, it reports the expected path and official URL.

## Architecture

`camera/image → MediaPipe hand detection → 21 landmarks → shared normalization/features → lightweight PyTorch MLP → Mudra + confidence`

The default Model A input is 63 values (21 × XYZ). Enable `features.use_angles` and/or `features.use_distances` in `configs/config.yaml` to experiment with Models B/C. `train.py` derives the actual input size from processed samples, so the network adapts automatically.

## Install (Windows)

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pytest -q
```

Python 3.11.9 is the intended version. CPU is fully supported; CUDA is selected automatically by PyTorch when available.

## Dataset and collection

### Using a mobile phone camera

Put the phone and computer on the same Wi-Fi network. In an Android IP Webcam/DroidCam app, start the video stream and copy its URL. Set `camera.source` in `configs/config.yaml` to that URL, for example:

```yaml
camera:
  source: http://192.168.1.25:8080/video
  width: 1280
  height: 720
  flip_horizontal: false
```

Alternatively, pass the URL directly:

```powershell
python -m src.data.collect gyan_mudra --samples 300 --camera "http://192.168.1.25:8080/video"
python predict.py --camera --camera-source "http://192.168.1.25:8080/video"
```

The URL differs by app. Use the exact stream URL shown by your phone app; do not use the phone's browser page URL. A local webcam index such as `0` continues to work.

If the DroidCam image is upside down, set `camera.rotation` to `180`. Use `90` or `270` for a sideways feed, or `0` for no rotation.

Create a class directory under `dataset/raw/` (for example, `dataset/raw/new_mudra/`). The sample class folders are only examples. Collect valid landmark samples—no camera frames are saved:

```powershell
python -m src.data.collect gyan_mudra --samples 300
python -m src.data.collect vayu_mudra --samples 300
python -m src.data.collect new_mudra --samples 300
```

Press Space to start/stop collection and Q to quit. For imported images, put images in each class directory; preprocessing will detect only their hands and store generated vectors in `dataset/processed/`.

When source frames come from recordings or people, name them `person_or_recording__frame_001.npy` (or image equivalent). The splitter treats the prefix before `__` as one group, preventing that group from crossing train/validation/test. Webcam samples are individually named, so collect separate sessions with a shared recording prefix if group isolation is needed.

Each valid sample is saved as a `(21, 3)` NumPy landmark array under `dataset/raw/<class>/`. Preprocessing applies the shared translation/scale normalization and writes feature vectors under `dataset/processed/<class>/`.

## Train, evaluate, export

```powershell
python train.py
```

This runs preprocess → group-aware split → train/validation → held-out evaluation → ONNX export. Results are written to `models/checkpoints/`, `models/best_model/`, `evaluation/`, and, only after successful training, `models/exported/mudra_model.onnx` plus `labels.json`. Classes are discovered dynamically from dataset directories; collect enough groups that every split has samples.

## Prediction

```powershell
python predict.py --image path\to\hand.jpg
python predict.py --camera
```

Live inference draws the detected landmarks and calls exactly `src.hand.normalization.landmark_features`, the same function preprocessing uses. Output has `class`, `confidence`, and a probability for every discovered label.

## Important files

- `src/hand/normalization.py`: validation, translation/scale normalization, and extensible feature generation.
- `src/data/collect.py`, `preprocess.py`, `split.py`: collection, conversion, and leakage-aware reproducible splitting.
- `src/training/dataset.py`, `model.py`, `train.py`, `validate.py`, `evaluate.py`: PyTorch data, MLP, metrics, checkpoints, and test reports.
- `src/training/export.py`: ONNX and labels export.
- `src/inference/predictor.py`, `realtime.py`: checkpoint inference and webcam UI.
- `configs/config.yaml`: all operational settings.

## Troubleshooting

- **No hand detected:** improve light/framing, increase `min_hand_detection_confidence` only after checking sample quality.
- **Empty split:** collect more distinct recording/person groups per class.
- **Missing model/labels:** run `python train.py` successfully before prediction.
- **Feature mismatch:** use the same config at prediction as training, particularly the `features` flags.

To verify landmark extraction without a webcam, inspect a processed sample:

```powershell
python -c "import numpy as np; x=np.load('dataset/raw/gyan_mudra/webcam_00000.npy'); print(x.shape, np.isfinite(x).all())"
```

Expected output begins with `(21, 3) True`.

The exported ONNX input is float32 `features` with shape `[batch, feature_count]`; its output is logits `[batch, class_count]`. This clean boundary is ready for a future FastAPI, mobile, desktop, or ONNX Runtime integration, without adding those layers now.

## Recognition API

After training/exporting the model, start the local API:

```powershell
.venv\Scripts\activate
python app.py
```

Open `http://127.0.0.1:8000/docs` for interactive API documentation. Upload a browser camera frame to `POST /api/v1/recognize` as multipart form field `image`; the response includes `class`, `confidence`, `matched`, and all class probabilities. It returns `422` when no hand is detected and `503` when trained artifacts are unavailable.

For the lowest latency, a client that already obtains the same normalized feature vector may send 63 values to `POST /api/v1/recognize/features`. Configure accepted browser origins and request/image limits under `api` in `configs/config.yaml`.
