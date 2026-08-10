"""Real-time webcam inference reusing the training feature function."""
from __future__ import annotations
import argparse
import cv2
from src.hand.detector import HandDetector, draw_landmarks
from src.hand.landmarks import first_hand_landmarks
from src.hand.normalization import landmark_features
from src.inference.predictor import MudraPredictor
from src.utils.config import load_config
from src.utils.camera import open_camera, orient_frame

def run(camera: int | str | None = None, config_path: str = "configs/config.yaml") -> None:
    config=load_config(config_path); camera_cfg=config.get("camera", {}); source=camera if camera is not None else camera_cfg.get("source", 0); predictor=MudraPredictor(labels_path=config["export"]["labels_path"]); capture=open_camera(source, camera_cfg.get("width"), camera_cfg.get("height"))
    with HandDetector(**config["hand"]) as detector:
        while True:
            ok,frame=capture.read()
            if not ok: raise RuntimeError("Webcam frame capture failed")
            frame=orient_frame(frame, rotation=camera_cfg.get("rotation", 0), flip_horizontal=camera_cfg.get("flip_horizontal", False))
            result=detector.process_rgb(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)); points=first_hand_landmarks(result); draw_landmarks(frame,result,detector)
            text="No hand detected"
            if points is not None:
                try:
                    output=predictor.predict(landmark_features(points,**config["features"])); text=f"{output['class']}: {output['confidence']:.1%}"
                except ValueError as error: text=f"Invalid hand: {error}"
            cv2.putText(frame,text,(10,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,0),2); cv2.imshow("Mudra recognition (Q to quit)",frame)
            if cv2.waitKey(1)&0xFF==ord("q"): break
    capture.release();cv2.destroyAllWindows()

if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--camera",help="webcam index or mobile stream URL");parser.add_argument("--config",default="configs/config.yaml");args=parser.parse_args();run(args.camera,args.config)
