import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
from typing import Optional, List, Dict


class PoseService:
    def __init__(self, model_path: str):
        self._model = model_path
        self._model_instance = self._load_model()

    def _load_model(self) -> vision.PoseLandmarker:
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self._model),
            running_mode=vision.RunningMode.IMAGE,
            output_segmentation_masks=False)
        return vision.PoseLandmarker.create_from_options(options)

    def infer_pose_landmarks(self, image: np.ndarray) -> Optional[List[Dict]]:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        results = self._model_instance.detect(mp_image)
        height, width, _ = image.shape
        landmarks = []
        if not results.pose_landmarks:
            return None
        for lm in results.pose_landmarks[0]:
            landmarks.append({
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": lm.visibility
                })
        return landmarks






