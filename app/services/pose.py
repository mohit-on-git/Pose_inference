import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
from typing import Optional, List, Dict

MODEL_PATH = "app/models/pose_landmarker_full.task"

options = vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.IMAGE,
    output_segmentation_masks=False
)

pose = vision.PoseLandmarker.create_from_options(options)



def infer_pose(image: np.ndarray) -> Optional[List[Dict]] :
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    results = pose.detect(mp_image)
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




