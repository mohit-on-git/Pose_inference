from pydantic import BaseModel
from typing import Optional, List, Dict

class PoseInferenceResponse(BaseModel):
    filename: str
    detected: bool
    keypoints_count: int
    landmarks: Optional[List[Dict]]