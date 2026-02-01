from fastapi import  File, UploadFile, APIRouter, HTTPException
from ..models.schemas import PoseInferenceResponse
import numpy as np
import cv2
from ..services.pose import infer_pose

router = APIRouter()

@router.post("/infer/pose", response_model=PoseInferenceResponse)
async def upload_photo(image: UploadFile = File(...)) -> PoseInferenceResponse:
    contents = await image.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")
    # HTTP streams send data in chunks over standard HTTP
    # We convert the bytes into a numerical array where each number is 8bits
    # Need to do this because we are working in a HTTP stream and cannot use cv2.imread
    # and need to use imdecode instead. imreade handles File I/O and decodes vs imdecode needs
    # an array to be fed to decode. Hence, we use numpy to convert the bytes buffer to array
    # `uint8` is REQUIRED because:
    # - each byte = 8 bits
    # - file data is not floating point
    # - OpenCV expects uint8 buffers for decoding

    image_arr = np.frombuffer(contents, dtype=np.uint8)

    # `cv2.imdecode` does ONLY decoding.
    # It does NOT perform file I/O.
    #
    # This is necessary because:
    # - We don't have a file path
    # - The image came from an HTTP request, not disk
    #
    # Result:
    # - NumPy array
    # - shape: (height, width, channels)
    # - dtype: uint8
    # - color format: BGR (OpenCV default)
    img_bgr = cv2.imdecode(image_arr, flags= cv2.IMREAD_COLOR )
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # return {"filename": image.filename,   #Type:ignore
    #         "detected": False,
    #         "keypoints_count": 0}

    landmark = infer_pose(img_rgb)
    detected = landmark is not None
    keypoints = len(landmark) if landmark else 0

    return {"filename": image.filename,  # Type:ignore
            "detected": detected,
            "keypoints_count": keypoints,
            "landmarks": landmark}


