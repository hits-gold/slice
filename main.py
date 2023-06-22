from detectarea import DetectArea
from fastapi import FastAPI, UploadFile
from fastapi.responses import Response
from PIL import Image
import cv2

import io
import numpy as np

app = FastAPI(default_response_class=Response)

@app.post("/upload_image")
async def upload_image(file: UploadFile) -> Response:
    # with Image.open(file.file) as img:
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    detect = DetectArea(img)
    img = detect.resizing()     
    img = Image.fromarray(img)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    processed_image_bytes = buffer.getvalue()
    
    return Response(content=processed_image_bytes, media_type="image/jpeg")