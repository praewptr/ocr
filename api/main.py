from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import easyocr
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

app = FastAPI(title="OCR Inference API", version="1.0")

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

def read_imagefile(file) -> np.ndarray:
    image = Image.open(BytesIO(file))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

@app.post("/ocr")
async def perform_ocr(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
    try:
        image = read_imagefile(await file.read())
        results = reader.readtext(image)
        response = []
        
        for (bbox, text, prob) in results:
            response.append({
                "text": str(text),
                "confidence": round(float(prob), 4),
                "bounding_box": {
                    "top_left": [float(bbox[0][0]), float(bbox[0][1])],
                    "top_right": [float(bbox[1][0]), float(bbox[1][1])],
                    "bottom_right": [float(bbox[2][0]), float(bbox[2][1])],
                    "bottom_left": [float(bbox[3][0]), float(bbox[3][1])]
                }
            })
        
        return {"results": response} 
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)