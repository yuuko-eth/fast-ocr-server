from __future__ import annotations

import io
import os
from typing import List, Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np

# ------------------------------------------------------------
# Config
# You can override these via environment variables if desired.
# ------------------------------------------------------------
LANG = os.getenv("OCR_LANG", "chinese_cht")
USE_DOC_ORIENTATION_CLASSIFY = os.getenv("OCR_USE_DOC_ORIENTATION_CLASSIFY", "true").lower() == "true"
USE_DOC_UNWARPING = os.getenv("OCR_USE_DOC_UNWARPING", "false").lower() == "true"
USE_TEXTLINE_ORIENTATION = os.getenv("OCR_USE_TEXTLINE_ORIENTATION", "true").lower() == "true"

# Single global OCR instance (heavyweight) – initialized at startup.
ocr: PaddleOCR | None = None

app = FastAPI(title="fast-ocr-server", version="0.1.0")


class OCRItem(BaseModel):
    text: str
    score: float
    # 4-point polygon (PaddleOCR style) – [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    bbox: List[List[float]]


class OCRResponse(BaseModel):
    items: List[OCRItem]
    # Raw PaddleOCR extras can be included later if needed
    # e.g., orientation info when exposed in future


def _bytes_to_numpy_image(data: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return np.array(img)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {e}") from e


@app.on_event("startup")
def _load_models() -> None:
    global ocr
    # Mirror your REPL init exactly (v3.0+ flags you shared)
    ocr = PaddleOCR(
        lang=LANG,
        use_doc_orientation_classify=USE_DOC_ORIENTATION_CLASSIFY,
        use_doc_unwarping=USE_DOC_UNWARPING,
        use_textline_orientation=USE_TEXTLINE_ORIENTATION,
    )
    # NOTE: First call will trigger model downloads if missing; allow time.


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/ocr")
async def run_ocr(file: UploadFile = File(...)) -> Any:
    if file.content_type is None or not file.content_type.startswith("image/"):
        # Paddle can accept PDFs (per-page) in some flows, but we keep it simple here.
        raise HTTPException(status_code=415, detail="Please upload an image file.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload.")

    img = _bytes_to_numpy_image(data)

    # PaddleOCR.  In v3.x, .ocr accepts ndarray or path. It returns a list (batch).
    # Each result item: [ [x1,y1]..[x4,y4], (text, score) ]
    assert ocr is not None
    res_cat = []

    results = ocr.predict(input=img)
    for result in results:
        print(result.json["res"])
        res_cat.append(result.json["res"])

    return res_cat
    
    # If single image, take the first batch element (some builds return already-flat)
    if isinstance(results, list) and results and isinstance(results[0], list) and results and results != []:
        maybe = results[0]
        if maybe and isinstance(maybe, list) and maybe and len(results) == 1:
            results = maybe

    items: List[OCRItem] = []
    for entry in results or []:
        try:
            box, (text, score) = entry
            items.append(OCRItem(text=text, score=float(score), bbox=[[float(x), float(y)] for x, y in box]))
        except Exception:
            # Be tolerant to any shape variations
            continue

    return JSONResponse(content=OCRResponse(items=items).model_dump())


# Optional small CLI to run with `uv run fast-ocr-server`
def cli() -> None:
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)


