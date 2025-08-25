from __future__ import annotations

import io
import os
from typing import List, Dict, Any, Optional

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
    # Simple bounding box [x1, y1, x2, y2] for easier use
    simple_bbox: List[float]


class DocPreprocessorResult(BaseModel):
    angle: float
    use_doc_orientation_classify: bool
    use_doc_unwarping: bool


class OCRResponse(BaseModel):
    items: List[OCRItem]
    doc_preprocessor: Optional[DocPreprocessorResult] = None
    model_settings: Dict[str, Any]
    text_detection_params: Dict[str, Any]
    text_type: str
    
    # For debugging/advanced use - include raw arrays
    textline_orientation_angles: List[float]


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
        
        # rec_batch_num: default 6. if accuracy drops, lower the number
        # see: http://www.paddleocr.ai/main/FAQ.html#q_36

        # rec_batch_num=1,
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

    # PaddleOCR v3+ predict method
    assert ocr is not None
    results = ocr.predict(input=img)
    
    # Parse the new format
    if not results or len(results) == 0:
        return JSONResponse(content=OCRResponse(
            items=[],
            model_settings={},
            text_detection_params={},
            text_type="general",
            textline_orientation_angles=[]
        ).model_dump())
    
    # Take first result (single image)
    result = results[0]
    
    # Extract data from the new format
    rec_texts = result.get("rec_texts", [])
    rec_scores = result.get("rec_scores", [])
    rec_polys = result.get("rec_polys", [])
    rec_boxes = result.get("rec_boxes", [])
    
    # Build OCR items
    items: List[OCRItem] = []
    for i, text in enumerate(rec_texts):
        score = rec_scores[i] if i < len(rec_scores) else 0.0
        poly = rec_polys[i] if i < len(rec_polys) else [[0,0],[0,0],[0,0],[0,0]]
        simple_box = rec_boxes[i] if i < len(rec_boxes) else [0,0,0,0]
        
        items.append(OCRItem(
            text=text,
            score=float(score),
            bbox=[[float(x), float(y)] for x, y in poly],
            simple_bbox=[float(x) for x in simple_box]
        ))
    
    # Extract preprocessing info
    doc_preprocessor = None
    if "doc_preprocessor_res" in result:
        preproc = result["doc_preprocessor_res"]
        doc_preprocessor = DocPreprocessorResult(
            angle=float(preproc.get("angle", 0)),
            use_doc_orientation_classify=preproc.get("model_settings", {}).get("use_doc_orientation_classify", False),
            use_doc_unwarping=preproc.get("model_settings", {}).get("use_doc_unwarping", False)
        )
    
    response = OCRResponse(
        items=items,
        doc_preprocessor=doc_preprocessor,
        model_settings=result.get("model_settings", {}),
        text_detection_params=result.get("text_det_params", {}),
        text_type=result.get("text_type", "general"),
        textline_orientation_angles=result.get("textline_orientation_angles", [])
    )
    
    return JSONResponse(content=response.model_dump())


@app.post("/ocr/raw")
async def run_ocr_raw(file: UploadFile = File(...)) -> Any:
    """Raw endpoint that returns unprocessed PaddleOCR results for debugging"""
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Please upload an image file.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload.")

    img = _bytes_to_numpy_image(data)
    
    assert ocr is not None
    results = ocr.predict(input=img)
    
    return JSONResponse(content=results)


# Optional small CLI to run with `uv run fast-ocr-server`
def cli() -> None:
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
