from __future__ import annotations

from contextlib import asynccontextmanager

import io
import os
import logging
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np

# GPU monitoring imports
# Install with: pip install pynvml
try:
    import pynvml
    GPU_MONITORING_AVAILABLE = True
except ImportError:
    pynvml = None
    GPU_MONITORING_AVAILABLE = False

logging.basicConfig(level=logging.INFO)

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ocr
    
    # Initialize GPU monitoring if available
    if GPU_MONITORING_AVAILABLE:
        try:
            pynvml.nvmlInit()
            gpu_count = pynvml.nvmlDeviceGetCount()
            logging.info(f"GPU monitoring initialized. Found {gpu_count} GPU(s)")
            
            # Log initial GPU temps
            gpus = _get_gpu_info()
            for gpu in gpus:
                logging.info(f"GPU {gpu.gpu_id}: {gpu.name} - {gpu.temperature}°C")
        except Exception as e:
            logging.warning(f"GPU monitoring failed to initialize: {e}")
    else:
        logging.info("GPU monitoring not available (pynvml not installed)")
    
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
    logging.info("PaddleOCR initialized successfully")

    yield

    print("Shutting down...")
app = FastAPI(title="fast-ocr-server", version="0.1.0", lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"])

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


class GPUInfo(BaseModel):
    gpu_id: int
    name: str
    temperature: float  # Celsius
    memory_used: int    # MB
    memory_total: int   # MB
    memory_percent: float
    power_draw: Optional[float] = None  # Watts
    utilization: Optional[float] = None  # Percentage


class SystemStatus(BaseModel):
    status: str
    gpu_available: bool
    gpus: List[GPUInfo] = []


def _bytes_to_numpy_image(data: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return np.array(img)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {e}") from e


def _get_gpu_info() -> List[GPUInfo]:
    """Get GPU temperature and stats using pynvml"""
    if not GPU_MONITORING_AVAILABLE:
        return []
    
    try:
        pynvml.nvmlInit()
        gpu_count = pynvml.nvmlDeviceGetCount()
        gpus = []
        
        for i in range(gpu_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            
            # Temperature
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except:
                temp = -1
            
            # Memory info
            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                mem_used = mem_info.used // (1024 * 1024)  # Convert to MB
                mem_total = mem_info.total // (1024 * 1024)
                mem_percent = (mem_info.used / mem_info.total) * 100
            except:
                mem_used = mem_total = mem_percent = 0
            
            # Power draw (optional - not all GPUs support this)
            try:
                power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to Watts
            except:
                power_draw = None
            
            # GPU utilization (optional)
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                utilization = util.gpu
            except:
                utilization = None
            
            gpus.append(GPUInfo(
                gpu_id=i,
                name=name,
                temperature=temp,
                memory_used=mem_used,
                memory_total=mem_total,
                memory_percent=mem_percent,
                power_draw=power_draw,
                utilization=utilization
            ))
        
        return gpus
    except Exception as e:
        logging.error(f"Failed to get GPU info: {e}")
        return []


def _check_gpu_temperature() -> None:
    """Log warning if GPU temperature is too high"""
    gpus = _get_gpu_info()
    for gpu in gpus:
        if gpu.temperature > 80:  # 80°C threshold
            logging.warning(f"GPU {gpu.gpu_id} ({gpu.name}) temperature HIGH: {gpu.temperature}°C")
        elif gpu.temperature > 85:  # 85°C critical
            logging.error(f"GPU {gpu.gpu_id} ({gpu.name}) temperature CRITICAL: {gpu.temperature}°C")




@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/status")
def system_status() -> SystemStatus:
    """Enhanced status with GPU monitoring"""
    gpus = _get_gpu_info()
    return SystemStatus(
        status="ok",
        gpu_available=GPU_MONITORING_AVAILABLE and len(gpus) > 0,
        gpus=gpus
    )


@app.get("/gpu")
def gpu_status() -> Dict[str, Any]:
    """Dedicated GPU monitoring endpoint"""
    if not GPU_MONITORING_AVAILABLE:
        return {
            "available": False,
            "error": "GPU monitoring not available. Install: pip install pynvml"
        }
    
    gpus = _get_gpu_info()
    
    # Check for temperature warnings
    warnings = []
    for gpu in gpus:
        if gpu.temperature > 80:
            warnings.append(f"GPU {gpu.gpu_id} temperature high: {gpu.temperature}°C")
        if gpu.memory_percent > 90:
            warnings.append(f"GPU {gpu.gpu_id} memory usage high: {gpu.memory_percent:.1f}%")
    
    return {
        "available": True,
        "gpu_count": len(gpus),
        "gpus": [gpu.model_dump() for gpu in gpus],
        "warnings": warnings
    }


@app.post("/ocr")
async def run_ocr(file: UploadFile = File(...)) -> Any:
    if file.content_type is None or not file.content_type.startswith("image/"):
        # Paddle can accept PDFs (per-page) in some flows, but we keep it simple here.
        raise HTTPException(status_code=415, detail="Please upload an image file.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload.")

    # Check GPU temperature before processing
    _check_gpu_temperature()

    img = _bytes_to_numpy_image(data)

    # PaddleOCR v3+ predict method
    assert ocr is not None
    results = ocr.predict(input=img)
    
    # Check GPU temperature after processing
    _check_gpu_temperature()
    
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

    # Check GPU temperature before processing
    _check_gpu_temperature()

    img = _bytes_to_numpy_image(data)
    
    assert ocr is not None
    results = ocr.predict(input=img)
    
    # Check GPU temperature after processing
    _check_gpu_temperature()
    
    return JSONResponse(content=results)


# Optional small CLI to run with `uv run fast-ocr-server`
def cli() -> None:
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
