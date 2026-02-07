# ML Microservice Spec

A reusable pattern for wrapping an accelerated ML model in a production-ready FastAPI service. Supply this document alongside the target model's API docs to scaffold a new service.

---

## 1. Project Structure

```
my-model-server/
├── main.py              # FastAPI app, endpoints, lifespan
├── device.py            # Runtime/device detection
├── Dockerfile           # Multi-stage, non-root, health check
├── pyproject.toml       # uv/pip project metadata
├── uv.lock
└── CLAUDE.md            # Project-specific instructions for Claude
```

Keep it flat. One `main.py` unless the model needs a dedicated wrapper module (e.g. pre/post-processing logic > ~80 lines — then split into `inference.py`).

---

## 2. Device Detection (`device.py`)

Auto-detect the best available accelerator. The service must start on any machine — never crash due to missing GPU.

```python
import logging

def get_device() -> str:
    """Return the best available device string: 'cuda', 'mps', or 'cpu'."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            logging.info(f"Using CUDA device: {name}")
            return "cuda"
        if torch.backends.mps.is_available():
            logging.info("Using MPS (Apple Silicon)")
            return "mps"
    except ImportError:
        pass  # torch not installed — model may use its own runtime

    logging.info("Falling back to CPU")
    return "cpu"
```

For models that don't use PyTorch (e.g. PaddleOCR uses PaddlePaddle, Whisper.cpp uses GGML):
- Check the framework's own device query (e.g. `paddle.device.get_device()`, onnxruntime providers).
- The function still returns a string; callers map it to the framework's device enum.

Pass the result into model init — never let a library silently pick a device.

---

## 3. FastAPI App Skeleton

### 3.1 Lifespan — Model Loading

Load the model once at startup using FastAPI's `lifespan` context manager. Store it as a module-level global.

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

model: MyModel | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    device = get_device()
    model = MyModel(device=device, **config)
    logging.info("Model loaded")
    yield
    # Optional cleanup (unload, release VRAM, etc.)

app = FastAPI(title="my-model-server", version="0.1.0", lifespan=lifespan)
```

**Rules:**
- Model init may download weights on first run — this is expected. Log it.
- Never lazy-load on first request. Startup should be the only slow path.
- The global is `None` until lifespan completes — endpoints check for this.

### 3.2 Configuration via Environment

All tunables come from env vars with sensible defaults. Define them at module top-level.

```python
import os

MODEL_NAME      = os.getenv("MODEL_NAME", "default-variant")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "20")) * 1024 * 1024
INFER_TIMEOUT   = int(os.getenv("INFER_TIMEOUT_SECONDS", "60"))
```

Common env vars every service should support:

| Env var | Default | Purpose |
|---------|---------|---------|
| `PORT` | `8000` | Listen port |
| `MODEL_NAME` | model-specific | Variant/checkpoint name |
| `MAX_UPLOAD_MB` | `20` | Upload size cap |
| `INFER_TIMEOUT_SECONDS` | `60` | Per-request timeout |
| `LOG_LEVEL` | `INFO` | Python logging level |

### 3.3 CORS

Always set all three wildcard fields so preflight (OPTIONS) requests work out of the box. Tighten `allow_origins` for production if needed.

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 4. Required Endpoints

### 4.1 Liveness — `GET /health`

Always returns 200. Used by Docker HEALTHCHECK and k8s `livenessProbe`. No dependency on model state.

```python
@app.get("/health")
def health():
    return {"status": "ok"}
```

### 4.2 Readiness — `GET /ready`

Returns 200 only when the model is loaded and ready to serve. Returns 503 otherwise. Used by k8s `readinessProbe` and load balancers.

```python
@app.get("/ready")
def readiness():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return {"status": "ready"}
```

### 4.3 Inference — `POST /infer` (or domain-specific name)

The main endpoint. Name it after what it does: `/ocr`, `/segment`, `/transcribe`, `/detect`, etc.

**Request format** — pick the simplest that fits:

| Input type | Accept as |
|-----------|-----------|
| Single image | `UploadFile` (multipart) |
| Audio file | `UploadFile` (multipart) |
| Image + params | `UploadFile` + `Form()` fields |
| JSON-only | Pydantic `Body` model |
| Batch of images | Multiple `UploadFile` or JSON with base64 |

**Response format** — always return structured JSON via a Pydantic model, not raw dicts.

---

## 5. Inference Execution Pattern

Every inference endpoint must follow this exact pattern:

```python
import asyncio

@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    # 1. Validate input
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large.")

    # 2. Check model readiness
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    # 3. Preprocess (sync, fast — OK on event loop)
    input_tensor = preprocess(data)

    # 4. Run inference off the event loop, with timeout
    loop = asyncio.get_event_loop()
    try:
        raw = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: model.predict(input_tensor)),
            timeout=INFER_TIMEOUT,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Inference timed out.")

    # 5. Postprocess and return
    return postprocess(raw)
```

**Why `run_in_executor`**: ML inference is CPU/GPU-bound. Running it directly in an `async def` blocks the entire event loop — health checks, concurrent requests, and timeouts all freeze. The thread pool executor releases the loop while inference runs.

**Why `wait_for`**: Pathological inputs can cause models to hang or run extremely long. The timeout returns a 504 instead of blocking forever.

---

## 6. Error Handling

Use HTTP status codes consistently:

| Code | When |
|------|------|
| 400 | Bad/unparseable input (corrupt image, wrong format) |
| 413 | Upload exceeds `MAX_UPLOAD_BYTES` |
| 415 | Wrong content type |
| 422 | Pydantic validation failure (automatic from FastAPI) |
| 503 | Model not loaded yet |
| 504 | Inference timeout |

Never let raw Python exceptions (assertions, KeyError, CUDA OOM) bubble up as 500s. Catch known failure modes from the model and map them:

```python
try:
    raw = await asyncio.wait_for(...)
except asyncio.TimeoutError:
    raise HTTPException(status_code=504, detail="Inference timed out.")
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        raise HTTPException(status_code=503, detail="GPU out of memory.")
    raise
```

---

## 7. GPU Monitoring (Optional)

If the service runs on NVIDIA GPUs, add a `GET /gpu` endpoint using `pynvml`. Make the import conditional so the service still works without it.

```python
try:
    import pynvml
    GPU_MONITORING = True
except ImportError:
    GPU_MONITORING = False
```

Log warnings when temperature exceeds thresholds. Check higher thresholds first:

```python
if gpu.temperature > 85:
    logging.error(f"GPU {i} CRITICAL: {gpu.temperature}C")
elif gpu.temperature > 80:
    logging.warning(f"GPU {i} HIGH: {gpu.temperature}C")
```

---

## 8. Dockerfile

```dockerfile
FROM python:3.12-slim AS base

# System deps for the specific model (e.g. libgl1 for OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    <model-specific-deps> \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd -m appuser
USER appuser
WORKDIR /app

# Install uv, then project deps
COPY --chown=appuser pyproject.toml uv.lock ./
RUN pip install uv && uv sync --frozen

COPY --chown=appuser . .

ENV PATH="/home/appuser/.local/bin:/app/.venv/bin:${PATH}"

HEALTHCHECK --interval=30s --timeout=10s \
  CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

EXPOSE ${PORT:-8000}
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "${PORT:-8000}"]
```

**Rules:**
- HEALTHCHECK targets `/health` (liveness), never `/ready`.
- Use `curl -f` so non-2xx exits with code 22.
- For GPU images, base on `nvidia/cuda:12.x-runtime-*` instead of `python:3.12-slim`.
- Keep the image minimal. No dev tools, no build deps in the final stage.

---

## 9. Kubernetes Probes

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /ready
    port: 8000
  initialDelaySeconds: 10    # model loading takes time
  periodSeconds: 5
  failureThreshold: 30       # allow up to ~2.5 min for large models

startupProbe:
  httpGet:
    path: /ready
    port: 8000
  periodSeconds: 10
  failureThreshold: 60       # allow up to 10 min for first-run weight downloads
```

---

## 10. Checklist for Wrapping a New Model

When starting a new service, supply this spec + the target model's API docs, then:

- [ ] **Identify the inference call**: What's the equivalent of `model.predict(input)`? What does it accept (numpy array, PIL Image, file path, tensor)? What does it return?
- [ ] **Identify input type**: Image, audio, video, text, or multi-modal?
- [ ] **Identify device support**: Does the model/framework support CUDA? MPS? How is the device passed in?
- [ ] **Identify heavyweight init**: What constructor loads the weights? What args does it take (model name, device, precision)?
- [ ] **Identify output structure**: What fields does the raw output contain? Design Pydantic response models from this.
- [ ] **Identify system deps**: Does it need `libgl1`, `ffmpeg`, `libsndfile1`, etc.?
- [ ] **Implement** `device.py`, `main.py`, `Dockerfile` following sections 2-8 above.
- [ ] **Verify**: `GET /health` returns 200 immediately. `GET /ready` returns 503 before model loads, 200 after. Inference endpoint returns structured JSON. Upload > `MAX_UPLOAD_MB` returns 413. Timeout returns 504.

---

## 11. Example: What to Supply for a New Model

To wrap model X, provide:

1. **This spec** (ML_MICROSERVICE.md)
2. **Model API docs** — the library's Python usage example showing:
   - How to instantiate the model
   - How to run inference
   - What the output format looks like
3. **Any special requirements** — e.g. "must support batch input", "needs streaming output for long audio", "returns masks as numpy arrays that should be encoded as PNG"

That's enough to generate a complete, production-ready service.
