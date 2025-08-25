# fast-ocr-server

no-bullshit OCR microservice. this uses CUDA so get Jensen's drivers ready.

## requirements

- CUDA enabled GPU, compute cap >= 6.0
- drivers installed
- docker or uv

## run

in docker:

```bash
docker build -t ocr .
docker run --gpus all -p 8000:80 ocr # remember to tweak your params
```

bare metal:

```bash
uv sync
./after-uv-install.sh
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

## endpoints

- `POST /ocr` - structured response with bounding boxes. body field is `file`, pass an image in
- `POST /ocr/raw` - raw PaddleOCR output  
- `GET /gpu` - temperature, memory, utilization
- `GET /status` - system status with GPU info
- `GET /health` - ping

## config

for .env:

```
OCR_LANG=chinese_cht
OCR_USE_DOC_ORIENTATION_CLASSIFY=true
OCR_USE_DOC_UNWARPING=false  
OCR_USE_TEXTLINE_ORIENTATION=true
PORT=8000
```

## dependencies

- CUDA 11.8+
- `pynvml` for GPU monitoring (optional)
- PaddlePaddle GPU 3.0

## notes

temperature warnings logged at 80°C. GPU memory usage tracked per request.

default lang: Chinese. Change `OCR_LANG` for other languages. may output simplified due to language model component. convert if needed

raw endpoint exists because the structured parsing might break with model updates. use accordingly.

## license

apache 2.0
