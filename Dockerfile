FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3-pip python3-venv \
 && rm -rf /var/lib/apt/lists/*

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates libgl1-mesa-glx libglib2.0-0

RUN groupadd -r appuser && useradd -r -g appuser -u 1000 appuser

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen

RUN uv pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

RUN uv pip install opencv-python

RUN uv pip install pynvml

RUN chown -R appuser:appuser /app

# home dir b/c paddle needs to cache

RUN mkdir /home/appuser

RUN chown -R appuser:appuser /home/appuser

# Copy app and switch to non-root
COPY --chown=appuser:appuser . .

USER appuser

ENV PATH="/app/.venv/bin/:${PATH}"

HEALTHCHECK --interval=30s --timeout=10s \
  CMD curl -f http://localhost:80/health || exit 1

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
