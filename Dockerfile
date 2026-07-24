# EquityCrew — production image
# Python 3.12 (local dev runs 3.9; the container is not bound by that).
# Runs as UID 1000: required by Hugging Face Spaces, harmless elsewhere.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Dependency layers first so app edits never trigger a torch reinstall.
# CPU-only torch FIRST: sentence-transformers would otherwise pull the CUDA
# build and add ~2GB of GPU libraries this image can never use.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0

COPY requirements.txt .
RUN pip install -r requirements.txt

# Create the runtime user before copying app code, so every file lands
# already-owned — a trailing `chown -R` would duplicate the tree into a layer.
RUN useradd --create-home --uid 1000 app && chown app:app /app
COPY --chown=app:app . .
USER app

# Bake the embedding model in so the first request doesn't pay a cold download
# (and so the container works on a locked-down network).
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

# Shell form so $PORT from the host platform (HF Spaces/Fly/Render) is expanded.
CMD uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}
