# Two stages. The builder installs the Python packages into a virtual environment, with compilers available in case
# a package has no ready-made wheel. The final image copies only that environment and the application (see
# .dockerignore), has no compilers, and runs as an ordinary user. Package versions come from requirements.lock.

FROM python:3.12-slim-trixie AS builder

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt requirements.lock ./
# CPU-only torch: the embedding, re-ranking and NLI models are small, and the LLM runs in the Ollama container,
# which has its own GPU support. The default (CUDA) build of torch would add several GB for nothing.
RUN pip install torch -c requirements.lock --extra-index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt -c requirements.lock


FROM python:3.12-slim-trixie

# libgomp1: threads for torch and onnxruntime. libgl1, libglib2.0-0: OpenCV, which the OCR of scanned pages needs.
# curl: health checks. age: encrypts backups (scripts/backup.py) and decrypts them for a restore test.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libgl1 libglib2.0-0 curl age \
    && rm -rf /var/lib/apt/lists/*

# A fixed user id, so that host folders mounted into the container can be given to it: chown -R 10001:10001 <dir>
RUN useradd --create-home --uid 10001 --user-group atts

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/models/hf

WORKDIR /app
# The code belongs to root and cannot be changed by the application. Only the folders it writes to (user database,
# vector store, logs, downloaded models) belong to the application user; Docker copies this ownership into new
# volumes mounted on these paths.
COPY . .
RUN mkdir -p /app/data /app/chroma_db /app/logs /models/hf \
    && chown -R atts:atts /app/data /app/chroma_db /app/logs /models

USER atts

EXPOSE 8000 8501

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
