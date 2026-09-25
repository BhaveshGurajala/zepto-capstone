FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ANONYMIZED_TELEMETRY=False \
    HF_HOME=/app/.hf_cache \
    MOCK_LLM=1

WORKDIR /app

# CPU-only torch keeps the image ~1.5 GB smaller than the default CUDA build
RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download all-MiniLM-L6-v2 at build time so the container starts without network access
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
# From here on, only use the cached model - never reach out to Hugging Face at run time
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

COPY app ./app
COPY docs ./docs
COPY main.py .

# Build the ChromaDB index into the image as well
RUN python -m app.ingest

EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
