# Doc Extractor demo image (CPU). Serves the Streamlit app; the fine-tuned model runs on Modal.
#   docker build -t docintel .
#   docker run --rm -p 8501:8501 docintel                                   # recorded examples, no GPU needed
#   docker run --rm -p 8501:8501 -e DOCINTEL_VLM_APP=docintel-vlm \
#       -e MODAL_TOKEN_ID=... -e MODAL_TOKEN_SECRET=... docintel            # live upload through Modal
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-lock.txt .
RUN pip install --no-cache-dir -r requirements-lock.txt

COPY . .
ENV PYTHONUNBUFFERED=1 PYTHONUTF8=1
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
