# ============================================================
# Dockerfile — Multilingual Document Intelligence Pipeline
# ============================================================
# Build:  docker build -t doc-intel .
# Run:    docker run -p 8501:8501 doc-intel
# ============================================================

FROM python:3.11-slim

# --- System dependencies ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Tesseract OCR + Hindi language data
    tesseract-ocr \
    tesseract-ocr-hin \
    tesseract-ocr-eng \
    # Poppler for PDF processing
    poppler-utils \
    # OpenCV dependencies
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    # Build tools
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# --- Working directory ---
WORKDIR /app

# --- Copy requirements first (layer cache) ---
COPY requirements.txt .

# --- Install Python dependencies ---
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Copy source code ---
COPY . .

# --- Create necessary directories ---
RUN mkdir -p models/ner models/classifier data/raw data/processed data/synthetic

# --- Streamlit config ---
RUN mkdir -p ~/.streamlit && \
    echo '[server]\nheadless = true\nport = 8501\naddress = "0.0.0.0"\n\n[browser]\ngatherUsageStats = false' > ~/.streamlit/config.toml

# --- Expose Streamlit port ---
EXPOSE 8501

# --- Health check ---
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# --- Entrypoint ---
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
