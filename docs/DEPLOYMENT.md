# Deployment Guide

## 1. Local Development Setup

### Linux / WSL2

```bash
# System dependencies
sudo apt-get update
sudo apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-hin \
    tesseract-ocr-eng \
    poppler-utils \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1

# Python environment
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Verify
python scripts/download_pretrained.py

# Run
streamlit run app.py
```

### Windows (Native)

1. **Tesseract OCR**:
   - Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
   - During install, check "Additional language data" → select Hindi
   - Note the install path (e.g., `C:\Program Files\Tesseract-OCR\`)
   - Add to PATH: System Properties → Environment Variables → PATH → add above path

2. **Poppler**:
   - Download from: https://github.com/oschwartz10612/poppler-windows/releases
   - Extract to `C:\poppler\`
   - Add to PATH: add `C:\poppler\bin`

3. **Set `.env` file**:
   ```
   TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
   POPPLER_PATH=C:\poppler\bin
   ```

4. **Install Python dependencies**:
   ```cmd
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

5. **Run**:
   ```cmd
   streamlit run app.py
   ```

### macOS

```bash
brew install tesseract tesseract-lang poppler
pip install -r requirements.txt
streamlit run app.py
```

---

## 2. Docker Deployment

### Build and Run

```bash
# Build
docker build -t doc-intel .

# Run
docker run -p 8501:8501 doc-intel

# With GPU support (NVIDIA)
docker run --gpus all -p 8501:8501 doc-intel

# With volume for persistent model storage
docker run -p 8501:8501 -v $(pwd)/models:/app/models doc-intel
```

### docker-compose.yml

```yaml
version: "3.9"

services:
  doc-intel:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    environment:
      - LOG_LEVEL=INFO
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

```bash
docker-compose up -d
docker-compose logs -f
```

---

## 3. Cloud Deployment

### Option A: Streamlit Community Cloud (Free)

Best for demos and small-scale use.

**Limitations**: No GPU, limited RAM (1GB), no Tesseract on path by default.

**Workaround for Tesseract**:
Create `packages.txt` in repo root:
```
tesseract-ocr
tesseract-ocr-hin
poppler-utils
```
Streamlit Cloud reads this and installs system packages automatically.

**Steps**:
1. Push code to GitHub (public repo)
2. Go to https://share.streamlit.io → Deploy an app
3. Set repo, branch (`main`), and main file (`app.py`)
4. Deploy

**Note**: Models are too large for Streamlit Cloud's free tier. Use `use_finetuned: false` in config (falls back to zero-shot which downloads at runtime).

### Option B: AWS EC2

```bash
# Launch t3.large (8GB RAM) or g4dn.xlarge (GPU, 16GB RAM)
# AMI: Ubuntu 22.04 LTS

# Connect
ssh -i your-key.pem ubuntu@<ec2-ip>

# Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker ubuntu

# Clone and run
git clone https://github.com/your-username/multilingual-doc-intel.git
cd multilingual-doc-intel
docker build -t doc-intel .
docker run -d -p 80:8501 doc-intel

# For GPU (g4dn instances):
# Install NVIDIA Container Toolkit first
docker run --gpus all -d -p 80:8501 doc-intel
```

Open `http://<ec2-ip>` in browser.

### Option C: HuggingFace Spaces (Free GPU Available)

1. Create a Space at https://huggingface.co/spaces
2. Select Streamlit SDK
3. Upload your code (or link GitHub repo)
4. Add `packages.txt` with `tesseract-ocr tesseract-ocr-hin poppler-utils`
5. Enable GPU (T4 Small) if available

---

## 4. Performance Optimization Tips

### For CPU Deployments

```yaml
# In configs/pipeline_config.yaml:
pipeline:
  parallel_ner_classifier: false  # Disable parallel for single-core

ner:
  max_chunk_length: 500           # Smaller chunks = faster per chunk
```

```python
# In app.py, load models with 4 threads:
import torch
torch.set_num_threads(4)
```

### For GPU Deployments

```python
# In src/ner/model.py and src/classifier/model.py,
# change device=-1 to device=0:
self._pipeline = hf_pipeline(..., device=0)
```

Or set via environment:
```bash
CUDA_VISIBLE_DEVICES=0 streamlit run app.py
```

### Caching Pretrained Models

HuggingFace downloads models to `~/.cache/huggingface/hub/` by default.

For Docker, mount the cache:
```bash
docker run -v ~/.cache/huggingface:/root/.cache/huggingface -p 8501:8501 doc-intel
```

---

## 5. Security Considerations

- **File uploads**: validate type and size (already enforced in `app.py`)
- **No external API calls**: all processing is local
- **No data persistence**: results are in-memory only; cleared on session end
- **Tesseract temp files**: auto-cleaned by pytesseract
- **Reverse proxy**: put nginx in front of Streamlit for production; set `server.headless = true`

### Nginx Config (for production)

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

### Streamlit Config for Production

Create `.streamlit/config.toml`:
```toml
[server]
headless = true
port = 8501
address = "0.0.0.0"
maxUploadSize = 50  # MB

[browser]
gatherUsageStats = false

[logger]
level = "info"
```
