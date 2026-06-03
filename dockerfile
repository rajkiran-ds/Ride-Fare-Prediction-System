# ============================================================
# RIDESENSE AI — DOCKERFILE
# Runs ML Training Pipeline Only
# Build : docker build -t ridesense-ai .
# Run   : docker run ridesense-ai
# ============================================================

FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip install dist/*.whl || true

CMD ["python", "main.py"]