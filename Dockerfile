FROM python:3.11-slim

LABEL maintainer="seu-email@example.com"
LABEL description="YOLO11 Product Detection API - ARM64 Optimized"

# Instalar dependências do sistema (ARM64 compatível)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libgthread-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar requirements
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY app/ ./app/

# Baixar modelo YOLO11n
RUN python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')" || true

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Variáveis de ambiente (podem ser sobrescritas)
ENV API_KEY=""
ENV WORKERS=1
ENV LOG_LEVEL=info

# Comando para iniciar
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
