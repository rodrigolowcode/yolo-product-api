FROM python:3.11-slim

# Metadados
LABEL maintainer="seu-email@example.com"
LABEL description="YOLO11 Product Detection API"

# Instalar dependências do sistema (compatível com Debian Trixie)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgthread-2.0-0 \
    libglx-mesa0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Diretório de trabalho
WORKDIR /app

# Copiar requirements primeiro (melhor cache do Docker)
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código da aplicação
COPY app/ ./app/

# Baixar modelo YOLO11n no build (cache do Docker)
RUN python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')" || true

# Expor porta
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Variáveis de ambiente (podem ser sobrescritas)
ENV WORKERS=1
ENV LOG_LEVEL=info

# Comando para iniciar (usar exec form para melhor signal handling)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
