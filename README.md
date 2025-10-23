# YOLO11 Product Detection API

API REST para detecção automática e crop de produtos em embalagens usando YOLO11 otimizado para CPU.

## Funcionalidades

- ✅ Detecção automática de produtos com YOLO11
- ✅ Seleção inteligente do produto principal (área + centralização + confiança)
- ✅ Crop automático com margem configurável
- ✅ Otimizado para CPU com OpenVINO (40+ FPS)
- ✅ API REST com FastAPI
- ✅ Docker pronto para deploy
- ✅ Documentação interativa (Swagger)

## Deploy no EasyPanel

1. Faça fork deste repositório
2. No EasyPanel, crie um novo App Service
3. Conecte ao seu repositório GitHub
4. Configure a porta: **8000**
5. Deploy automático!

## Endpoints

- `GET /health` - Health check
- `POST /detect` - Detecta e retorna JSON + imagem em base64
- `POST /detect/image` - Detecta e retorna apenas a imagem cropada
- `GET /docs` - Documentação interativa

## Exemplo de Uso
curl -X POST https://seu-dominio.com/detect 
-F "file=@produto.jpg" 
-F "confidence=0.3" 
-F "margin=0.15"


## Desenvolvimento Local

Instalar dependênciaspip install -r requirements.txtRodar servidoruvicorn app.main:app --reloadAcessar docshttp://localhost:8000/docs
## Docker
Builddocker build -t yolo-product-api .Rundocker run -p 8000:8000 yolo-product-api
## Requisitos

- Python 3.11+
- CPU com 2+ cores (recomendado)
- 2-4GB RAM
