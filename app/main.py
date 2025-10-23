from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import Response
import base64
from .detector import ProductDetector
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(
    title="YOLO11 Product Detection API",
    description="API para detectar e fazer crop de produtos em embalagens",
    version="1.0.0"
)

# Inicializar detector (será carregado ao iniciar)
detector = None

@app.on_event("startup")
async def startup_event():
    """Carregar modelo ao iniciar a aplicação"""
    global detector
    logger.info("Carregando modelo YOLO11...")
    detector = ProductDetector(model_path='yolo11n.pt', device='cpu')
    logger.info("Modelo carregado com sucesso!")

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": detector is not None
    }

@app.post("/detect")
async def detect_product(
    file: UploadFile = File(...),
    confidence: float = Query(0.3, ge=0.1, le=1.0),
    margin: float = Query(0.1, ge=0.0, le=0.5),
    area_weight: float = Query(0.5, ge=0.0, le=1.0),
    center_weight: float = Query(0.3, ge=0.0, le=1.0),
    conf_weight: float = Query(0.2, ge=0.0, le=1.0)
):
    """
    Detecta produtos e retorna informações + imagem cropada em base64
    
    - **file**: Imagem do produto (JPEG, PNG)
    - **confidence**: Threshold de confiança (0.1-1.0)
    - **margin**: Margem adicional no crop (0-0.5)
    - **area_weight**: Peso da área na seleção
    - **center_weight**: Peso da centralização na seleção
    - **conf_weight**: Peso da confiança na seleção
    """
    if not detector:
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    
    # Validar tipo de arquivo
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail="Arquivo deve ser uma imagem"
        )
    
    try:
        # Ler imagem
        image_bytes = await file.read()
        
        # Processar
        result = detector.detect_and_crop(
            image_bytes,
            conf_threshold=confidence,
            margin_percent=margin,
            return_base64=True
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Erro no processamento: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/image")
async def detect_product_image(
    file: UploadFile = File(...),
    confidence: float = Query(0.3, ge=0.1, le=1.0),
    margin: float = Query(0.1, ge=0.0, le=0.5)
):
    """
    Retorna apenas a imagem cropada (binary)
    Útil para exibir diretamente no navegador
    """
    if not detector:
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    
    try:
        image_bytes = await file.read()
        
        result = detector.detect_and_crop(
            image_bytes,
            conf_threshold=confidence,
            margin_percent=margin,
            return_base64=False
        )
        
        if not result["success"]:
            raise HTTPException(status_code=404, detail=result["message"])
        
        return Response(
            content=result["cropped_bytes"],
            media_type="image/jpeg"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro no processamento: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    """Endpoint raiz com informações da API"""
    return {
        "message": "YOLO11 Product Detection API",
        "endpoints": {
            "/health": "Health check",
            "/detect": "Detectar e retornar JSON com imagem em base64",
            "/detect/image": "Detectar e retornar imagem cropada diretamente",
            "/docs": "Documentação interativa"
        }
    }
