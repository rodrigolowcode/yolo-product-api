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
    confidence: float = Query(0.3, ge=0.1, le=1.0, description="Threshold de confiança"),
    margin: float = Query(0.1, ge=0.0, le=0.5, description="Margem adicional no crop"),
    max_size: int = Query(1920, ge=640, le=4096, description="Tamanho máximo da imagem"),
    quality: int = Query(85, ge=50, le=100, description="Qualidade JPEG (70-95 recomendado)"),
    area_weight: float = Query(0.5, ge=0.0, le=1.0, description="Peso da área na seleção"),
    center_weight: float = Query(0.3, ge=0.0, le=1.0, description="Peso da centralização"),
    conf_weight: float = Query(0.2, ge=0.0, le=1.0, description="Peso da confiança")
):
    """
    Detecta produtos e retorna imagem otimizada (cropada se encontrado, ou original comprimida)
    
    **Comportamento:**
    - Se produto encontrado: Retorna crop otimizado do produto
    - Se produto NÃO encontrado: Retorna imagem original comprimida e redimensionada
    
    **Resposta sempre inclui:**
    - `success`: true se produto encontrado, false caso contrário
    - `message`: Mensagem descritiva
    - `image`: Imagem em base64 (sempre presente)
    - `file_size_kb`: Tamanho do arquivo processado
    
    **Parâmetros de Detecção:**
    - **file**: Imagem do produto (JPEG, PNG)
    - **confidence**: Threshold de confiança (0.1-1.0)
    - **margin**: Margem adicional no crop quando produto encontrado (0-0.5)
    
    **Parâmetros de Compressão:**
    - **max_size**: Tamanho máximo da imagem (640-4096px)
    - **quality**: Qualidade JPEG (50-100, recomendado 80-90)
    
    **Parâmetros de Seleção:**
    - **area_weight**: Peso da área na seleção do produto
    - **center_weight**: Peso da centralização na seleção
    - **conf_weight**: Peso da confiança na seleção
    """
    if not detector:
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    
    # Validar tipo de arquivo
    if file.content_type and not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail="Arquivo deve ser uma imagem"
        )
    
    try:
        # Ler imagem
        image_bytes = await file.read()
        
        # Processar (sempre retorna resultado, mesmo sem detecção)
        result = detector.detect_and_crop(
            image_bytes,
            conf_threshold=confidence,
            margin_percent=margin,
            return_base64=True,
            max_size=max_size,
            jpeg_quality=quality
        )
        
        # Sempre retorna 200 OK com o resultado
        return result
        
    except Exception as e:
        logger.error(f"Erro no processamento: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail={
                "success": False,
                "message": "Erro interno ao processar imagem",
                "error": str(e),
                "image_processed": False
            }
        )

@app.post("/detect/image")
async def detect_product_image(
    file: UploadFile = File(...),
    confidence: float = Query(0.3, ge=0.1, le=1.0),
    margin: float = Query(0.1, ge=0.0, le=0.5),
    max_size: int = Query(1920, ge=640, le=4096),
    quality: int = Query(85, ge=50, le=100)
):
    """
    Retorna apenas a imagem (binary) - cropada se produto encontrado, ou original comprimida
    
    **Comportamento:**
    - Se produto encontrado: Retorna crop do produto
    - Se produto NÃO encontrado: Retorna imagem original comprimida
    
    **Headers de resposta incluem:**
    - `X-Product-Found`: "true" ou "false"
    - `X-Image-Width`: Largura final
    - `X-Image-Height`: Altura final
    - `X-File-Size-KB`: Tamanho do arquivo
    
    Útil para exibir diretamente no navegador ou salvar
    """
    if not detector:
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    
    try:
        image_bytes = await file.read()
        
        result = detector.detect_and_crop(
            image_bytes,
            conf_threshold=confidence,
            margin_percent=margin,
            return_base64=False,
            max_size=max_size,
            jpeg_quality=quality
        )
        
        # Preparar headers informativos
        headers = {
            "Content-Disposition": "inline; filename=produto_processado.jpg",
            "X-Product-Found": str(result["success"]).lower(),
            "X-Image-Width": str(result["final_size"]["width"]),
            "X-Image-Height": str(result["final_size"]["height"]),
            "X-File-Size-KB": str(result["file_size_kb"]),
            "X-Message": result["message"]
        }
        
        # Adicionar info do produto se encontrado
        if result["success"] and "main_product" in result:
            headers["X-Product-Class"] = result["main_product"]["class"]
            headers["X-Product-Confidence"] = str(result["main_product"]["confidence"])
        
        return Response(
            content=result.get("image_bytes") or result.get("cropped_bytes"),
            media_type="image/jpeg",
            headers=headers
        )
        
    except Exception as e:
        logger.error(f"Erro no processamento: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    """Endpoint raiz com informações da API"""
    return {
        "message": "YOLO11 Product Detection API - ARM64 Optimized",
        "version": "1.0.0",
        "features": [
            "Detecção automática de produtos",
            "Seleção inteligente do produto principal",
            "Crop automático quando produto encontrado",
            "Sempre retorna imagem otimizada (mesmo sem detecção)",
            "Compressão e redimensionamento inteligente",
            "Mantém qualidade visual"
        ],
        "behavior": {
            "product_found": "Retorna crop otimizado do produto",
            "product_not_found": "Retorna imagem original comprimida e redimensionada",
            "always_returns": "Status 200 OK com imagem processada"
        },
        "endpoints": {
            "/health": "Health check",
            "/detect": "Detectar e retornar JSON com imagem em base64",
            "/detect/image": "Detectar e retornar imagem diretamente (binary)",
            "/docs": "Documentação interativa"
        }
    }
