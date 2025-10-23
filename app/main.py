from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Security, Header
from fastapi.security import APIKeyHeader
from fastapi.responses import Response
import base64
from .detector import ProductDetector
import logging
import os

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Key Configuration
API_KEY = os.getenv("API_KEY", "")  # Carregar da variável de ambiente
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def verify_api_key(api_key: str = Security(api_key_header)):
    """
    Valida a API Key recebida no header
    """
    # Se API_KEY não foi configurada, permitir acesso (modo desenvolvimento)
    if not API_KEY:
        logger.warning("⚠️  API_KEY não configurada - modo desenvolvimento ativo!")
        return True
    
    # Validar API Key
    if api_key != API_KEY:
        logger.warning(f"❌ Tentativa de acesso com API Key inválida")
        raise HTTPException(
            status_code=403,
            detail="API Key inválida ou ausente. Inclua o header X-API-Key na requisição."
        )
    
    return True

# Inicializar FastAPI
app = FastAPI(
    title="YOLO11 Product Detection API",
    description="API protegida para detectar e fazer crop de produtos em embalagens - Otimizada para LLM",
    version="1.0.0"
)

# Inicializar detector (será carregado ao iniciar)
detector = None

@app.on_event("startup")
async def startup_event():
    """Carregar modelo ao iniciar a aplicação"""
    global detector
    logger.info("🚀 Iniciando YOLO11 Product Detection API...")
    
    # Verificar se API Key está configurada
    if API_KEY:
        logger.info(f"🔒 API Key configurada - autenticação ativada")
    else:
        logger.warning("⚠️  API Key NÃO configurada - acesso público permitido!")
    
    logger.info("📦 Carregando modelo YOLO11...")
    detector = ProductDetector(model_path='yolo11n.pt', device='cpu')
    logger.info("✅ Modelo carregado com sucesso!")

@app.get("/health")
def health_check():
    """
    Health check endpoint (público - sem autenticação)
    """
    return {
        "status": "healthy",
        "model_loaded": detector is not None,
        "authentication": "enabled" if API_KEY else "disabled"
    }

@app.post("/detect/image")
async def detect_product_image(
    file: UploadFile = File(...),
    confidence: float = Query(0.3, ge=0.1, le=1.0, description="Threshold de confiança"),
    margin: float = Query(0.1, ge=0.0, le=0.5, description="Margem no crop"),
    max_size: int = Query(1920, ge=640, le=4096, description="Tamanho máximo"),
    quality: int = Query(85, ge=50, le=100, description="Qualidade JPEG"),
    authenticated: bool = Security(verify_api_key)
):
    """
    **[RECOMENDADO PARA LLM]** 🔒 Requer API Key
    
    Retorna SEMPRE uma imagem otimizada (binário JPEG):
    - Se produto encontrado: retorna crop do produto
    - Se produto NÃO encontrado: retorna imagem original comprimida
    
    **Autenticação:**
    Inclua o header: `X-API-Key: sua_chave_aqui`
    
    **Comportamento:**
    - Sempre HTTP 200 (nunca falha)
    - Sempre retorna imagem JPEG otimizada
    - Headers HTTP indicam se produto foi encontrado
    - Ideal para enviar direto para LLM
    
    **Headers de Resposta:**
    - `X-Product-Found`: "true" ou "false"
    - `X-Product-Class`: Nome da classe (se encontrado)
    - `X-Product-Confidence`: Confiança 0-1 (se encontrado)
    - `X-Image-Width`: Largura final
    - `X-Image-Height`: Altura final
    - `X-File-Size-KB`: Tamanho do arquivo
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
            "Content-Disposition": "inline; filename=produto_otimizado.jpg",
            "X-Product-Found": str(result["success"]).lower(),
            "X-Image-Width": str(result["final_size"]["width"]),
            "X-Image-Height": str(result["final_size"]["height"]),
            "X-File-Size-KB": str(result["file_size_kb"]),
            "X-Detections-Total": str(result.get("detections", 0)),
            "X-Image-Processed": "true"
        }
        
        # Adicionar info do produto se encontrado
        if result["success"] and "main_product" in result:
            headers["X-Product-Class"] = result["main_product"]["class"]
            headers["X-Product-Confidence"] = str(result["main_product"]["confidence"])
            headers["X-Selection-Score"] = str(result["main_product"]["selection_score"])
        
        return Response(
            content=result.get("image_bytes"),
            media_type="image/jpeg",
            headers=headers
        )
        
    except Exception as e:
        logger.error(f"Erro no processamento: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect")
async def detect_product(
    file: UploadFile = File(...),
    confidence: float = Query(0.3, ge=0.1, le=1.0, description="Threshold de confiança"),
    margin: float = Query(0.1, ge=0.0, le=0.5, description="Margem no crop"),
    max_size: int = Query(1920, ge=640, le=4096, description="Tamanho máximo"),
    quality: int = Query(85, ge=50, le=100, description="Qualidade JPEG"),
    authenticated: bool = Security(verify_api_key)
):
    """
    **[PARA APPS/DEBUG]** 🔒 Requer API Key
    
    Retorna JSON completo com metadados + imagem em base64
    
    **Autenticação:**
    Inclua o header: `X-API-Key: sua_chave_aqui`
    
    Útil quando você precisa:
    - Saber SE o produto foi encontrado
    - Ver confiança e coordenadas de detecção
    - Debugging e análise
    - Interfaces que precisam de metadados
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
        image_bytes = await file.read()
        
        result = detector.detect_and_crop(
            image_bytes,
            conf_threshold=confidence,
            margin_percent=margin,
            return_base64=True,
            max_size=max_size,
            jpeg_quality=quality
        )
        
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

@app.get("/")
def read_root():
    """Endpoint raiz com informações da API (público)"""
    return {
        "message": "YOLO11 Product Detection API - ARM64 Optimized for LLM",
        "version": "1.0.0",
        "authentication": {
            "required": bool(API_KEY),
            "header": "X-API-Key",
            "status": "enabled" if API_KEY else "disabled (development mode)"
        },
        "recommendation": {
            "for_llm": "Use /detect/image (retorna binário direto)",
            "for_apps": "Use /detect (retorna JSON + base64)"
        },
        "endpoints": {
            "/health": "Health check (público)",
            "/detect/image": "Retorna binário JPEG otimizado 🔒",
            "/detect": "Retorna JSON completo 🔒",
            "/docs": "Documentação interativa"
        }
    }
