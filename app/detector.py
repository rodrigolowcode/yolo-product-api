import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)

class ProductDetector:
    def __init__(self, model_path='yolo11n.pt', device='cpu'):
        """Inicializa o detector YOLO11 otimizado para ARM64"""
        self.model = YOLO(model_path)
        self.device = device
        self.optimized_model = None
        
        # Tentar exportar para NCNN (melhor para ARM64)
        try:
            logger.info("Exportando modelo para NCNN (otimizado para ARM64)...")
            self.model.export(format='ncnn', imgsz=640)
            ncnn_path = model_path.replace('.pt', '_ncnn_model')
            self.optimized_model = YOLO(ncnn_path)
            logger.info("Modelo NCNN carregado com sucesso!")
        except Exception as e:
            logger.warning(f"NCNN export falhou: {e}")
            
            # Fallback para ONNX Runtime
            try:
                logger.info("Tentando ONNX Runtime...")
                self.model.export(format='onnx', simplify=True, imgsz=640)
                onnx_path = model_path.replace('.pt', '.onnx')
                self.optimized_model = YOLO(onnx_path, task='detect')
                logger.info("Modelo ONNX carregado com sucesso!")
            except Exception as e2:
                logger.warning(f"ONNX export falhou: {e2}")
                logger.info("Usando modelo PyTorch padrão (mais lento)")
                self.optimized_model = self.model
        
        # Usar modelo otimizado se disponível
        if self.optimized_model:
            self.model = self.optimized_model
    
    def select_main_product(self, results, img_shape, 
                           area_weight=0.5, 
                           center_weight=0.3, 
                           conf_weight=0.2):
        """
        Seleciona o produto principal baseado em área, centralização e confiança
        """
        img_h, img_w = img_shape[:2]
        img_center = np.array([img_w / 2, img_h / 2])
        img_area = img_h * img_w
        
        best_box = None
        best_score = 0
        
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
                
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # 1. Score de área (normalizado)
                box_area = (x2 - x1) * (y2 - y1)
                area_score = box_area / img_area
                
                # 2. Score de centralização (invertido e normalizado)
                box_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                distance = np.linalg.norm(img_center - box_center)
                max_distance = np.linalg.norm(img_center)
                center_score = 1 - (distance / max_distance)
                
                # 3. Score de confiança
                conf_score = float(box.conf[0].cpu().numpy())
                
                # Score combinado
                total_score = (area_score * area_weight + 
                              center_score * center_weight + 
                              conf_score * conf_weight)
                
                if total_score > best_score:
                    best_score = total_score
                    best_box = box
        
        return best_box, best_score
    
    def resize_and_compress(self, img, max_width=1920, max_height=1920, 
                           quality=85, maintain_aspect=True):
        """
        Redimensiona e comprime imagem mantendo qualidade visual
        
        Args:
            img: Imagem OpenCV (numpy array)
            max_width: Largura máxima
            max_height: Altura máxima
            quality: Qualidade JPEG (70-95 recomendado)
            maintain_aspect: Manter aspect ratio
        
        Returns:
            Tupla (imagem_bytes, largura_final, altura_final)
        """
        h, w = img.shape[:2]
        
        # Calcular novo tamanho se necessário
        if w > max_width or h > max_height:
            if maintain_aspect:
                # Calcular escala mantendo aspect ratio
                scale = min(max_width / w, max_height / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
            else:
                new_w = min(w, max_width)
                new_h = min(h, max_height)
            
            # Redimensionar com LANCZOS para melhor qualidade
            img_resized = cv2.resize(
                img, 
                (new_w, new_h), 
                interpolation=cv2.INTER_LANCZOS4
            )
            logger.info(f"Imagem redimensionada de {w}x{h} para {new_w}x{new_h}")
        else:
            img_resized = img
            new_w, new_h = w, h
        
        # Comprimir com OpenCV usando qualidade otimizada
        encode_params = [
            cv2.IMWRITE_JPEG_QUALITY, quality,
            cv2.IMWRITE_JPEG_OPTIMIZE, 1,  # Otimização adicional
            cv2.IMWRITE_JPEG_PROGRESSIVE, 1  # Progressive JPEG (melhor para web)
        ]
        
        success, buffer = cv2.imencode('.jpg', img_resized, encode_params)
        
        if not success:
            logger.error("Falha ao comprimir imagem")
            # Fallback sem compressão especial
            success, buffer = cv2.imencode('.jpg', img_resized)
        
        compressed_bytes = buffer.tobytes()
        
        # Log do tamanho
        original_size = img.nbytes / 1024  # KB
        compressed_size = len(compressed_bytes) / 1024  # KB
        compression_ratio = (1 - compressed_size / original_size) * 100
        
        logger.info(
            f"Compressão: {original_size:.1f}KB → {compressed_size:.1f}KB "
            f"({compression_ratio:.1f}% redução)"
        )
        
        return compressed_bytes, new_w, new_h
    
    def detect_and_crop(self, image_bytes, conf_threshold=0.3, 
                       margin_percent=0.1, return_base64=True,
                       max_size=1920, jpeg_quality=85):
        """
        Detecta produtos e retorna o crop do produto principal.
        Se não detectar, retorna a imagem original comprimida.
        
        Args:
            image_bytes: Bytes da imagem
            conf_threshold: Threshold de confiança (0-1)
            margin_percent: Margem adicional no crop (0-1)
            return_base64: Retornar imagem em base64
            max_size: Tamanho máximo da imagem (largura/altura)
            jpeg_quality: Qualidade JPEG (70-95)
        """
        try:
            # Converter bytes para imagem
            image = Image.open(io.BytesIO(image_bytes))
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            original_h, original_w = img.shape[:2]
            
            # Rodar inferência (otimizada para ARM64)
            results = self.model.predict(
                img, 
                conf=conf_threshold, 
                device=self.device,
                verbose=False,
                imgsz=640
            )
            
            # Verificar se há detecções
            total_detections = 0
            if results and len(results) > 0 and results[0].boxes is not None:
                total_detections = len(results[0].boxes)
            
            # Se não detectou nada, retornar imagem original comprimida
            if total_detections == 0:
                logger.warning("Nenhum produto detectado - retornando imagem original comprimida")
                
                # Comprimir e redimensionar imagem original
                compressed_bytes, final_w, final_h = self.resize_and_compress(
                    img,
                    max_width=max_size,
                    max_height=max_size,
                    quality=jpeg_quality,
                    maintain_aspect=True
                )
                
                response = {
                    "success": False,
                    "message": "Produto não encontrado na imagem",
                    "detections": 0,
                    "image_processed": True,
                    "original_size": {"width": original_w, "height": original_h},
                    "final_size": {"width": final_w, "height": final_h},
                    "file_size_kb": round(len(compressed_bytes) / 1024, 2),
                    "compression_quality": jpeg_quality
                }
                
                if return_base64:
                    import base64
                    response["image"] = base64.b64encode(compressed_bytes).decode('utf-8')
                else:
                    response["image_bytes"] = compressed_bytes
                
                return response
            
            # Selecionar produto principal
            best_box, score = self.select_main_product(results, img.shape)
            
            # Se não encontrou um produto válido após seleção
            if best_box is None:
                logger.warning("Produto detectado mas não passou nos critérios - retornando imagem original")
                
                # Comprimir e redimensionar imagem original
                compressed_bytes, final_w, final_h = self.resize_and_compress(
                    img,
                    max_width=max_size,
                    max_height=max_size,
                    quality=jpeg_quality,
                    maintain_aspect=True
                )
                
                response = {
                    "success": False,
                    "message": "Produto não encontrado na imagem",
                    "detections": total_detections,
                    "image_processed": True,
                    "original_size": {"width": original_w, "height": original_h},
                    "final_size": {"width": final_w, "height": final_h},
                    "file_size_kb": round(len(compressed_bytes) / 1024, 2),
                    "compression_quality": jpeg_quality
                }
                
                if return_base64:
                    import base64
                    response["image"] = base64.b64encode(compressed_bytes).decode('utf-8')
                else:
                    response["image_bytes"] = compressed_bytes
                
                return response
            
            # PRODUTO ENCONTRADO - Processar crop
            # Extrair coordenadas
            x1, y1, x2, y2 = map(int, best_box.xyxy[0].cpu().numpy())
            confidence = float(best_box.conf[0].cpu().numpy())
            class_id = int(best_box.cls[0].cpu().numpy())
            class_name = self.model.names[class_id]
            
            # Adicionar margem
            h, w = img.shape[:2]
            margin_x = int((x2 - x1) * margin_percent)
            margin_y = int((y2 - y1) * margin_percent)
            
            x1 = max(0, x1 - margin_x)
            y1 = max(0, y1 - margin_y)
            x2 = min(w, x2 + margin_x)
            y2 = min(h, y2 + margin_y)
            
            # Crop da imagem
            cropped = img[y1:y2, x1:x2]
            
            # Redimensionar e comprimir
            cropped_bytes, final_w, final_h = self.resize_and_compress(
                cropped,
                max_width=max_size,
                max_height=max_size,
                quality=jpeg_quality,
                maintain_aspect=True
            )
            
            # Resposta com produto encontrado
            response = {
                "success": True,
                "message": "Produto encontrado e processado com sucesso",
                "detections": total_detections,
                "image_processed": True,
                "main_product": {
                    "class": class_name,
                    "confidence": round(confidence, 4),
                    "selection_score": round(score, 4),
                    "bbox": {
                        "x1": x1, "y1": y1,
                        "x2": x2, "y2": y2
                    },
                    "cropped_size": {
                        "width": x2 - x1, 
                        "height": y2 - y1
                    }
                },
                "original_size": {"width": w, "height": h},
                "final_size": {"width": final_w, "height": final_h},
                "file_size_kb": round(len(cropped_bytes) / 1024, 2),
                "compression_quality": jpeg_quality
            }
            
            if return_base64:
                import base64
                response["image"] = base64.b64encode(cropped_bytes).decode('utf-8')
            else:
                response["image_bytes"] = cropped_bytes
            
            return response
            
        except Exception as e:
            logger.error(f"Erro ao processar imagem: {str(e)}", exc_info=True)
            return {
                "success": False,
                "message": f"Erro ao processar imagem: {str(e)}",
                "detections": 0,
                "image_processed": False,
                "error_type": type(e).__name__
            }
