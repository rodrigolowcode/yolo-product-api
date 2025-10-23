import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io

class ProductDetector:
    def __init__(self, model_path='yolo11n.pt', device='cpu'):
        """Inicializa o detector YOLO11 otimizado para CPU"""
        self.model = YOLO(model_path)
        self.device = device
        
        # Exportar para OpenVINO na primeira vez
        try:
            self.model.export(format='openvino', int8=True)
            # Recarregar modelo OpenVINO
            self.model = YOLO(model_path.replace('.pt', '_openvino_model'))
        except:
            pass  # Se já foi exportado, continua
    
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
        best_result = None
        
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
                    best_result = result
        
        return best_box, best_result, best_score
    
    def detect_and_crop(self, image_bytes, conf_threshold=0.3, 
                       margin_percent=0.1, return_base64=True):
        """
        Detecta produtos e retorna o crop do produto principal
        """
        # Converter bytes para imagem
        image = Image.open(io.BytesIO(image_bytes))
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Rodar inferência
        results = self.model.predict(
            img, 
            conf=conf_threshold, 
            device=self.device,
            verbose=False
        )
        
        # Selecionar produto principal
        best_box, best_result, score = self.select_main_product(
            results, 
            img.shape
        )
        
        if best_box is None:
            return {
                "success": False,
                "message": "Nenhum produto detectado",
                "detections": 0
            }
        
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
        
        # Converter para bytes
        _, buffer = cv2.imencode('.jpg', cropped)
        cropped_bytes = buffer.tobytes()
        
        # Resposta
        response = {
            "success": True,
            "detections": len(results[0].boxes) if results[0].boxes else 0,
            "main_product": {
                "class": class_name,
                "confidence": confidence,
                "selection_score": score,
                "bbox": {
                    "x1": x1, "y1": y1,
                    "x2": x2, "y2": y2
                },
                "original_size": {"width": w, "height": h},
                "cropped_size": {
                    "width": x2 - x1, 
                    "height": y2 - y1
                }
            }
        }
        
        if return_base64:
            import base64
            response["cropped_image"] = base64.b64encode(cropped_bytes).decode('utf-8')
        else:
            response["cropped_bytes"] = cropped_bytes
        
        return response
