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
