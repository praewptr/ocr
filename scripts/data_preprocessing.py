import cv2
import numpy as np
from pathlib import Path

class ImagePreprocessor:
    def __init__(self, config):
        self.config = config
    
    def resize_crop_for_trocr(self, crop, size=384):
        """Resize and pad crop to fixed size for TrOCR"""
        h, w = crop.shape[:2]
        scale = size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        pad_top = (size - new_h) // 2
        pad_bottom = size - new_h - pad_top
        pad_left = (size - new_w) // 2
        pad_right = size - new_w - pad_left
        
        padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                   borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
        return padded
    
    def is_valid_crop(self, crop, min_std=5):
        """Check if crop has enough variation (not blank/uniform)"""
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return False
        
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return np.std(gray) >= min_std
    
    def enhance_image(self, crop):
        # Convert to grayscale
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Convert back to BGR
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        return enhanced_bgr
    
    def process_regions(self, regions, output_dir, enhance=False):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        processed_data = []
        
        for region in regions:
            crop = region['crop']
            

            if not self.is_valid_crop(crop):
                continue
 
            if enhance:
                crop = self.enhance_image(crop)
            
        
            processed_crop = self.resize_crop_for_trocr(crop, size=self.config.trocr_img_size)
            
         
            output_path = output_dir / region['filename']
            cv2.imwrite(str(output_path), processed_crop)
            
            processed_data.append({
                'filename': region['filename'],
                'text': region['text'],
                'confidence': region['confidence']
            })
        
        return processed_data