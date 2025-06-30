from pathlib import Path
import cv2
import easyocr
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import torch
import os

class OCRExtractor:
    def __init__(self, config):
        self.config = config
        self.reader = easyocr.Reader(config.languages, gpu=True)
        
    def resize_before_ocr(self, img, max_size=640):
        h, w = img.shape[:2]
        scale = max_size / max(h, w)
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return img
    
    def extract_text_regions(self, img_path):
        """Extract text regions using EasyOCR"""
        img = cv2.imread(str(img_path))
        if img is None:
            return []

        img_stem = img_path.stem.replace(" ", "_")
        img = self.resize_before_ocr(img, max_size=1024)
        results = self.reader.readtext(img)
        
        extracted_regions = []
        for i, (bbox, text, conf) in enumerate(results):
            if conf < self.config.confidence_threshold or len(text.strip()) < 1:
                continue

            x_min = int(min(p[0] for p in bbox))
            y_min = int(min(p[1] for p in bbox))
            x_max = int(max(p[0] for p in bbox))
            y_max = int(max(p[1] for p in bbox))
            
            if x_max <= x_min or y_max <= y_min:
                continue

            crop = img[y_min:y_max, x_min:x_max]
            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                continue
                
            extracted_regions.append({
                'crop': crop,
                'text': text.strip(),
                'confidence': conf,
                'filename': f"{img_stem}_{i}.jpg"
            })
        
        return extracted_regions

    def process_all_images(self, input_dir, max_workers=None):
        """Process all images in directory"""
        image_files = list(Path(input_dir).rglob("*.[jp][pn]g"))
        
        if max_workers is None:
            max_workers = 1 if torch.cuda.is_available() else max(2, os.cpu_count() // 2)
        
        all_regions = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for result in tqdm(executor.map(self.extract_text_regions, image_files), 
                             total=len(image_files)):
                if result:
                    all_regions.extend(result)
        
        return all_regions