import cv2
import numpy as np
from pathlib import Path
from collections import Counter


class DataValidator:
    def __init__(self, config):
        self.config = config
    
    def validate_images(self, image_dir):
        """Validate all images in directory"""
        image_dir = Path(image_dir)
        issues = []
        
        for img_path in image_dir.glob("*.jpg"):
            img = cv2.imread(str(img_path))
            if img is None:
                issues.append(f"Cannot read: {img_path.name}")
                continue
            
            # Check image dimensions
            h, w = img.shape[:2]
            if h != self.config.trocr_img_size or w != self.config.trocr_img_size:
                issues.append(f"Wrong size {w}x{h}: {img_path.name}")
            
            # Check if image is too dark/bright
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            if mean_brightness < 50 or mean_brightness > 200:
                issues.append(f"Brightness issue ({mean_brightness:.1f}): {img_path.name}")
        
        return issues
    
    def validate_ground_truth(self, gt_path):
        """Validate ground truth file"""
        issues = []
        
        with open(gt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines):
            line = line.strip()
            if '\t' not in line:
                issues.append(f"Line {i+1}: Missing tab separator")
                continue
            
            filename, text = line.split('\t', 1)
            
            if not filename.endswith('.jpg'):
                issues.append(f"Line {i+1}: Invalid filename format")
            
            if len(text.strip()) == 0:
                issues.append(f"Line {i+1}: Empty text")
        
        return issues
    
    def generate_statistics(self, gt_path):
        """Generate dataset statistics"""
        with open(gt_path, 'r', encoding='utf-8') as f:
            lines = [line.strip().split('\t') for line in f if '\t' in line]
        
        texts = [text for _, text in lines]
        text_lengths = [len(text) for text in texts]
        word_counts = [len(text.split()) for text in texts]
        
        stats = {
            'total_samples': len(lines),
            'avg_text_length': np.mean(text_lengths),
            'avg_word_count': np.mean(word_counts),
            'max_text_length': max(text_lengths),
            'min_text_length': min(text_lengths)
        }
        
        return stats