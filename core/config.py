from pathlib import Path
import torch
import os



class Config:
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    INPUT_ROOT = PROJECT_ROOT / "data" / "raw" / "Tobacco3482-jpg"
    DATA_DIR = PROJECT_ROOT / "data"
    REC_IMG_DIR = PROJECT_ROOT / "data" / "processed" / "ocr_dataset" / "images"
    REC_GT_PATH = PROJECT_ROOT / "data" / "processed" / "ocr_dataset" / "gt.txt"
    MODEL_OUTPUT_DIR = PROJECT_ROOT / "model" / "trocr-finetuned"
    LOGS_DIR = PROJECT_ROOT / "logs"
    
    # OCR Settings
    languages = ['en']
    confidence_threshold = 0.6
    trocr_img_size = 384
    
    # Processing Settings
    ENHANCE_IMAGES = False
    MAX_SAMPLES = 2000
    VAL_RATIO = 0.1
    RANDOM_SEED = 42
    
    # Performance Settings
    MAX_WORKERS = 1 if torch.cuda.is_available() else max(2, os.cpu_count() // 2)
    
    # Model Settings
    MODEL_NAME = "microsoft/trocr-base-printed"
    BATCH_SIZE = 4
    EPOCHS = 3
    USE_FP16 = True
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")