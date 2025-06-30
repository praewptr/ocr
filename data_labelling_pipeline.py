import sys
import os
from pathlib import Path
import time
import logging
from scripts.data_extraction import OCRExtractor
from scripts.data_preprocessing import ImagePreprocessor
from scripts.data_validation import DataValidator
from core.config import Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OCRPipeline:
    def __init__(self, config):
        self.config = config
        self.extractor = OCRExtractor(config)
        self.preprocessor = ImagePreprocessor(config)
        self.validator = DataValidator(config)

    
    def run_step_1_extraction(self):
        logger.info("[STEP 1] Starting text extraction...")
        
        try:
            regions = self.extractor.process_all_images(
                input_dir=self.config.INPUT_ROOT,
                max_workers=self.config.MAX_WORKERS
            )
        
            
            return regions
            
        except Exception as e:
            logger.error(f"[STEP 1] Failed: {str(e)}")
            raise
    
    def run_step_2_preprocessing(self, regions):
        logger.info("[STEP 2] Starting image preprocessing...")

        try:
            processed_data = self.preprocessor.process_regions(
                regions=regions,
                output_dir=self.config.REC_IMG_DIR,
                enhance=self.config.ENHANCE_IMAGES
            )
            
            # Create ground truth file
            self.config.REC_GT_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config.REC_GT_PATH, "w", encoding="utf-8") as f:
                for item in processed_data:
                    f.write(f"{item['filename']}\t{item['text']}\n")
    
            logger.info(f"[STEP 2] Ground truth saved to: {self.config.REC_GT_PATH}")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"[STEP 2] Failed: {str(e)}")
            raise
    
    def run_step_3_validation(self):
        logger.info("[STEP 3] Starting data validation...")

        try:
            image_issues = self.validator.validate_images(self.config.REC_IMG_DIR)
            gt_issues = self.validator.validate_ground_truth(self.config.REC_GT_PATH)
            stats = self.validator.generate_statistics(self.config.REC_GT_PATH)
            
            if image_issues:
                logger.warning(f"[STEP 3] Found {len(image_issues)} image issues:")
                for issue in image_issues[:5]:
                    logger.warning(f"  - {issue}")
                if len(image_issues) > 5:
                    logger.warning(f"  ... and {len(image_issues) - 5} more issues")
            else:
                logger.info("[STEP 3] All images passed validation")
            
            if gt_issues:
                logger.warning(f"[STEP 3] Found {len(gt_issues)} ground truth issues:")
                for issue in gt_issues[:5]:
                    logger.warning(f"  - {issue}")
            else:
                logger.info("[STEP 3] Ground truth file passed validation")
            
            
            return {"image_issues": image_issues, "gt_issues": gt_issues, "stats": stats}
            
        except Exception as e:
            logger.error(f"[STEP 3] Failed: {str(e)}")
            raise
 
    def run_full_pipeline(self):
        """Run the complete OCR data preparation pipeline (3 steps only)"""
    
        
        logger.info("Starting OCR Data Pipeline...")
        logger.info(f"Input directory: {self.config.INPUT_ROOT}")
        logger.info(f"Output directory: {self.config.REC_IMG_DIR}")
  
        
        try:

            regions = self.run_step_1_extraction()
            self.run_step_2_preprocessing(regions)
            self.run_step_3_validation()
           
           
        except Exception as e:
            logger.error("Pipeline failed!")
            logger.error(f"Error: {str(e)}")
            raise

def main():

    config = Config()
    

    pipeline = OCRPipeline(config)
    try:
        # Run full pipeline
        pipeline.run_full_pipeline()
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()