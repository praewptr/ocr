import sys
from pathlib import Path
from core.config import Config
from models.trainer import OCRTrainer

def main():
    # Load configuration
    config = Config()
    
    # Create trainer
    trainer = OCRTrainer(config)
    
    # Create datasets
    train_dataset, val_dataset = trainer.create_datasets(
        image_dir=config.REC_IMG_DIR,
        gt_path=config.REC_GT_PATH,
        val_ratio=config.VAL_RATIO,
        max_samples=config.MAX_SAMPLES
    )
    
    # Create training arguments
    training_args = trainer.create_training_args(
        output_dir=config.MODEL_OUTPUT_DIR,
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        use_fp16=config.USE_FP16
    )
    
    # Train the model
    model_trainer, results = trainer.train(train_dataset, val_dataset, training_args)
    
    # Save the model
    trainer.save_model(config.MODEL_OUTPUT_DIR)
    
    print("Training completed successfully!")
    print(f"Final WER: {results['eval_wer']:.4f}")
    print(f"Final CER: {results['eval_cer']:.4f}")

if __name__ == "__main__":
    main()