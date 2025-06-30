import random
from pathlib import Path
import torch
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from .dataset import OCRDataset
from .metrics import create_compute_metrics_function

class OCRTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model and processor
        self.processor = TrOCRProcessor.from_pretrained(config.MODEL_NAME, use_fast=True)
        self.model = VisionEncoderDecoderModel.from_pretrained(config.MODEL_NAME)
        
        # Configure model tokens
        self.model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.model.to(self.device)
        
    def load_gt_and_split(self, gt_path, val_ratio=0.1, seed=42, max_samples=None):
        """Load ground truth and split into train/validation sets"""
        with open(gt_path, encoding="utf-8") as f:
            lines = [line.strip().split("\t") for line in f if "\t" in line]

        random.seed(seed)
        random.shuffle(lines)

        if max_samples:
            lines = lines[:min(len(lines), max_samples)]

        split_idx = int(len(lines) * val_ratio)
        return lines[split_idx:], lines[:split_idx]  # train, val
    
    def create_datasets(self, image_dir, gt_path, val_ratio=0.1, max_samples=None):
        """Create training and validation datasets"""
        train_samples, val_samples = self.load_gt_and_split(
            gt_path, val_ratio, max_samples=max_samples
        )
        
        train_dataset = OCRDataset(train_samples, image_dir, self.processor)
        val_dataset = OCRDataset(val_samples, image_dir, self.processor)
        
        print(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")
        
        return train_dataset, val_dataset
    
    def create_training_args(self, output_dir, batch_size=4, epochs=2, use_fp16=True):
        """Create training arguments"""
        return Seq2SeqTrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=5e-5,
            num_train_epochs=epochs,
            logging_steps=20,
            eval_strategy="epoch",
            save_strategy="no",
            predict_with_generate=True,
            fp16=use_fp16,
            logging_dir="./logs",
            report_to="tensorboard",
        )
    
    def train(self, train_dataset, val_dataset, training_args):
        """Train the model"""
        # Create compute_metrics function with processor
        compute_metrics_fn = create_compute_metrics_function(self.processor)
        
        # Create trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.processor,
            compute_metrics=compute_metrics_fn,
        )
        
        # Train the model
        print("Starting training...")
        trainer.train()
        
        # Evaluate
        results = trainer.evaluate()
        print("Evaluation results:", results)
        print("WER:", results["eval_wer"])
        print("CER:", results["eval_cer"])
        
        return trainer, results
    
    def save_model(self, output_dir):
        """Save the trained model and processor"""
        self.model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)
        print(f"Model saved to: {output_dir}")