import random
from pathlib import Path

class DataSplitter:
    def __init__(self, config):
        self.config = config
    
    def load_gt_and_split(self, gt_path, val_ratio=0.1, seed=42, max_samples=None):
        """Load ground truth and split into train/val"""
        with open(gt_path, encoding="utf-8") as f:
            lines = [line.strip().split("\t") for line in f if "\t" in line]

        random.seed(seed)
        random.shuffle(lines)

        if max_samples:
            lines = lines[:min(len(lines), max_samples)]

        split_idx = int(len(lines) * val_ratio)
        train_samples = lines[split_idx:]
        val_samples = lines[:split_idx]
        
        return train_samples, val_samples
    
    def save_splits(self, train_samples, val_samples, output_dir):
        """Save train/val splits to separate files"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save train split
        with open(output_dir / "train_gt.txt", "w", encoding="utf-8") as f:
            for filename, text in train_samples:
                f.write(f"{filename}\t{text}\n")
        
        # Save validation split
        with open(output_dir / "val_gt.txt", "w", encoding="utf-8") as f:
            for filename, text in val_samples:
                f.write(f"{filename}\t{text}\n")
        
        print(f"✅ Saved {len(train_samples)} training samples")
        print(f"✅ Saved {len(val_samples)} validation samples")