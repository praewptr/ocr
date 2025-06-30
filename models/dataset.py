from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class OCRDataset(Dataset):
    def __init__(self, samples, image_dir, processor):
        self.samples = samples
        self.image_dir = Path(image_dir)
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, text = self.samples[idx]
        image_path = self.image_dir / filename
        image = Image.open(image_path).convert("RGB")
        image = image.resize((128, 128))

        encoding = self.processor(
            images=image,
            text=text,
            padding="max_length",
            return_tensors="pt",
            max_length=32,
            truncation=True,
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}