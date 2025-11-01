import os
import json
from pathlib import Path
from typing import Optional, Callable, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class XRayReportDataset(Dataset):
    """
    Memory-efficient dataset for X-ray images and medical reports.
    
    Args:
        root_dir: Path to mimicDatatotal directory
        transform: Optional image transforms (if None, uses default)
        report_key: JSON key containing the report text (default: 'report')
    """
    
    def __init__(
        self, 
        root_dir: str = None,
        data_root: str = None,
        split: str = 'train',
        transform: Optional[Callable] = None,
        report_key: str = 'caption',
        max_samples: Optional[int] = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42
    ):
        # Support both root_dir and data_root parameters
        self.root_dir = Path(data_root if data_root else root_dir)
        self.split = split
        self.report_key = report_key
        self.max_samples = max_samples
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed
        
        # Default transform for X-rays
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        
        # Build file index (stores only paths, not data)
        all_samples = self._build_index()
        
        # Split the data if no split directories exist
        split_dir = self.root_dir / self.split
        if not split_dir.exists():
            # Perform train/val/test split
            all_samples = self._perform_split(all_samples)
        
        self.samples = all_samples
        
        # Apply max_samples limit if specified
        if self.max_samples is not None:
            self.samples = self.samples[:self.max_samples]
    
    def _build_index(self):
        """Build index of all image-report pairs for the specified split."""
        samples = []
        
        # Check if split-specific directory exists
        split_dir = self.root_dir / self.split
        if split_dir.exists():
            # Split is a subdirectory (e.g., data_root/train/, data_root/val/)
            search_dir = split_dir
        else:
            # No split subdirectory, use root_dir directly
            search_dir = self.root_dir
        
        # Iterate through folder structure
        for folder in sorted(search_dir.iterdir()):
            if not folder.is_dir():
                continue
            
            # Look for .png and .json files
            png_files = list(folder.glob("*.png"))
            json_files = list(folder.glob("*.json"))
            
            # Match pairs based on filename prefix
            for png_file in png_files:
                base_name = png_file.stem  # e.g., "00000"
                json_file = folder / f"{base_name}.json"
                
                if json_file.exists():
                    samples.append({
                        'image_path': str(png_file),
                        'json_path': str(json_file)
                    })
        
        return samples
    
    def _perform_split(self, samples):
        """Split samples into train/val/test sets."""
        import random
        
        # Set seed for reproducibility
        random.seed(self.seed)
        
        # Shuffle samples
        shuffled = samples.copy()
        random.shuffle(shuffled)
        
        total = len(shuffled)
        train_size = int(total * self.train_ratio)
        val_size = int(total * self.val_ratio)
        
        # Split indices
        if self.split == 'train':
            return shuffled[:train_size]
        elif self.split == 'val':
            return shuffled[train_size:train_size + val_size]
        elif self.split == 'test':
            return shuffled[train_size + val_size:]
        else:
            raise ValueError(f"Unknown split: {self.split}. Use 'train', 'val', or 'test'.")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            dict with keys:
                - 'image': Transformed image tensor
                - 'report': Medical report text
                - 'image_path': Path to image (for debugging)
        """
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Load report from JSON
        with open(sample['json_path'], 'r') as f:
            data = json.load(f)
            # Try multiple possible keys for the report text
            report = data.get(self.report_key, "")
            
            # Fallback to other common keys if empty
            if not report:
                for key in ['caption', 'report', 'findings', 'impression', 'text']:
                    if key in data and data[key]:
                        report = data[key]
                        break
            
            # If still empty, log warning
            if not report:
                print(f"Warning: No text found in {sample['json_path']}")
                report = ""
        
        return {
            'image': image,
            'report': report,
            'image_path': sample['image_path']
        }