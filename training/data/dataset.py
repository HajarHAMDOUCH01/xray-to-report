import os
import json
from pathlib import Path
from typing import Optional, Callable, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class XRayReportDataset(Dataset):
    
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
        seed: int = 42,
        skip_corrupted: bool = True
    ):
        self.root_dir = Path(data_root if data_root else root_dir)
        self.split = split
        self.report_key = report_key
        self.max_samples = max_samples
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed
        self.skip_corrupted = skip_corrupted
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        
        all_samples = self._build_index()
        
        split_dir = self.root_dir / self.split
        if not split_dir.exists():
            all_samples = self._perform_split(all_samples)
        
        self.samples = all_samples
        
        if self.max_samples is not None:
            self.samples = self.samples[:self.max_samples]
    
    def _build_index(self):
        """Build index of all image-report pairs for the specified split."""
        samples = []
        
        split_dir = self.root_dir / self.split
        if split_dir.exists():
            search_dir = split_dir
        else:
            search_dir = self.root_dir
        
        for folder in sorted(search_dir.iterdir()):
            if not folder.is_dir():
                continue
            
            png_files = list(folder.glob("*.png"))
            json_files = list(folder.glob("*.json"))
            
            for png_file in png_files:
                base_name = png_file.stem  # e.g., "00000"
                json_file = folder / f"{base_name}.json"
                
                if json_file.exists():
                    if self.skip_corrupted:
                        try:
                            with Image.open(png_file) as img:
                                img.verify()  
                            
                            samples.append({
                                'image_path': str(png_file),
                                'json_path': str(json_file)
                            })
                            
                        except (IOError, OSError, Image.DecompressionBombError) as e:
                            print(f"Warning: Skipping corrupted image {png_file}: {e}")
                            continue  
                    else:
                        samples.append({
                            'image_path': str(png_file),
                            'json_path': str(json_file)
                        })
        
        print(f"Found {len(samples)} valid samples for {self.split} split")
        return samples
    
    def _perform_split(self, samples):
        """Split samples into train/val/test sets."""
        import random
        
        random.seed(self.seed)
        
        shuffled = samples.copy()
        random.shuffle(shuffled)
        
        total = len(shuffled)
        train_size = int(total * self.train_ratio)
        val_size = int(total * self.val_ratio)
        
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
        
        try:
            with Image.open(sample['image_path']) as img:
                image = img.convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            with open(sample['json_path'], 'r') as f:
                data = json.load(f)
                report = data.get(self.report_key, "")
                
                if not report:
                    for key in ['caption', 'report', 'findings', 'impression', 'text']:
                        if key in data and data[key]:
                            report = data[key]
                            break
                
                if not report:
                    print(f"Warning: No text found in {sample['json_path']}")
                    report = ""
            
            return {
                'image': image,
                'report': report,
                'image_path': sample['image_path']
            }
            
        except (IOError, OSError, Image.DecompressionBombError, Image.UnidentifiedImageError) as e:
            if self.skip_corrupted:
                print(f"Warning: Skipping corrupted sample {sample['image_path']}: {e}")

                placeholder_image = torch.zeros(3, 224, 224) if self.transform else None
                return {
                    'image': placeholder_image,
                    'report': "",
                    'image_path': sample['image_path'],
                    'corrupted': True  
                }
            else:
                raise
    
    def get_valid_samples_count(self):
        """Get the number of valid (non-corrupted) samples."""
        if not hasattr(self, '_valid_count'):
            self._valid_count = sum(1 for i in range(len(self.samples)) 
                                  if not self._is_corrupted(i))
        return self._valid_count
    
    def _is_corrupted(self, idx):
        """Check if a sample is corrupted (internal method)."""

        return False