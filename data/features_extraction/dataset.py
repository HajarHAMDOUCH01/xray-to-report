from torch.utils.data import Dataset
from PIL import Image
import json
import os
from pathlib import Path

class MIMICDataset(Dataset):
    def __init__(self, root_dir, start_idx, end_idx, transform=None):
        """
        Args:
            root_dir: Path to directory containing folders 00000, 00001, etc.
            start_idx: Starting folder index (e.g., 0)
            end_idx: Ending folder index (e.g., 999)
            transform: torchvision transforms for images
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []  # Will store (folder_id, png_path, json_path)
        
        for idx in range(start_idx, end_idx + 1):
            folder_id = f"{idx:05d}"  # Format as 00000, 00001, etc.
            folder_path = self.root_dir / folder_id
            
            if not folder_path.exists():
                print(f"Warning: Folder {folder_id} does not exist, skipping...")
                continue
            
            png_files = list(folder_path.glob("*.png"))
            json_files = list(folder_path.glob("*.json"))
            
            if len(png_files) != 1 or len(json_files) != 1:
                print(f"Warning: Folder {folder_id} has {len(png_files)} PNGs and {len(json_files)} JSONs, skipping...")
                continue
            
            self.samples.append({
                'folder_id': folder_id,
                'png_path': png_files[0],
                'json_path': json_files[0]
            })
        
        print(f"Loaded {len(self.samples)} valid samples from {start_idx:05d} to {end_idx:05d}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns:
            image_tensor: Tensor of shape (3, 224, 224) normalized to [-1, 1]
            report_text: String containing the medical report
            folder_id: String like "00000" for tracking
        """
        sample = self.samples[idx]
        
        image = Image.open(sample['png_path']).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        with open(sample['json_path'], 'r') as f:
            report_data = json.load(f)
        
        report_text = report_data.get('caption', '')
        
        if not report_text or report_text.strip() == '':
            report_text = "No findings reported."  # Placeholder
        
        return image, report_text, sample['folder_id']