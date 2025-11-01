import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import json
import numpy as np

class XRayReportDataset(Dataset):
    def __init__(self, data_root, split='train', transform=None, max_samples=None):
        """
        Args:
            data_root: path to mimicDatatotal folder
            split: 'train', 'val', or 'test'
            transform: image transforms
            max_samples: for debugging, limit dataset size
        """
        self.data_root = data_root
        self.transform = transform
        self.split = split
        
        self.sample_folders = [f for f in os.listdir(data_root) 
                             if os.path.isdir(os.path.join(data_root, f))]
        self.sample_folders.sort()
        
        if max_samples:
            self.sample_folders = self.sample_folders[:max_samples]
        
        total_samples = len(self.sample_folders)
        train_end = int(0.8 * total_samples)
        val_end = int(0.9 * total_samples)
        
        if split == 'train':
            self.sample_folders = self.sample_folders[:train_end]
        elif split == 'val':
            self.sample_folders = self.sample_folders[train_end:val_end]
        elif split == 'test':
            self.sample_folders = self.sample_folders[val_end:]
        
        print(f"{split} dataset: {len(self.sample_folders)} samples")
    
    def __len__(self):
        return len(self.sample_folders)
    
    def __getitem__(self, idx):
        folder_name = self.sample_folders[idx]
        folder_path = os.path.join(self.data_root, folder_name)
        
        json_path = os.path.join(folder_path, f"{folder_name}.json")
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        report_text = ""
        for field in ['findings', 'impression', 'comparison']:
            if field in data and data[field]:
                report_text += data[field] + " "
        
        report_text = report_text.strip()
        if not report_text:  
            report_text = "No findings reported."
        
        image_path = os.path.join(folder_path, f"{folder_name}.png")
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(np.array(image)).permute(2, 0, 1).float()
            image = (image / 127.5) - 1.0  # [0,255] -> [-1,1]
        
        return {
            'image': image,
            'report': report_text,
            'sample_id': folder_name
        }