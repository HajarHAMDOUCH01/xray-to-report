from PIL import Image
import json
import os
from pathlib import Path
from torch.utils.data import Dataset

class MIMICDataset(Dataset):
    def __init__(self, root_dir, start_idx, end_idx, transform=None, validate_images=True):
        """
        Args:
            root_dir: Path to directory containing folders 00000, 00001, etc.
            start_idx: Starting folder index (e.g., 0)
            end_idx: Ending folder index (e.g., 999)
            transform: torchvision transforms for images
            validate_images: If True, verify images can be opened during init
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []  # Valid samples only
        self.corrupted_samples = []  # Track problematic samples
        
        print(f"Scanning folders {start_idx:05d} to {end_idx:05d}...")
        
        # Build list of folder paths with zero-padding
        for idx in range(start_idx, end_idx + 1):
            folder_id = f"{idx:05d}"
            folder_path = self.root_dir / folder_id
            
            # Check if folder exists
            if not folder_path.exists():
                print(f"  Warning: Folder {folder_id} does not exist, skipping...")
                self.corrupted_samples.append({
                    'folder_id': folder_id,
                    'reason': 'folder_not_found'
                })
                continue
            
            # Find PNG and JSON files
            png_files = list(folder_path.glob("*.png"))
            json_files = list(folder_path.glob("*.json"))
            
            # Verify exactly one of each
            if len(png_files) != 1 or len(json_files) != 1:
                print(f"  Warning: Folder {folder_id} has {len(png_files)} PNGs and {len(json_files)} JSONs, skipping...")
                self.corrupted_samples.append({
                    'folder_id': folder_id,
                    'reason': 'missing_files',
                    'png_count': len(png_files),
                    'json_count': len(json_files)
                })
                continue
            
            png_path = png_files[0]
            json_path = json_files[0]
            
            # Validate image can be opened (CRITICAL STEP)
            if validate_images:
                try:
                    with Image.open(png_path) as img:
                        img.verify()  # Check if it's a valid image
                    
                    # Re-open for actual loading (verify() closes the file)
                    with Image.open(png_path) as img:
                        img.load()  # Actually load pixel data
                        
                except Exception as e:
                    print(f" ERROR: Corrupted image in {folder_id}: {str(e)}")
                    self.corrupted_samples.append({
                        'folder_id': folder_id,
                        'reason': 'corrupted_image',
                        'error': str(e),
                        'png_path': str(png_path)
                    })
                    continue  # Skip this sample
            
            # Validate JSON can be parsed
            try:
                with open(json_path, 'r') as f:
                    report_data = json.load(f)
                    
                # Check if caption exists
                if 'caption' not in report_data or not report_data['caption'].strip():
                    print(f"  Warning: {folder_id} has empty caption, using placeholder")
                
            except Exception as e:
                print(f" ERROR: Corrupted JSON in {folder_id}: {str(e)}")
                self.corrupted_samples.append({
                    'folder_id': folder_id,
                    'reason': 'corrupted_json',
                    'error': str(e),
                    'json_path': str(json_path)
                })
                continue
            
            # If all validations pass, add to valid samples
            self.samples.append({
                'folder_id': folder_id,
                'png_path': png_path,
                'json_path': json_path,
                'original_index': idx  # Store original position
            })
        
        # Summary
        total_expected = end_idx - start_idx + 1
        valid_count = len(self.samples)
        corrupted_count = len(self.corrupted_samples)
        
        print(f"\n Dataset Summary:")
        print(f"   Expected samples: {total_expected}")
        print(f"    Valid samples: {valid_count}")
        print(f"    Corrupted/missing: {corrupted_count}")
        
        if self.corrupted_samples:
            print(f"\n  Corrupted samples details:")
            for item in self.corrupted_samples:
                print(f"      {item['folder_id']}: {item['reason']}")
    
    def get_corrupted_info(self):
        """Return list of corrupted samples for logging"""
        return self.corrupted_samples
    
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
        
        # Load image
        image = Image.open(sample['png_path']).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Load JSON and extract report text
        with open(sample['json_path'], 'r') as f:
            report_data = json.load(f)
        
        # Extract caption with fallback
        report_text = report_data.get('caption', '').strip()
        if not report_text:
            report_text = "No findings reported."
        
        return image, report_text, sample['folder_id']