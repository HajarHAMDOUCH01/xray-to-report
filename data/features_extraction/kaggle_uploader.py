import os
import json
import torch
import tempfile
from pathlib import Path

class KaggleUploader:
    def __init__(self, dataset_slug, username):
        self.dataset_slug = dataset_slug
        self.username = username
        # self.staging_dir = Path(staging_dir)
        # self.staging_dir.mkdir(exist_ok=True)  
        self.setup_kaggle_api()
        self.dataset_exists = self.check_dataset_exists()

    def setup_kaggle_api(self):
      """Setup Kaggle API authentication"""
      try:
          from kaggle.api.kaggle_api_extended import KaggleApi
          self.api = KaggleApi()
          self.api.authenticate()
          print("Kaggle API authenticated")
      except Exception as e:
          print(f"Failed to authenticate Kaggle API: {e}")
          raise
    
    def check_dataset_exists(self):
        """Check if dataset exists"""
        try:
            self.api.dataset_metadata(self.dataset_slug)
            print(f" Dataset {self.dataset_slug} exists")
            return True
        except:
            print(f" Dataset {self.dataset_slug} will be created")
            return False
    
    def process_and_upload_batch(self, batch_idx, batch_data, version_notes=""):
        """Process batch and upload immediately"""
        # 1. Save batch to temp file
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            torch.save(batch_data, tmp.name)
        
        # 2. Upload this single batch
        try:
            if self.dataset_exists:
                self.api.dataset_create_version(
                    folder=os.path.dirname(tmp.name),
                    version_notes=f"{version_notes} - Batch {batch_idx}",
                    quiet=False
                )
            else:
                self.api.dataset_create_new(
                    folder=os.path.dirname(tmp.name),
                    public=False,
                    quiet=False
                )
                self.dataset_exists = True
            
            print(f" Batch {batch_idx} uploaded")
            return True
        finally:
            # 3. Clean up temp file
            os.unlink(tmp.name)

    def _create_dataset_metadata(self, folder, version_notes):
        """Create dataset-metadata.json"""
        metadata = {
            "title": "MIMIC Features V1",
            "id": self.dataset_slug,
            "licenses": [{"name": "CC0-1.0"}]
        }
        
        with open(Path(folder) / "dataset-metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)