import os
import json
import torch
import tempfile
import shutil
from pathlib import Path

class KaggleUploader:
    def __init__(self, dataset_slug, username):
        self.dataset_slug = dataset_slug
        self.username = username
        self.setup_kaggle_api()
        self.dataset_exists = self.check_dataset_exists()
        self.uploaded_batches = set()  # Track uploaded batch indices
        self.staging_dir = Path("./kaggle_staging")
        self.staging_dir.mkdir(exist_ok=True)

    def setup_kaggle_api(self):
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            self.api = KaggleApi()
            self.api.authenticate()
            print("Kaggle API authenticated")
        except Exception as e:
            print(f"Failed to authenticate Kaggle API: {e}")
            raise
    
    def check_dataset_exists(self):
        try:
            self.api.dataset_metadata(self.dataset_slug)
            print(f" Dataset {self.dataset_slug} exists")
            return True
        except:
            print(f" Dataset {self.dataset_slug} will be created")
            return False

    def _create_dataset_metadata(self, folder, title, description=""):
        metadata = {
            "title": title,
            "id": f"{self.username}/{self.dataset_slug}",
            "licenses": [{"name": "CC0-1.0"}],
            "description": description
        }
        
        metadata_path = Path(folder) / "dataset-metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def upload_all_batches_final(self):
        """Upload all batches together at the end"""
        batch_files = list(self.staging_dir.glob("batch_*.pt"))
        if not batch_files:
            print("No batches found in staging directory")
            return False
            
        temp_dir = None
        try:
            temp_dir = tempfile.mkdtemp()
            temp_path = Path(temp_dir)
            
            # Copy all staged batches to temp directory
            for batch_file in batch_files:
                shutil.copy2(batch_file, temp_path / batch_file.name)
            
            print(f" Consolidated {len(batch_files)} batches for final upload")
            print(f" Batch files: {[f.name for f in batch_files]}")
            
            self._create_dataset_metadata(
                temp_dir,
                title="MIMIC Features - Complete Dataset",
                description=f"Complete feature embeddings for MIMIC dataset - {len(batch_files)} batches"
            )
            
            print(" Creating final dataset version with all batches...")
            
            if self.dataset_exists:
                result = self.api.dataset_create_version(
                    folder=temp_dir,
                    version_notes=f"Complete dataset with {len(batch_files)} batches",
                    quiet=False
                )
            else:
                result = self.api.dataset_create_new(
                    folder=temp_dir,
                    public=False,
                    quiet=False
                )
                self.dataset_exists = True
            
            print("✓ All batches uploaded successfully in final consolidation")
            return True
            
        except Exception as e:
            print(f"✗ Failed to upload final consolidation: {e}")
            return False
        finally:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def cleanup(self):
        if self.staging_dir.exists():
            shutil.rmtree(self.staging_dir)
            print("Cleaned up staging directory")