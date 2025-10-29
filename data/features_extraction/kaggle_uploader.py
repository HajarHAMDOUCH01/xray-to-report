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
        self.uploaded_batches = []  
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
        print(f" Created dataset-metadata.json at {metadata_path}")

    def _prepare_upload_directory(self, batch_idx, batch_data):
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        
        current_batch_file = f"batch_{batch_idx:04d}.pt"
        current_batch_path = temp_path / current_batch_file
        torch.save(batch_data, current_batch_path)
        print(f" Saved batch data to {current_batch_path}")
        
        staging_batch_path = self.staging_dir / current_batch_file
        torch.save(batch_data, staging_batch_path)
        
        self.uploaded_batches.append(current_batch_file)
        
        for batch_file in self.uploaded_batches:
            if batch_file != current_batch_file:  
                source_path = self.staging_dir / batch_file
                if source_path.exists():
                    shutil.copy2(source_path, temp_path / batch_file)
                    print(f" Copied previous batch: {batch_file}")
        
        print(f" Upload directory contains {len(self.uploaded_batches)} batch files")
        return temp_dir, temp_path

    def process_and_upload_batch(self, batch_idx, batch_data, version_notes=""):
        temp_dir = None
        try:
            temp_dir, temp_path = self._prepare_upload_directory(batch_idx, batch_data)
            
            self._create_dataset_metadata(
                temp_dir, 
                title=f"MIMIC Features - Up to Batch {batch_idx}",
                description=f"Feature embeddings for MIMIC dataset - Batches 0 to {batch_idx}"
            )
            
            # Upload
            if self.dataset_exists:
                print(f" Creating new version including batches 0 to {batch_idx}...")
                result = self.api.dataset_create_version(
                    folder=temp_dir,
                    version_notes=f"{version_notes} - Includes batches 0 to {batch_idx}",
                    quiet=False
                )
            else:
                print(f" Creating new dataset with batch {batch_idx}...")
                result = self.api.dataset_create_new(
                    folder=temp_dir,
                    public=False,
                    quiet=False
                )
                self.dataset_exists = True
            
            print(f"✓ Batch {batch_idx} uploaded successfully (total: {len(self.uploaded_batches)} batches)")
            return True
            
        except Exception as e:
            print(f"✗ Failed to upload batch {batch_idx}: {e}")
            if batch_idx < len(self.uploaded_batches):
                self.uploaded_batches.pop()
            return False
        finally:
            # Clean up temporary directory
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def cleanup(self):
        if self.staging_dir.exists():
            shutil.rmtree(self.staging_dir)
            print("Cleaned up staging directory")