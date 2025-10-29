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
        self.uploaded_batches = set()
        self.staging_dir = Path("./kaggle_staging")
        self.staging_dir.mkdir(exist_ok=True)
        self.chunk_counter = 0

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

    def upload_chunk(self, upload_checkpoint, chunk_name=None):
        """Upload a chunk of batches up to the checkpoint"""
        batch_files = list(self.staging_dir.glob("batch_*.pt"))
        if not batch_files:
            print("No batches found in staging directory")
            return False
            
        # Sort batches numerically
        batch_files.sort(key=lambda x: int(x.stem.split('_')[1]))
        
        # Get batches up to checkpoint
        batches_to_upload = [bf for bf in batch_files if int(bf.stem.split('_')[1]) <= upload_checkpoint]
        
        if not batches_to_upload:
            print(f"No batches to upload for checkpoint {upload_checkpoint}")
            return False
            
        temp_dir = None
        try:
            temp_dir = tempfile.mkdtemp()
            temp_path = Path(temp_dir)
            
            # Copy selected batches to temp directory
            for batch_file in batches_to_upload:
                shutil.copy2(batch_file, temp_path / batch_file.name)
            
            self.chunk_counter += 1
            if chunk_name is None:
                chunk_name = f"chunk_{self.chunk_counter}"
            
            print(f" Uploading {len(batches_to_upload)} batches as {chunk_name}")
            print(f" Batch range: {batches_to_upload[0].name} to {batches_to_upload[-1].name}")
            
            self._create_dataset_metadata(
                temp_dir,
                title=f"MIMIC Features - {chunk_name}",
                description=f"Feature embeddings for MIMIC dataset - {chunk_name} ({len(batches_to_upload)} batches)"
            )
            
            if self.dataset_exists:
                # KEY FIX: Don't clear staging after successful upload
                # We need to keep track of what we've uploaded
                result = self.api.dataset_create_version(
                    folder=temp_dir,
                    version_notes=f"{chunk_name} - batches up to {upload_checkpoint}",
                    quiet=False,
                    # This is important - it tells Kaggle to merge files rather than replace
                    delete_old_versions=False
                )
                print(f"✓ Created new version with {len(batches_to_upload)} additional batches")
            else:
                result = self.api.dataset_create_new(
                    folder=temp_dir,
                    public=False,
                    quiet=False
                )
                self.dataset_exists = True
                print(f"✓ Created new dataset with {len(batches_to_upload)} batches")
            
            # Mark these batches as uploaded so we don't re-upload them
            for batch_file in batches_to_upload:
                batch_idx = int(batch_file.stem.split('_')[1])
                self.uploaded_batches.add(batch_idx)
                
            print(f"✓ {chunk_name} uploaded successfully ({len(batches_to_upload)} batches)")
            return True
            
        except Exception as e:
            print(f"✗ Failed to upload {chunk_name}: {e}")
            return False
        finally:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def clear_staging(self):
        """Clear only the batches that have been successfully uploaded"""
        if self.staging_dir.exists():
            files_cleared = 0
            for file in self.staging_dir.glob("batch_*.pt"):
                batch_idx = int(file.stem.split('_')[1])
                if batch_idx in self.uploaded_batches:
                    file.unlink()
                    files_cleared += 1
            print(f"  Cleared {files_cleared} uploaded files from staging")

    def upload_all_batches_final(self):
        """Upload all remaining batches at the end"""
        return self.upload_chunk(upload_checkpoint=9999, chunk_name="final_chunk")

    def cleanup(self):
        if self.staging_dir.exists():
            shutil.rmtree(self.staging_dir)
            print("Cleaned up staging directory")