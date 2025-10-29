import config
import torch
import gc
import time
from models import VGG19, LLMEmbedder
from feature_extractor import FeatureExtractor
from kaggle_uploader import KaggleUploader
import os

import psutil

def print_memory_usage():
    """Print current RAM and GPU usage"""
    # RAM
    ram = psutil.virtual_memory()
    print(f"   RAM: {ram.used/1e9:.1f}/{ram.total/1e9:.1f} GB ({ram.percent}%)")
    
    # GPU
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / 1e9
        gpu_max = torch.cuda.max_memory_allocated() / 1e9
        print(f"   GPU: {gpu_mem:.1f} GB (peak: {gpu_max:.1f} GB)")

def main():
    print("="*50)
    print("MIMIC Feature Extraction Pipeline (RAM-Optimized)")
    print("="*50)
    
    # Initialize models
    print("\n1. Loading models...")
    vgg = VGG19().to(config.DEVICE).eval()
    llm = LLMEmbedder(config).to(config.DEVICE).eval()
    print(" Models loaded")
    
    extractor = FeatureExtractor(vgg, llm, config)
    uploader = KaggleUploader(
        dataset_slug=config.KAGGLE_DATASET_SLUG,
        username=config.KAGGLE_USERNAME # Persistent across batches!
    )
    
    # Calculate batches
    batch_ranges = []
    for batch_idx in range(config.NUM_BATCHES + 1):
        start = batch_idx * config.BATCH_SIZE
        end = min(start + config.BATCH_SIZE - 1, config.END_IDX)
        if start <= config.END_IDX:
            batch_ranges.append((start, end, batch_idx))
    
    print(f"\n2. Will process {len(batch_ranges)} batches")
    print(f"   Upload every: {config.UPLOAD_EVERY_N_BATCHES} batches\n")
    
    # Track statistics
    all_corrupted_samples = []
    successful_uploads = 0
    failed_uploads = 0
    total_start_time = time.time()
    
    # Process batches
    for start, end, batch_idx in batch_ranges:
        batch_start_time = time.time()
        
        print(f"\n{'='*50}")
        print(f"Batch {batch_idx + 1}/{len(batch_ranges)}")
        print(f"{'='*50}")
        
        try:
            # Extract features
            batch_data = extractor.extract_batch(start, end)
            
            # Track corrupted
            if 'corrupted_samples' in batch_data:
                all_corrupted_samples.extend(batch_data['corrupted_samples'])
            
            extraction_time = time.time() - batch_start_time
            print(f"  Extraction time: {extraction_time:.1f}s")
            

            
            upload_start = time.time()
            
            version_notes = f"Batches {batch_idx} through {batch_idx}"
            success = uploader.process_and_upload_batch(
                batch_idx, 
                batch_data, 
                version_notes=f"Batch {batch_idx}"
            )
            
            upload_time = time.time() - upload_start
            print(f"  Upload time: {upload_time:.1f}s")
            
            if success:
                successful_uploads += 1
            else:
                failed_uploads += 0
            
            # Print progress
            print_memory_usage()
            batch_total_time = time.time() - batch_start_time
            avg_time = (time.time() - total_start_time) / (batch_idx + 1)
            remaining = avg_time * (len(batch_ranges) - batch_idx - 1)
            
            print(f"\n Progress: {batch_idx + 1}/{len(batch_ranges)} batches")
            print(f"   Time this batch: {batch_total_time:.1f}s")
            print(f"   Estimated remaining: {remaining/60:.1f} minutes")
            # Clear memory
            del batch_data
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"\n ERROR processing batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final summary
    total_time = time.time() - total_start_time
    
    print("\n" + "="*50)
    print(" EXTRACTION COMPLETE")
    print("="*50)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Successful batches: {successful_uploads}/{len(batch_ranges)}")
    print(f"Corrupted samples: {len(all_corrupted_samples)}")

if __name__ == "__main__":
    main()