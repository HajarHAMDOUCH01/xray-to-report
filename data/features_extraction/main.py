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
    print("MIMIC Feature Extraction Pipeline (Chunked Upload)")
    print("="*50)
    
    print("\n1. Loading models...")
    vgg = VGG19().to(config.DEVICE).eval()
    llm = LLMEmbedder(config).to(config.DEVICE).eval()
    print(" Models loaded")
    
    extractor = FeatureExtractor(vgg, llm, config)
    uploader = KaggleUploader(
        dataset_slug=config.KAGGLE_DATASET_SLUG,
        username=config.KAGGLE_USERNAME
    )
    
    batch_ranges = []
    for batch_idx in range(config.NUM_BATCHES + 1):
        start = batch_idx * config.BATCH_SIZE
        end = min(start + config.BATCH_SIZE - 1, config.END_IDX)
        if start <= config.END_IDX:
            batch_ranges.append((start, end, batch_idx))
    
    print(f"\n2. Will process {len(batch_ranges)} batches")
    print(f"   Upload chunks: at 15 batches and at completion\n")
    
    all_corrupted_samples = []
    successful_batches = 0
    failed_batches = 0
    total_start_time = time.time()
    
    # Upload chunks at these batch indices
    upload_checkpoints = [1,2,3]  # Upload after batches 15 and 30
    
    for start, end, batch_idx in batch_ranges:
        batch_start_time = time.time()
        
        print(f"\n{'='*50}")
        print(f"Batch {batch_idx + 1}/{len(batch_ranges)}")
        print(f"{'='*50}")
        
        try:
            batch_data = extractor.extract_batch(start, end)
            
            if 'corrupted_samples' in batch_data:
                all_corrupted_samples.extend(batch_data['corrupted_samples'])
            
            extraction_time = time.time() - batch_start_time
            print(f"  Extraction time: {extraction_time:.1f}s")
            
            # Save to staging
            current_batch_file = f"batch_{batch_idx:04d}.pt"
            staging_batch_path = uploader.staging_dir / current_batch_file
            torch.save(batch_data, staging_batch_path)
            print(f"  Saved batch {batch_idx} to staging")
            
            successful_batches += 1
            
            # Clear memory but keep files in staging
            del batch_data
            gc.collect()
            torch.cuda.empty_cache()
            
            # Memory management only - no uploads during processing
            print_memory_usage()
            batch_total_time = time.time() - batch_start_time
            avg_time = (time.time() - total_start_time) / (batch_idx + 1)
            remaining = avg_time * (len(batch_ranges) - batch_idx - 1)
            
            print(f"\n Progress: {batch_idx + 1}/{len(batch_ranges)} batches")
            print(f"   Time this batch: {batch_total_time:.1f}s")
            print(f"   Estimated remaining: {remaining/60:.1f} minutes")
            
        except Exception as e:
            print(f"\n ERROR processing batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            failed_batches += 1
            continue
    
    # Final upload of remaining batches
    try:
        total_time = time.time() - total_start_time
        
        print("\n" + "="*50)
        print(" EXTRACTION COMPLETE")
        print("="*50)
        print(f"Total extraction time: {total_time/60:.1f} minutes")
        print(f"Successful batches: {successful_batches}/{len(batch_ranges)}")
        print(f"Failed batches: {failed_batches}/{len(batch_ranges)}")
        print(f"Corrupted samples: {len(all_corrupted_samples)}")
        
        # Upload all batches at once
        if successful_batches > 0:
            print("\nStarting final upload of all batches...")
            final_batches = list(uploader.staging_dir.glob("batch_*.pt"))
            if final_batches:
                upload_start = time.time()
                success = uploader.upload_chunk(
                    upload_checkpoint=len(batch_ranges)-1, 
                    chunk_name="complete_dataset"
                )
                upload_time = time.time() - upload_start
                if success:
                    print(f"✓ Complete dataset uploaded in {upload_time/60:.1f} minutes")
                    print(f"✓ Uploaded {len(final_batches)} total batches")
                else:
                    print("✗ Final upload failed")
            else:
                print("No batches found for upload")
                
    finally:
        uploader.cleanup()

if __name__ == "__main__":
    main()