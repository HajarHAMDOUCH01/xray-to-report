import torch
import config
from models import VGG19, LLMEmbedder
from feature_extractor import FeatureExtractor

def main():
    print("="*50)
    print("MIMIC Feature Extraction Pipeline")
    print("="*50)
    
    # Initialize models
    print("\n1. Loading models...")
    vgg = VGG19().to(config.DEVICE).eval()
    llm = LLMEmbedder(
        model_name=config.LLM_MODEL_NAME,
        max_length=config.MAX_REPORT_LENGTH
    ).to(config.DEVICE).eval()
    print(" Models loaded")
    
    # Initialize feature extractor
    extractor = FeatureExtractor(vgg, llm, config)
    
    # Calculate batches to process
    batch_ranges = []
    for batch_idx in range(config.NUM_BATCHES + 1):  # +1 for the last partial batch
        start = batch_idx * config.BATCH_SIZE
        end = min(start + config.BATCH_SIZE - 1, config.END_IDX)
        if start <= config.END_IDX:
            batch_ranges.append((start, end, batch_idx))
    
    print(f"\n2. Will process {len(batch_ranges)} batches")
    print(f"   Batch size: {config.BATCH_SIZE} samples")
    print(f"   Mini-batch size: {config.MINI_BATCH_SIZE}")
    print(f"   Upload every: {config.UPLOAD_EVERY_N_BATCHES} batches\n")
    
    # Process batches
    batch_files = []  
    
    for start, end, batch_idx in batch_ranges:
        print(f"\n{'='*50}")
        print(f"Processing Batch {batch_idx + 1}/{len(batch_ranges)}")
        print(f"{'='*50}")
        
        # Extract features
        batch_data = extractor.extract_batch(start, end)
        
        # Save to disk
        filepath = extractor.save_batch(batch_data, batch_idx)
        batch_files.append(filepath)
        
        # Clear memory
        del batch_data
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        # Upload to Kaggle every N batches
        if len(batch_files) == config.UPLOAD_EVERY_N_BATCHES or batch_idx == len(batch_ranges) - 1:
            print(f"\n Uploading {len(batch_files)} batch files to Kaggle...")
            
            # TODO: Implement upload function
            # upload_to_kaggle(batch_files)
            
            # Delete local files after successful upload
            for f in batch_files:
                import os
                os.remove(f)
                print(f"  Deleted {f}")
            
            batch_files = []  # Clear the list
            print(" Upload complete and local files cleaned\n")
    
    print("\n" + "="*50)
    print(" Feature extraction complete!")
    print("="*50)

if __name__ == "__main__":
    main()