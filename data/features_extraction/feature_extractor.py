import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import gc
from pathlib import Path

class FeatureExtractor:
    def __init__(self, vgg_model, llm_model, config):
        self.vgg = vgg_model
        self.llm = llm_model
        self.config = config
        self.device = config.DEVICE

    def extract_batch(self, start_idx, end_idx):
        from dataset import MIMICDataset
        from torchvision import transforms
        
        # image_transforms = transforms.Compose([
        #     transforms.Resize((self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        # ])
        
        dataset = MIMICDataset(
            root_dir=self.config.DATA_ROOT,
            start_idx=start_idx,
            end_idx=end_idx,
            transform=None # because normalization is done by vgg19 forward 
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.MINI_BATCH_SIZE,  
            shuffle=False,  
            num_workers=self.config.NUM_WORKERS,
            prefetch_factor=self.config.PREFETCH_FACTOR,
            pin_memory=True  
        )
        
        all_conv3_4 = []
        all_conv4_4 = []
        all_conv5_4 = []
        all_report_emb = []
        all_sample_ids = []
        
        print(f"Processing samples {start_idx:05d} to {end_idx:05d}...")
        
        with torch.no_grad():  
            for images, texts, folder_ids in tqdm(dataloader, desc=f"Batch {start_idx}-{end_idx}"):
                images = images.to(self.device)
                
                conv3_4, conv4_4, conv5_4 = self.vgg(images)
                
                report_emb = self.llm(texts)
                
                all_conv3_4.append(conv3_4.cpu())
                all_conv4_4.append(conv4_4.cpu())
                all_conv5_4.append(conv5_4.cpu())
                all_report_emb.append(report_emb.cpu())
                all_sample_ids.extend(folder_ids)  
                
                if len(all_conv3_4) % 10 == 0:  
                    torch.cuda.empty_cache()
        
        print("Consolidating features...")
        batch_data = {
            'conv3_4': torch.cat(all_conv3_4, dim=0).half(),  
            'conv4_4': torch.cat(all_conv4_4, dim=0).half(),
            'conv5_4': torch.cat(all_conv5_4, dim=0).half(),
            'report_embeddings': torch.cat(all_report_emb, dim=0).half(),
            'sample_ids': all_sample_ids
        }
        
        num_samples = len(dataset)
        assert batch_data['conv3_4'].shape[0] == num_samples
        assert batch_data['conv4_4'].shape[0] == num_samples
        assert batch_data['conv5_4'].shape[0] == num_samples
        assert batch_data['report_embeddings'].shape[0] == num_samples
        assert len(batch_data['sample_ids']) == num_samples
        
        print(f" Extracted {num_samples} samples successfully")
        
        del all_conv3_4, all_conv4_4, all_conv5_4, all_report_emb
        gc.collect()
        torch.cuda.empty_cache()
        
        return batch_data

    def save_batch(self, batch_data, batch_idx):
        """Save batch and return filename"""
        import os
        
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        
        start = batch_idx * self.config.BATCH_SIZE
        end = min(start + self.config.BATCH_SIZE - 1, self.config.END_IDX)
        
        filename = f"batch_{start:05d}-{end:05d}.pt"
        filepath = Path(self.config.OUTPUT_DIR) / filename
        
        print(f"Saving {filename}...")
        torch.save(batch_data, filepath)
        
        if not filepath.exists():
            raise IOError(f"Failed to save {filename}")
        
        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f" Saved {filename} ({file_size_mb:.1f} MB)")
        
        return str(filepath)