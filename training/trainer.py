"""Main training script for Q-Former"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch
import yaml
from torch.utils.data import DataLoader
import sys
import numpy as np
from tqdm import tqdm

sys.path.append("/content/xray-to-report")

from models.vgg_net.features_extractor import VGG19
from models.qformer.qformer import HierarchicalXRayQformer
from models.text_encoder.embeddings_extractor import LLMEmbedder
from training.train import TrainQformer
from data.dataset import XRayReportDataset
def create_collate_fn(skip_corrupted=True):
    """Create a collate function that filters out corrupted samples."""
    def collate_fn(batch):
        if skip_corrupted:
            valid_batch = [item for item in batch if not item.get('corrupted', False)]
            if not valid_batch:
                return None
            batch = valid_batch
        
        return torch.utils.data.default_collate(batch)
    
    return collate_fn
def setup_training(config_path="/content/xray-to-report/training/configs/config.yaml"):
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    training_config = config['training']
    model_config = config['model']
    data_config = config['data']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Initializing VGG19...")
    vgg = VGG19()
    
    print("Initializing ClinicalBERT...")
    llm_embedder = LLMEmbedder(model_config)
    
    print("Initializing Q-Former...")
    qformer = HierarchicalXRayQformer(
        num_queries=model_config.get('num_queries', 32),
        hidden_dim=model_config.get('hidden_dim', 768),
        num_layers=model_config.get('num_layers', 6),
        num_heads=model_config.get('num_heads', 12),
        intermediate_size=model_config.get('intermediate_size', 3072),
        dropout=model_config.get('dropout', 0.1)
    )
    
    print("Loading datasets...")
    train_dataset = XRayReportDataset(
        data_root=data_config['data_root'],
        split='train',
        train_ratio=0.8,
        val_ratio=0.1,
    )

    val_dataset = XRayReportDataset(
        data_root=data_config['data_root'],
        split='val',
        train_ratio=0.8,  
        val_ratio=0.1,    
        transform=None,
        max_samples=data_config.get('max_samples')
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=data_config.get('num_workers', 2),  
        collate_fn=create_collate_fn(skip_corrupted=True),
        pin_memory=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=data_config.get('num_workers', 2),
        collate_fn=create_collate_fn(skip_corrupted=True),
        pin_memory=True
    )
    
    print("Initializing trainer...")
    trainer = TrainQformer(
        qformer=qformer.to(device),
        vgg=vgg.to(device),
        llmEmbedder=llm_embedder.to(device),
        training_config=training_config
    )
    
    return trainer, train_dataloader, val_dataloader

def main():
    """Main training function"""
    
    trainer, train_dataloader, val_dataloader = setup_training()
    
    print("Starting training...")
    print(f"Training samples: {len(train_dataloader.dataset)}")
    print(f"Validation samples: {len(val_dataloader.dataset)}")
    print(f"Batch size: {train_dataloader.batch_size}")
    print(f"Number of epochs: {trainer.config['num_epochs']}")
    print("-" * 50)
    
    try:
        trainer.train(train_dataloader, val_dataloader, checkpoint_path=None)
        print("Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer.save_checkpoint("interrupted_training.pth")
        print("Saved interrupted training checkpoint")
    
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise

if __name__ == "__main__":
    main()