"""Only the Qformer is going to be trained :
Image: X-ray → VGG19 (frozen) → multi-scale features
Text: Report → Bio_ClinicalBERT (frozen) → text embedding
Q-Former: Takes image features → produces query embeddings
Loss: Compare Q-Former outputs with ClinicalBERT embeddings
"""

"""Training of the qformer with contrastive loss"""
from tqdm import tqdm
from torch import Tensor, nn
import torch.nn.functional as F
from torch.optim import AdamW
import torch
import sys 
import os

sys.path.append("/content/xray-to-report")
from models.vgg_net.features_extractor import VGG19
from models.qformer.qformer import HierarchicalXRayQformer
from models.text_encoder.embeddings_extractor import LLMEmbedder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# training_config = {
#     "batch_size" : 32,
#     "num_epochs" : 50,
#     "lr" : 1e-4,
#     "temperature" : 0.07
# }

class TrainQformer:
    def __init__(self, qformer, vgg, llmEmbedder, training_config):
        self.qformer_model = qformer.to(device)
        self.vgg_model = vgg
        self.llmembedder_model = llmEmbedder
        self.config = training_config
        self.device = device

        self.optimizer = AdamW(self.qformer_model.parameters(), lr=training_config["lr"])

    def contrastive_loss(self, image_features, text_embeddings, temperature=0.07):
        # print("DEBUG image shape:", image_features.shape)
        # print("DEBUG text shape:", text_embeddings.shape)

        if len(image_features.shape) == 3:
            image_features = image_features.mean(dim=1)
        # this is L2 Normalization of the features 
        image_features = F.normalize(image_features, p=2, dim=1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)

        # similarity matrix 
        logits = torch.matmul(image_features, text_embeddings.T) /temperature
        batch_size = image_features.shape[0] # geting the batch size 
        labels = torch.arange(batch_size, device=self.device)

        loss_i = F.cross_entropy(logits, labels)      # image to text
        loss_t = F.cross_entropy(logits.T, labels)    # text to image

        return (loss_i + loss_t) / 2
    def save_checkpoint(self, path, epoch):
        torch.save({
            'qformer_state_dict': self.qformer_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch' : epoch+1,
            'config': self.config
        }, path)
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.qformer_model.load_state_dict(checkpoint['qformer_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        return epoch
    
    def train_epoch(self, dataloader):
        # initializing vgg and text encoder 
        self.vgg_model.eval().requires_grad_(False)
        self.llmembedder_model.eval().requires_grad_(False)

        self.vgg_model.to(device)
        self.llmembedder_model.to(device)

        self.qformer_model.train()

        total_loss = 0
        num_batches = 0

        for batch in tqdm(dataloader, desc="Training"):
            xrays = batch['image'].to(self.device)
            reports = batch['report']
            # print(reports)

            self.optimizer.zero_grad()

            with torch.no_grad():
                vgg_features = self.vgg_model(xrays)
                text_embeddings = self.llmembedder_model(reports)

            image_embeddings = self.qformer_model(vgg_features)

            loss = self.contrastive_loss(
                image_embeddings,
                text_embeddings,
                temperature=self.config["temperature"]
            )
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
        return total_loss / num_batches 


    def validate(self, dataloader):
        self.qformer_model.eval()
        
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                xrays = batch['image'].to(self.device)
                reports = batch['report']
                
                vgg_features = self.vgg_model(xrays)
                text_embeddings = self.llmembedder_model(reports)
                
                image_embeddings = self.qformer_model(vgg_features)
                
                loss = self.contrastive_loss(
                    image_embeddings, 
                    text_embeddings,
                    temperature=self.config["temperature"]
                )
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, train_dataloader, val_dataloader=None, checkpoint_path=None):
        start_epoch = 0
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            epoch = self.load_checkpoint(checkpoint_path) # models are class attributes , thei states are changed here 
            start_epoch = epoch+1 # for example we save epoch 1 and we start directly from 2
            print(f"checkpoint laoded from : {checkpoint_path} and starting from epoch : ", start_epoch)
        best_val_loss = float('inf')
        for epoch in range(start_epoch, self.config["num_epochs"]):
            print(f"Epoch {epoch+1}/{self.config['num_epochs']}")
            train_loss = self.train_epoch(train_dataloader)
            print(f"Train Loss: {train_loss:.4f}")

            if val_dataloader is not None:
                val_loss = self.validate(val_dataloader)
                print(f"Val Loss: {val_loss:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(f"/content/drive/MyDrive/best_qformer_{epoch+1}.pth")
                    print(f"Saved best model for epoch {epoch+1}!")
            
            print("-" * 50)


