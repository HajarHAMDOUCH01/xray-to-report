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

class TrainQformer:
    # they are initialized in trainer and the models are moved to cuda before using them to instantiate this class 
    def __init__(self, qformer, vgg, llmEmbedder, training_config):    
        self.device = device
        # just for consistency     
        self.qformer_model = qformer.to(device) 
        self.vgg_model = vgg.to(device)
        self.llmembedder_model = llmEmbedder.to(device)

        self.config = training_config
        
        self.optimizer = AdamW(self.qformer_model.parameters(), lr=training_config["lr"], fused=True if device.type == 'cuda' else False)

        # we have to track 
        self.best_val_loss = float('inf')

    def contrastive_loss(self, image_features, text_embeddings, temperature=0.07):
        # print("DEBUG image shape:", image_features.shape)
        # print("DEBUG text shape:", text_embeddings.shape)

        if len(image_features.shape) == 3:
            image_features = image_features.mean(dim=1)
        # this is L2 Normalization of the features and the embdessings
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
            'epoch' : epoch,
            'config': self.config
        }, path)
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.qformer_model.load_state_dict(checkpoint['qformer_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch = checkpoint['epoch']
        # return epoch
        return 1
    
    def train_epoch(self, dataloader):
        # initializing vgg and text encoder 
        self.vgg_model.eval().requires_grad_(False)
        self.llmembedder_model.eval().requires_grad_(False)

        # initalizing the qformer 
        self.qformer_model.train().requires_grad_(True)

        total_loss = 0
        num_batches = 0

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for batch in tqdm(dataloader, desc="Training"):
            xrays = batch['image'].to(self.device, non_blocking=True) # the image is a tensor , so it's moved to gpu 
            reports = batch['report']
            # print(reports)

            self.optimizer.zero_grad()

            with torch.no_grad():
                assert xrays.device == self.vgg_model.weight.device , "xrays and vgg should be in the same device !"
                vgg_features = self.vgg_model(xrays)
                assert xrays.device == self.vgg_model.weight.device , "xrays and vgg should be in the same device !"
                text_embeddings = self.llmembedder_model(reports)

                # now vgg_features and the embeddings are retuned by models that are already on device so they are on that device 

            image_embeddings = self.qformer_model(vgg_features)
            

            loss = self.contrastive_loss(
                image_embeddings,
                text_embeddings,
                temperature=self.config["temperature"]
            )
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                            self.qformer_model.parameters(), 
                            self.config["max_grad_norm"]
                        )
            self.optimizer.step()
            self.optimizer.zero_grad()
            total_loss += loss.item()
            num_batches += 1
        return total_loss / num_batches 


    def validate(self, dataloader):
        self.qformer_model.eval()
        self.vgg_model.eval()
        self.llmembedder_model.eval()
        
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                xrays = batch['image'].to(self.device, non_blocking=True)
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
        start_epoch = 1
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            epoch = self.load_checkpoint(checkpoint_path) # models are class attributes , thei states are changed here 
            start_epoch = epoch + 1 # for example we save epoch 1 and we start directly from 2
            print(f"checkpoint laoded from : {checkpoint_path} and starting from epoch : ", start_epoch)

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        for epoch in range(start_epoch, self.config["num_epochs"]):
            print(f"Epoch {epoch}/{self.config['num_epochs']}")
            train_loss = self.train_epoch(train_dataloader)
            print(f"Train Loss: {train_loss:.4f}")

            if val_dataloader is not None:
                val_loss = self.validate(val_dataloader)
                print(f"Val Loss: {val_loss:.4f}")
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(f"/content/drive/MyDrive/best_qformer_{epoch}.pth", epoch)
                    print(f"Saved best model for epoch {epoch}!")
            del train_loss, val_loss
            
            print("-" * 50)


