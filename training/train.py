"""Only the Qformer is going to be trained :
Image: X-ray → VGG19 (frozen) → multi-scale features
Text: Report → Bio_ClinicalBERT (frozen) → text embedding
Q-Former: Takes image features → produces query embeddings
Loss: Compare Q-Former outputs with ClinicalBERT embeddings
"""

"""Training of the qformer with contrastive loss"""

from torch import Tensor, nn
import torch
import sys 

sys.path.append("")
from models.vgg_net.features_extractor import VGG19
from models.qformer.qformer import HierarchicalXRayQformer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

training_config = {
    "batch_size" : 32,
    "num_epoches" : 50,
    ""
}

class TrainQformer:
    def __init__(self, qformer, vgg, llmEmbedder, training_config):
        self.qformer_model = qformer
        self.vgg_model = vgg
        self.llmembedder_model = llmEmbedder
        self.config = training_config

        self.device = device

    def train_epoch(self):
        # initializing vgg and text encoder 
        vgg = 


