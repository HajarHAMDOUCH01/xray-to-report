import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import sys 
sys.path.append("/content/xray-to-report")

import data.features_extraction.config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LLMEmbedder(nn.Module):
    def __init__(self, config):
        super(LLMEmbedder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.max_length = 512
        self.model.to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, texts):
        """
        inputs : list of strings (reports)
        outputs : (batch_size, hidden_dim)
        """
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

            attention_mask = inputs['attention_mask'].unsqueeze(-1)
            masked_embeddings = outputs.last_hidden_state * attention_mask

            sum_embeddings = masked_embeddings.sum(dim=1)
            sum_mask = attention_mask.sum(dim=1)

            embeddings = sum_embeddings / torch.clamp(sum_mask, min=1e-9)

            return embeddings