import sys
import os
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import XRayReportDataset
from models.vgg_net.features_extractor import VGG19
from models.text_encoder.embeddings_extractor import LLMEmbedder

print("Testing dataset...")
dataset = XRayReportDataset(data_root="mimicDatatotal", max_samples=10)
sample = dataset[0]
print(f"Image shape: {sample['image'].shape}")
print(f"Report: {sample['report'][:100]}...")

print("\nTesting models...")
vgg = VGG19()
llm = LLMEmbedder({"LLM_MODEL_NAME": "emilyalsentzer/Bio_ClinicalBERT", "MAX_REPORT_LENGTH": 512})

# Test forward pass
image = sample['image'].unsqueeze(0)  
with torch.no_grad():
    features = vgg(image)
    text_emb = llm([sample['report']])

print(f"VGG features: {[f.shape for f in features]}")
print(f"Text embedding: {text_emb.shape}")

print("All tests passed! ")