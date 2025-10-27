import torch
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = ""  
OUTPUT_DIR = "./temp_batches"  


LLM_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

# Processing parameters
# 30632 
TOTAL_SAMPLES = 2000
BATCH_SIZE = 1000  
START_IDX = 0
END_IDX = 1999
MINI_BATCH_SIZE = 32  
NUM_WORKERS = 4 
PREFETCH_FACTOR = 2
UPLOAD_EVERY_N_BATCHES = 5  

# Model configurations
IMAGE_SIZE = 224  
MAX_REPORT_LENGTH = 512  

# Kaggle dataset info
KAGGLE_DATASET_SLUG = "hajarhamdouch01/mimic-features-v1"
KAGGLE_USERNAME = "hajarhamdouch01"

# Feature dimensions
EXPECTED_SHAPES = {
    'conv3_4': (256, 56, 56),
    'conv4_4': (512, 28, 28),
    'conv5_4': (512, 14, 14),
    'report_embeddings': (768,)  
}

NUM_BATCHES = TOTAL_SAMPLES // BATCH_SIZE