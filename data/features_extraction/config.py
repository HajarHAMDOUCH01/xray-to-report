
DATA_ROOT = ""  
OUTPUT_DIR = "./temp_batches"  


LANGUAGE_ENCODER_NAME = "emilyalsentzer/Bio_ClinicalBERT"

# Processing parameters
BATCH_SIZE = 1000  
MINI_BATCH_SIZE = 16  
NUM_WORKERS = 4 
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