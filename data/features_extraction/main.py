import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from models import VGG19, LLMEmbedder  
from dataset import MIMICDataset
import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# test with first 10 samples (folders 00000-00009)
print("Creating dataset...")
test_dataset = MIMICDataset(
    root_dir=config["DATA_ROOT"],  
    start_idx=0,
    end_idx=9,
    transform=image_transforms
)

print(f"Dataset size: {len(test_dataset)}")

# Create DataLoader
test_loader = DataLoader(
    test_dataset,
    batch_size=10,      
    shuffle=False,
    num_workers=2       
)

print("Loading models...")
vgg = VGG19().to(device).eval()
llm = LLMEmbedder(model_name="emilyalsentzer/Bio_ClinicalBERT", max_length=512).to(device).eval()

print("Models loaded!")

print("\nProcessing batch...")
for images, texts, folder_ids in test_loader:
    print(f"Batch info:")
    print(f"  Images shape: {images.shape}")  # Should be (10, 3, 224, 224)
    print(f"  Number of texts: {len(texts)}")
    print(f"  Folder IDs: {folder_ids}")
    
    images = images.to(device)
    
    print("\nExtracting VGG19 features...")
    with torch.no_grad():
        vgg_features = vgg(images)  # Returns [conv3_4, conv4_4, conv5_4]
    
    conv3_4, conv4_4, conv5_4 = vgg_features
    
    print(f"  conv3_4: {conv3_4.shape}")  # Expected: (10, 256, 56, 56)
    print(f"  conv4_4: {conv4_4.shape}")  # Expected: (10, 512, 28, 28)
    print(f"  conv5_4: {conv5_4.shape}")  # Expected: (10, 512, 14, 14)
    
    print("\nExtracting report embeddings...")
    with torch.no_grad():
        report_embeddings = llm(texts) 
    
    print(f"  report_embeddings: {report_embeddings.shape}")  
    
    # Check GPU memory
    print(f"\nGPU Memory Used: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    
    # Verify all shapes are correct
    assert conv3_4.shape == (10, 256, 56, 56), "conv3_4 shape mismatch!"
    assert conv4_4.shape == (10, 512, 28, 28), "conv4_4 shape mismatch!"
    assert conv5_4.shape == (10, 512, 14, 14), "conv5_4 shape mismatch!"
    assert report_embeddings.shape == (10, 768), "report_embeddings shape mismatch!"
    
    print("\n All shapes correct!")
    
    # Sample one report to verify quality
    print(f"\nSample report text (first 200 chars):")
    print(texts[0][:200])
    
    print(f"\nSample embedding values (first 10):")
    print(report_embeddings[0, :10])
    
    break  