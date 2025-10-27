import h5py
import os
import torch 
from PIL import Image
from torchvision import transforms
import glob

import sys
sys.path.append("/content/xray-to-report")
from models.vgg_net.features_extractor import VGG19

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 100
DATASET_PATH = ""

vgg_net = VGG19().to(device)
vgg_net.eval()

transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# STEP 1: Filter valid images first
def get_valid_image_paths(image_paths):
    """Check which images can be opened and return valid paths."""
    valid_paths = []
    corrupted_count = 0
    
    print("Checking image validity...")
    for img_path in image_paths:
        try:
            # Try to open and verify
            with Image.open(img_path) as img:
                img.verify()  # Verify it's a valid image
            valid_paths.append(img_path)
        except Exception as e:
            corrupted_count += 1
            print(f"Corrupted/missing: {img_path}")
    
    print(f"\nTotal images found: {len(image_paths)}")
    print(f"Valid images: {len(valid_paths)}")
    print(f"Corrupted/missing images: {corrupted_count}")
    
    return valid_paths, corrupted_count


def preprocess_data(image_paths, save_format: str = "hdf5", transform=transforms):
    # Filter valid images first
    valid_image_paths, corrupted_count = get_valid_image_paths(image_paths)
    
    if len(valid_image_paths) == 0:
        print("No valid images found!")
        return
    
    ACTUAL_DATASET_SIZE = len(valid_image_paths)
    
    with h5py.File('extracted_features_vgg19.hdf5', 'w') as hf: 
        # Create datasets with ACTUAL size
        conv3_4_layer = hf.create_dataset('conv3_4_features', 
                                          shape=(ACTUAL_DATASET_SIZE, 256, 56, 56), 
                                          compression='gzip', 
                                          chunks=(BATCH_SIZE, 256, 56, 56))
        conv4_4_layer = hf.create_dataset('conv4_4_features', 
                                          shape=(ACTUAL_DATASET_SIZE, 512, 28, 28), 
                                          compression='gzip', 
                                          chunks=(BATCH_SIZE, 512, 28, 28))
        conv5_4_layer = hf.create_dataset('conv5_4_features', 
                                          shape=(ACTUAL_DATASET_SIZE, 512, 14, 14), 
                                          compression='gzip', 
                                          chunks=(BATCH_SIZE, 512, 14, 14))

        processed_count = 0
        
        for batch_idx in range(0, len(valid_image_paths), BATCH_SIZE):
            batch_images = []
            batch_paths = valid_image_paths[batch_idx:batch_idx + BATCH_SIZE]
            
            # STEP 2: Skip corrupted images during batch processing
            for image_path in batch_paths:
                try:
                    img = Image.open(image_path).convert('RGB')
                    if transform is not None:
                        img = transform(img)
                    batch_images.append(img)
                except Exception as e:
                    print(f"Error loading {image_path}: {e}")
                    # Add black placeholder image
                    batch_images.append(torch.zeros(3, 224, 224))
            
            if len(batch_images) == 0:
                continue
                
            batch_tensor = torch.stack(batch_images).to(device)
            
            with torch.no_grad():
                extracted_features = vgg_net(batch_tensor)
            
            # Write actual batch size
            actual_batch_size = len(batch_images)
            end_idx = processed_count + actual_batch_size
            
            hf['conv3_4_features'][processed_count:end_idx] = extracted_features[0].cpu().numpy()
            hf['conv4_4_features'][processed_count:end_idx] = extracted_features[1].cpu().numpy()
            hf['conv5_4_features'][processed_count:end_idx] = extracted_features[2].cpu().numpy()
            
            processed_count += actual_batch_size
            
            if (batch_idx // BATCH_SIZE) % 10 == 0:
                print(f"Processed {processed_count}/{ACTUAL_DATASET_SIZE} images")
            
            # Clean up memory
            del batch_tensor, extracted_features, batch_images
            torch.cuda.empty_cache()
    
    print(f"\nFeature extraction complete!")
    print(f"Total processed: {processed_count}")
    print(f"Saved to: extracted_features_vgg19.hdf5")


if __name__ == "__main__":
    all_image_paths = sorted(glob.glob(os.path.join(DATASET_PATH, "*/*/*.png")))
    preprocess_data(all_image_paths)