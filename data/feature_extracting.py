import h5py
import os
import torch 
from PIL import Image
from torchvision import transforms
import numpy as np

import sys
sys.path.append("")
from models.vvg_net.features_extractor import VGG19

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 100
DATASET_SIZE = 30400
DATASET_PATH = ""
vgg_net = VGG19().to(device)
vgg_net.eval()
image_paths = sorted(os.glob.glob(os.path.join(DATASET_PATH, "*/*/*.png")))

transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def preprocess_data(image_paths, save_format: str = "hdf5", transform=None):
    with h5py.File('extracted_features_vgg19.hdf5', 'w') as hf: 
        conv3_4_layer = hf.create_dataset('conv3_4_features', shape=(DATASET_SIZE // 10, 256, 56,56), compression='gzip', chunks=(DATASET_SIZE // 10, 256, 56,56))
        conv4_4_layer = hf.create_dataset('conv4_4_features', shape=(DATASET_SIZE // 10, 512, 28,28), compression='gzip', chunks=(DATASET_SIZE // 10, 512, 28,28))
        conv5_4_layer = hf.create_dataset('conv5_4_features', shape=(DATASET_SIZE // 10, 512, 14,14), compression='gzip', chunks=(DATASET_SIZE // 10, 512, 14,14))

        for batch_idx in range(0, len(image_paths), BATCH_SIZE):
            batch_images = []
            batch_paths = image_paths[batch_idx:batch_idx + BATCH_SIZE]
            for image_path in batch_paths:
                img = Image.open(image_path).convert('RGB')
                if transform is not None:
                    img = transform(img)
                batch_images.append(img)
            batch_tensor = torch.stack(batch_images).to(device)
            with torch.no_grad():
                extracted_features = vgg_net(batch_tensor)
            end_idx = min(batch_idx + BATCH_SIZE, DATASET_SIZE)
            hf['conv3_4_features'][batch_idx:end_idx] = extracted_features[0].cpu().numpy()
            hf['conv4_4_features'][batch_idx:end_idx] = extracted_features[1].cpu().numpy()
            hf['conv5_4_features'][batch_idx:end_idx] = extracted_features[2].cpu().numpy()

