import torch
import torchvision.transforms as transforms
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader
import kagglehub
import os
import shutil
import random
from torch.utils.data import Dataset

# 1. Transforms (Downscaling to 64x64 for FPGA hardware efficiency)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3), 
    transforms.RandomHorizontalFlip(p=0.5), 
    transforms.ToTensor(), 
])

# 2. Custom Collate Function
def yolo_collate_fn(batch):
    images = []
    annotations = []
    for img, ann in batch:
        images.append(img)
        annotations.append(ann)
    
    images = torch.stack(images, dim=0)
    return images, annotations

# 3. Kaggle API Downloader for the huanghanchina dataset
def setup_data_via_api():
    print("Downloading the huanghanchina dataset via Kaggle API...")
    
    # Pointing exactly to the dataset you linked
    kaggle_path = kagglehub.dataset_download("huanghanchina/pascal-voc-2012")
    
    local_data_dir = './data'
    vocdevkit_path = os.path.join(local_data_dir, 'VOCdevkit')
    
    if not os.path.exists(vocdevkit_path):
        print("Structuring data locally for PyTorch...")
        os.makedirs(local_data_dir, exist_ok=True)
        
        # Search the Kaggle cache for the VOC2012 folder
        for root, dirs, files in os.walk(kaggle_path):
            if 'VOC2012' in dirs:
                source_voc = os.path.join(root, 'VOC2012')
                target_voc = os.path.join(vocdevkit_path, 'VOC2012')
                
                print(f"Copying files into {target_voc}... (Please wait)")
                shutil.copytree(source_voc, target_voc)
                print("Copy complete!")
                break
    else:
        print("Data is already correctly structured in ./data/VOCdevkit")
        
    return local_data_dir

class BalancedDefenseDataset(Dataset):
    """
    Hackathon Dataset Manager:
    Keeps all ~1000 airplane targets, and mixes in ~1000 negative samples
    (cats, trains, empty skies) to teach the network how to ignore noise.
    """
    def __init__(self, raw_dataset, negative_ratio=1.0):
        self.raw = raw_dataset
        
        positives = []
        negatives = []

        print("Scanning dataset for targets and negative samples...")
        for i in range(len(raw_dataset)):
            # PASCAL VOC stores annotations in a dictionary
            ann = raw_dataset[i][1]['annotation']
            objects = ann.get('object', [])
            
            # Check if there is an airplane in this image
            has_plane = any(obj['name'] == 'aeroplane' for obj in objects)
            
            if has_plane:
                positives.append(i)
            else:
                negatives.append(i)

        # 1. Keep all images that have airplanes
        # 2. Randomly sample 'negative_ratio' times that amount of non-airplane images
        random.seed(42) # Ensure reproducible hackathon results
        num_negatives = int(len(positives) * negative_ratio)
        sampled_negatives = random.sample(negatives, min(num_negatives, len(negatives)))

        self.valid_indices = positives + sampled_negatives
        random.shuffle(self.valid_indices) # Mix them up!
        
        print(f"Dataset Balanced! Targets: {len(positives)} | Negatives: {len(sampled_negatives)}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        image, target = self.raw[real_idx]

        # CRITICAL FIX: We must strip out the bounding boxes for cats/trains/etc.
        # If it is not an aeroplane, the encoder receives an empty list [] 
        # This forces the YOLO loss function to push confidence to 0.0!
        objects = target['annotation'].get('object', [])
        filtered_objects = [obj for obj in objects if obj['name'] == 'aeroplane']
        
        target['annotation']['object'] = filtered_objects

        return image, target

# 4. Initialize DataLoader
def get_dataloader(batch_size=8):
    
    data_root = setup_data_via_api()
    
    print("Initializing PyTorch VOCDetection...")
    raw_dataset = VOCDetection(root=data_root, year='2012', image_set='train', 
                               download=False, transform=transform)
    
    # Wrap the raw dataset in our custom filter!
    target_dataset = BalancedDefenseDataset(raw_dataset)
    
    # Now the dataloader will ONLY pull images that actually contain airplanes
    dataloader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, 
                            collate_fn=yolo_collate_fn)
    return dataloader

if __name__ == "__main__":
    # Test the pipeline
    loader = get_dataloader(batch_size=4)
    images, annotations = next(iter(loader))
    
    print("\n--- Pipeline Test Successful ---")
    print(f"Batch Image Tensor Shape: {images.shape}")