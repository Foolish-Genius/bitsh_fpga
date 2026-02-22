import torch
import torchvision.transforms as transforms
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader
import kagglehub
import os
import shutil

# 1. Transforms (Downscaling to 64x64 for FPGA hardware efficiency)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
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

class AeroplaneOnlyDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset):
        self.dataset = original_dataset
        self.valid_indices = []
        
        print("Scanning 11,000+ images for aerial targets... (This takes about 10 seconds)")
        for i in range(len(original_dataset)):
            img, ann = original_dataset[i]
            objects = ann["annotation"].get("object", [])
            
            if isinstance(objects, dict):
                objects = [objects]
            
            # Check if an airplane exists in this specific image
            has_plane = any(obj["name"] == "aeroplane" for obj in objects)
            if has_plane:
                self.valid_indices.append(i)
                
        print(f"Filter Complete! Found {len(self.valid_indices)} high-value target images.")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        return self.dataset[self.valid_indices[idx]]

# 4. Initialize DataLoader
def get_dataloader(batch_size=8):
    data_root = setup_data_via_api()
    
    print("Initializing PyTorch VOCDetection...")
    raw_dataset = VOCDetection(root=data_root, year='2012', image_set='train', 
                               download=False, transform=transform)
    
    # Wrap the raw dataset in our custom filter!
    target_dataset = AeroplaneOnlyDataset(raw_dataset)
    
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