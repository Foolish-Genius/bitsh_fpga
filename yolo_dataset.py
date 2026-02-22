import torch
import torchvision.transforms as transforms
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader

# https://gemini.google.com/share/74e3bc3ac7ff
# 1. Transforms: Hardware constraints dictate our architecture. 
# To fit within the limited BRAM of your target hardware, we aggressively 
# downscale the images to 64x64 to match the MicroSpikingYOLO backbone you just ran.
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(), # Converts pixel values from 0-255 to floats between 0.0-1.0
])

# 2. Custom Collate Function: 
# This prevents PyTorch from crashing when images in the same batch have a different number of bounding boxes.
def yolo_collate_fn(batch):
    images = []
    annotations = []
    for img, ann in batch:
        images.append(img)
        annotations.append(ann)
    
    # Stack images into a single hardware-ready tensor: [Batch, Channels, Height, Width]
    images = torch.stack(images, dim=0)
    return images, annotations

def get_dataloader(batch_size=8):
    print("Initializing PASCAL VOC Dataset...")
    print("Note: The first run will download the dataset (~2GB). Grab a coffee!")
    
    # Load the training data (Changed download to False)
    dataset = VOCDetection(root='./data', year='2012', image_set='train', 
                           download=False, transform=transform)
    
    # Wrap it in a DataLoader to handle batching and shuffling
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                            collate_fn=yolo_collate_fn)
    return dataloader

if __name__ == "__main__":
    # Test the loader with a batch size of 4
    loader = get_dataloader(batch_size=4)
    
    # Grab one batch of data to verify the pipeline
    images, annotations = next(iter(loader))
    
    print("\n--- Pipeline Test Successful ---")
    print(f"Batch Image Tensor Shape: {images.shape}")
    print(f"Number of annotation sets in this batch: {len(annotations)}")
    
    # Dig into the raw annotation XML for the first image to see the bounding box data
    print("\nRaw Object Data for Image 1:")
    objects = annotations[0]['annotation']['object']
    for obj in objects:
        name = obj['name']
        bbox = obj['bndbox']
        print(f" - Found Object: '{name}' | Coordinates: {bbox}")
    
