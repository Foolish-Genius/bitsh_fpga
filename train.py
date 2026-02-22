import torch
import torch.optim as optim
from spiking_yolo import MiniSpikingYOLO
from yolo_dataset import get_dataloader
from yolo_encoder import YOLOEncoder
from yolo_loss import YoloLoss

def train_model():
    # 1. Hardware Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Firing up the training loop on: {device}")

    # 2. Initialize Core Components
    model = MiniSpikingYOLO().to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4) # Learning rate

    # Add a scheduler that cuts the learning rate if the loss stops decreasing
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    loss_fn = YoloLoss()
    encoder = YOLOEncoder()
    
    # 3. Load Data Pipeline
    # Using a batch size of 16 to maximize GPU usage
    dataloader = get_dataloader(batch_size=16) 
    num_steps = 16 # The temporal window for the SNN
    epochs = 100

    print("\n--- Starting Full Spiking YOLO Training ---")
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        # Loop through every batch of images from PASCAL VOC
        for batch_idx, (images, annotations) in enumerate(dataloader):
            images = images.to(device)
            
            # Translate raw text annotations into YOLO target tensors
            target_tensors = []
            for ann in annotations:
                target = encoder.encode(ann, img_size=64)
                target_tensors.append(target)
            
            # Stack the individual targets into a single batch tensor
            targets = torch.stack(target_tensors).to(device)

            # --- THE CORE SNN TRAINING LOOP ---
            optimizer.zero_grad()
            
            # Forward Pass: Unrolls the network over 'num_steps'
            outputs = model(images, num_steps)
            
            # Error Calculation
            loss = loss_fn(outputs, targets)
            
            # Backpropagation Through Time (BPTT) using Surrogate Gradients
            loss.backward()
            # --- NEW: Gradient Clipping ---
            # This acts like a fuse, blowing if the gradients surge too high, 
            # protecting your model weights from exploding.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            total_loss += loss.item()

            # Print an update every 10 batches
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch {batch_idx} | Loss: {loss.item():.4f}")
            break
        # End of Epoch summary
        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        print(f"=== Epoch {epoch+1} Complete | Average Loss: {avg_loss:.4f} ===\n")

    # 4. Save the Weights for Hardware Deployment
    torch.save(model.state_dict(), "spiking_yolo_weights.pt")
    print("Training finished! Weights saved to spiking_yolo_weights.pt")

if __name__ == "__main__":
    train_model()