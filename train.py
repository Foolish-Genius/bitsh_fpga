import torch
import torch.optim as optim
from tiny_yolo_qat import TinyYOLOQAT
from yolo_dataset import get_dataloader
from yolo_encoder import YOLOEncoder
from yolo_loss_qat import YoloLossQAT


def train_model():
    # 1. Hardware Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")

    # 2. Initialize Model
    model = TinyYOLOQAT(grid_size=7, num_boxes=1, num_classes=1).to(device)
        
    model.fuse_model()
    TinyYOLOQAT.prepare_qat(model)
    model = model.to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    loss_fn = YoloLossQAT()
    encoder = YOLOEncoder()

    # 5. Data
    dataloader = get_dataloader(batch_size=16)
    epochs = 50

    print("\n--- Starting Tiny YOLO + QAT Training ---")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (images, annotations) in enumerate(dataloader):
            images = images.to(device)

            # Encode ground-truth annotations into YOLO target tensors
            target_tensors = []
            for ann in annotations:
                target = encoder.encode(ann, img_size=64)
                target_tensors.append(target)
            targets = torch.stack(target_tensors).to(device)

            # Forward
            optimizer.zero_grad()
            outputs = model(images)

            # Loss
            loss = loss_fn(outputs, targets)

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            if batch_idx % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}] | "
                    f"Batch {batch_idx} | "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = epoch_loss / max(num_batches, 1)
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"=== Epoch {epoch+1} Complete | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"LR: {current_lr:.6f} ===\n"
        )

    # 6. Save the QAT-Float weights just in case
    torch.save(model.state_dict(), "tiny_yolo_qat_float_weights.pt")
    print("Float32 QAT weights saved to tiny_yolo_qat_float_weights.pt")

    # 7. Convert directly to INT8
    print("\n--- Converting to INT8 ---")
    # Move the trained model to CPU and put it in eval mode
    model = model.cpu()
    model.eval()
    
    # Convert the floating point weights to discrete 8-bit integers
    TinyYOLOQAT.convert_to_quantized(model)
    
    # Save the final FPGA-ready hardware weights
    torch.save(model.state_dict(), "tiny_yolo_int8_weights.pt")
    print("INT8 weights saved to tiny_yolo_int8_weights.pt")
    print("\nTraining complete!")

if __name__ == "__main__":
    train_model()