import torch
import torchvision.transforms as transforms
from torchvision.ops import nms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from spiking_yolo import MiniSpikingYOLO

# The 20 categories our model was trained on
VOC_CLASSES = ["aeroplane"]

def visualize_prediction(image_path, weights_path="spiking_yolo_weights.pt"):
    print("Loading Spiking YOLO Brain...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load the Model Architecture and Trained Weights
    model = MiniSpikingYOLO().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval() # Set model to evaluation (testing) mode

    # 2. Prepare the Image
    raw_img = Image.open(image_path).convert("RGB")
    
    # We must format the image exactly how the hardware expects it
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    
    # Add a batch dimension of 1 -> Shape becomes [1, 3, 64, 64]
    img_tensor = transform(raw_img).unsqueeze(0).to(device) 

    # 3. Run the Forward Pass (The SNN Temporal Loop)
    num_steps = 16
    with torch.no_grad(): # Disable gradient calculation for speed
        output = model(img_tensor, num_steps) 
    
    # Strip the batch dimension to make array math easier -> Shape: [7, 7, 30]
    output = output.squeeze(0).cpu()
    
    S = 7
    img_size = 64

    boxes = []
    scores = []
    labels = []

    for i in range(S):
        for j in range(S):

            conf = output[i, j, 1].item()

            # Skip weak object predictions
            if conf < 0.7:
                continue

            class_scores = output[i, j, :1]
            class_idx = 0

            score = conf

            #if score < 0.5:
                #continue

            x_cell, y_cell, w, h = output[i, j, 2:6]

            x_center = (j + x_cell.item()) / S * img_size
            y_center = (i + y_cell.item()) / S * img_size
            # 1. Prevent the network from guessing a size larger than the image (1.0)
            box_w = min(w.item(), 1.0) * img_size
            box_h = min(h.item(), 1.0) * img_size

            # 2. Calculate corners
            raw_x1 = x_center - box_w / 2
            raw_y1 = y_center - box_h / 2
            raw_x2 = x_center + box_w / 2
            raw_y2 = y_center + box_h / 2

           # 3. YOLOv5 SHIFT: Unchain the center from the grid cell so it can slide right
            x_cell_shifted = x_cell.item() * 2.0 - 0.5
            y_cell_shifted = y_cell.item() * 2.0 - 0.5

            x_center = (j + x_cell_shifted) / S * img_size
            y_center = (i + y_cell_shifted) / S * img_size
            
            # 4. Prevent the network from guessing a size larger than the image
            box_w = min(w.item(), 1.0) * img_size
            box_h = min(h.item(), 1.0) * img_size

            raw_x1 = x_center - box_w / 2
            raw_y1 = y_center - box_h / 2
            raw_x2 = x_center + box_w / 2
            raw_y2 = y_center + box_h / 2

            # 5. CLAMP: Force the box to stay inside the 64x64 screen
            x1 = max(0.0, raw_x1)
            y1 = max(0.0, raw_y1)
            x2 = min(float(img_size), raw_x2)
            y2 = min(float(img_size), raw_y2)

            boxes.append([x1, y1, x2, y2])
            scores.append(score)
            labels.append(class_idx)

    # --- APPLY NMS ---
    if len(boxes) > 0:
        boxes = torch.tensor(boxes)
        scores = torch.tensor(scores)

        keep = nms(boxes, scores, 0.3)

        boxes = boxes[keep]
        scores = scores[keep]
        labels = [labels[k] for k in keep]

    # 4. Plotting Setup (MOVED THIS UP!)
    # We must create the 'ax' canvas BEFORE we try to draw boxes on it
    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.imshow(img_tensor.squeeze(0).cpu().permute(1, 2, 0))

    # --- DRAW FINAL BOXES ---
    if len(boxes) > 0:
        for box, score, cls in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box.tolist()
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                    linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1-2,
                    f"{VOC_CLASSES[cls]} ({score:.2f})",
                    color='white', fontsize=10,
                    backgroundcolor='red', weight='bold') 

    # Add our debug print to see the max confidence the network outputted
    print(f"DEBUG: Max Confidence Score in this image: {output[..., 1].max().item():.4f}")

    plt.title("Spiking YOLO Inference (64x64 Hardware Spec)")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Point this to any image inside your downloaded dataset!
    test_image = "./data/VOCdevkit/VOC2012/JPEGImages/2007_000033.jpg" 
    visualize_prediction(test_image)