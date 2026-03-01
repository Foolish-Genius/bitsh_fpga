import torch
import torch.ao.quantization as quant
import torchvision.transforms as transforms
from torchvision.ops import nms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from tiny_yolo_qat import TinyYOLOQAT

# The class our model detects
VOC_CLASSES = ["aeroplane"]


def load_model(weights_path, quantized=False, device="cpu"):
    """
    Load the Tiny YOLO + QAT model.

    Args:
        weights_path : path to saved state_dict
        quantized    : True  → load INT8 weights (must run on CPU)
                       False → load float32 weights (CPU or GPU)
        device       : 'cpu' or 'cuda'
    """
    model = TinyYOLOQAT(grid_size=7, num_boxes=1, num_classes=1)

    if quantized:
        # 1. TEMPORARY FIX: Model MUST be in train mode to attach QAT observers
        model.train() 
        
        model.fuse_model()
        model.qconfig = quant.get_default_qat_qconfig("fbgemm")
        quant.prepare_qat(model, inplace=True)
        
        # 2. Convert to the final INT8 structure
        TinyYOLOQAT.convert_to_quantized(model)
        
        # 3. Load the pre-trained 8-bit weights
        model.load_state_dict(
            torch.load(weights_path, map_location="cpu", weights_only=True)
        )
        print("Loaded INT8 quantized model (CPU only).")
    else:
        model.load_state_dict(
            torch.load(weights_path, map_location=device, weights_only=True)
        )
        model.to(device)
        print(f"Loaded float32 model on {device}.")

    model.eval()
    return model


def decode_predictions(output, img_size=64, conf_thresh=0.3, S=7, C=1):
    """
    Decode raw YOLO grid tensor → lists of boxes, scores, and labels.

    Args:
        output      : (S, S, C + 5*B) tensor — single image, no batch dim
        img_size    : pixel size the model was trained on (64)
        conf_thresh : minimum objectness to keep
        S           : grid size
        C           : number of classes

    Returns:
        boxes  : list of [x1, y1, x2, y2] in pixel coords
        scores : list of float
        labels : list of str
    """
    boxes = []
    scores = []
    labels = []

    for i in range(S):
        for j in range(S):
            # Objectness confidence — apply sigmoid since output is raw logit
            conf = torch.sigmoid(output[i, j, C]).item()
            if conf < conf_thresh:
                continue

            # Class score — sigmoid on logit, multiply by objectness
            class_prob = torch.sigmoid(output[i, j, 0]).item()
            score = conf * class_prob
            if score < conf_thresh:
                continue

            # Box decoding
            x_cell = output[i, j, C + 1].item()
            y_cell = output[i, j, C + 2].item()
            w = output[i, j, C + 3].item()
            h = output[i, j, C + 4].item()

            # Convert cell-relative coords → absolute pixel coords
            x_center = (j + x_cell) / S * img_size
            y_center = (i + y_cell) / S * img_size
            box_w = abs(w) * img_size
            box_h = abs(h) * img_size

            x1 = max(0, x_center - box_w / 2)
            y1 = max(0, y_center - box_h / 2)
            x2 = min(img_size, x_center + box_w / 2)
            y2 = min(img_size, y_center + box_h / 2)

            # Skip degenerate boxes
            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            scores.append(score)
            labels.append(VOC_CLASSES[0])

    return boxes, scores, labels


def apply_nms(boxes, scores, labels, iou_thresh=0.3):
    """Filter overlapping detections with Non-Maximum Suppression."""
    if len(boxes) == 0:
        return [], [], []

    boxes_t = torch.tensor(boxes, dtype=torch.float32)
    scores_t = torch.tensor(scores, dtype=torch.float32)

    keep = nms(boxes_t, scores_t, iou_thresh)

    boxes_out = boxes_t[keep].tolist()
    scores_out = scores_t[keep].tolist()
    labels_out = [labels[k] for k in keep]

    return boxes_out, scores_out, labels_out


def visualize_prediction(
    image_path,
    weights_path="tiny_yolo_float_weights.pt",
    quantized=False,
    conf_thresh=0.3,
    iou_thresh=0.3,
):
    """
    Run Tiny YOLO + QAT inference on a single image and display results.
    """
    device = "cpu" if quantized else ("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load model
    model = load_model(weights_path, quantized=quantized, device=device)

    # 2. Prepare image
    raw_img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(raw_img).unsqueeze(0).to(device)  # (1, 3, 64, 64)

    # 3. Forward pass
    with torch.no_grad():
        output = model(img_tensor)  # (1, 7, 7, 6)

    output = output.squeeze(0).cpu()  # (7, 7, 6)

    # 4. Decode & NMS
    boxes, scores, labels = decode_predictions(
        output, img_size=64, conf_thresh=conf_thresh
    )
    boxes, scores, labels = apply_nms(boxes, scores, labels, iou_thresh=iou_thresh)

    # 5. Debug info
    max_conf = torch.sigmoid(output[..., 1]).max().item()
    print(f"Detections after NMS : {len(boxes)}")
    print(f"Max confidence (σ)   : {max_conf:.4f}")

    # 6. Plot
    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.imshow(img_tensor.squeeze(0).cpu().permute(1, 2, 0))

    for box, score, cls in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor="lime",
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x1,
            max(y1 - 2, 0),
            f"{cls} {score:.2f}",
            color="white",
            fontsize=8,
            fontweight="bold",
            bbox=dict(facecolor="green", alpha=0.6, pad=1),
        )

    plt.title("Tiny YOLO + QAT Inference (64×64)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # --- Float32 inference (GPU or CPU) ---
    test_image = "./data/VOCdevkit/VOC2012/JPEGImages/2007_000033.jpg"
    # visualize_prediction(
    #     test_image,
    #     weights_path="tiny_yolo_float_weights.pt",
    #     quantized=False,
    #     conf_thresh=0.3,
    # )

    # --- INT8 inference (CPU only) ---
    # Uncomment the lines below after you have trained and saved INT8 weights:
    visualize_prediction(
        test_image,
        weights_path="tiny_yolo_int8_weights.pt",
        quantized=True,
        conf_thresh=0.3,
    )