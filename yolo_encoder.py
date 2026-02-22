import torch

# VOC class list (20 classes)
VOC_CLASSES = [
    "aeroplane","bicycle","bird","boat","bottle","bus","car","cat",
    "chair","cow","diningtable","dog","horse","motorbike","person",
    "pottedplant","sheep","sofa","train","tvmonitor"
]

class YOLOEncoder:
    def __init__(self, S=7, B=2, C=20):
        self.S = S
        self.B = B
        self.C = C

    def encode(self, annotation, img_size=64):
        """
        Converts VOC annotation dict → YOLO tensor
        Output shape: (S, S, C + 5B)
        """
        target = torch.zeros((self.S, self.S, self.C + self.B * 5))

        objects = annotation["annotation"].get("object", [])

        # If only one object, wrap in list
        if isinstance(objects, dict):
            objects = [objects]

        for obj in objects:
            class_name = obj["name"]
            if class_name not in VOC_CLASSES:
                continue

            class_idx = VOC_CLASSES.index(class_name)

            bbox = obj["bndbox"]

            xmin = float(bbox["xmin"])
            ymin = float(bbox["ymin"])
            xmax = float(bbox["xmax"])
            ymax = float(bbox["ymax"])

            # Normalize (because we resized to 64x64)
            xmin /= img_size
            ymin /= img_size
            xmax /= img_size
            ymax /= img_size

            # Convert to YOLO format
            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2
            width = xmax - xmin
            height = ymax - ymin

            # Determine grid cell
            i = int(self.S * y_center)
            j = int(self.S * x_center)

            # Clamp to grid
            i = min(self.S - 1, i)
            j = min(self.S - 1, j)

            # Coordinates relative to grid cell
            x_cell = self.S * x_center - j
            y_cell = self.S * y_center - i

            # If no object already assigned to this cell
            if target[i, j, self.C] == 0:
                # One-hot class (Indices 0 to 19)
                target[i, j, class_idx] = 1

                # Confidence Score (Index 20)
                target[i, j, self.C] = 1.0 
                
                # Bounding Box: x_cell, y_cell, width, height (Indices 21 to 24)
                target[i, j, self.C+1 : self.C+5] = torch.tensor([
                    x_cell, y_cell, width, height
                ])

        return target
