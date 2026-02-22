import torch
import torch.nn as nn

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=1, C=1):
        super().__init__()
        # FIX: Using 'mean' instead of 'sum' prevents the 90,000+ loss explosion
        self.mse = nn.MSELoss(reduction="mean")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = 5.0
        self.lambda_noobj = 0.5

    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)
        exists_box = target[..., self.C].unsqueeze(3) 

        # --- PART 1: BOUNDING BOX COORDINATES ---
        box_targets = exists_box * target[..., self.C+1:self.C+5]
        box_predictions = exists_box * predictions[...,self.C+1:self.C+5]
        
        pred_xy = box_predictions[..., 0:2]
        pred_wh = box_predictions[..., 2:4]
        # STABILIZER: Force a minimum size and use absolute values to prevent NaN
        # This prevents the "tiny line" box problem
        pred_wh_fixed = torch.abs(pred_wh) + 1e-6 
        box_predictions_fixed = torch.cat([pred_xy, pred_wh_fixed], dim=-1)

        target_xy = box_targets[..., 0:2]
        target_wh = box_targets[..., 2:4] 
        box_targets_fixed = torch.cat([target_xy, target_wh], dim=-1)
        
        box_loss = self.mse(
            torch.flatten(box_predictions_fixed, end_dim=-2),
            torch.flatten(box_targets_fixed, end_dim=-2)
        )

        # --- PART 2: OBJECT CONFIDENCE ---
        pred_box_confidence = exists_box * predictions[..., self.C:self.C+1]
        target_box_confidence = exists_box * target[..., self.C:self.C+1]
        object_loss = self.mse(
            torch.flatten(pred_box_confidence),
            torch.flatten(target_box_confidence)
        )

        # --- PART 3: NO-OBJECT CONFIDENCE ---
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., self.C:self.C+1], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., self.C:self.C+1], start_dim=1)
        )

        # --- PART 4: CLASSES (Back to pure stable MSE) ---
        class_pred = exists_box * predictions[..., :self.C]
        class_target = exists_box * target[..., :self.C]
        
        class_loss = self.mse(
            torch.flatten(class_pred, end_dim=-2),
            torch.flatten(class_target, end_dim=-2)
        )

        # Combine
        total_loss = (self.lambda_coord * box_loss) + object_loss + (self.lambda_noobj * no_object_loss) + class_loss
        return total_loss

        # Combine
        total_loss = (self.lambda_coord * box_loss) + object_loss + (self.lambda_noobj * no_object_loss) + class_loss
        return total_loss
