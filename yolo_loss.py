import torch
import torch.nn as nn

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20, lambda_coord=5, lambda_noobj=0.5):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        
        self.S = S
        self.B = B
        self.C = C
        
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, predictions, target):
        """
        predictions: (N, S, S, C + 5B)
        target:      (N, S, S, C + 5B)
        """

        N = predictions.size(0)

        # Reshape
        predictions = predictions.reshape(N, self.S, self.S, self.C + self.B * 5)
        target = target.reshape(N, self.S, self.S, self.C + self.B * 5)

        # Object mask
        obj_mask = target[..., self.C].unsqueeze(3)  # confidence of first box

        # ----------------------------
        # BOX COORD LOSS
        # ----------------------------
        box_pred = predictions[..., self.C:self.C+4]
        box_target = target[..., self.C:self.C+4]

        # sqrt on width & height (YOLO trick)
        box_pred[..., 2:4] = torch.sign(box_pred[..., 2:4]) * torch.sqrt(
            torch.abs(box_pred[..., 2:4]) + 1e-6
        )
        box_target[..., 2:4] = torch.sqrt(box_target[..., 2:4])

        coord_loss = self.mse(
            obj_mask * box_pred,
            obj_mask * box_target
        )

        # ----------------------------
        # OBJECT CONFIDENCE LOSS
        # ----------------------------
        conf_pred = predictions[..., self.C+4:self.C+5]
        conf_target = target[..., self.C+4:self.C+5]

        obj_loss = self.mse(
            obj_mask * conf_pred,
            obj_mask * conf_target
        )

        noobj_loss = self.mse(
            (1 - obj_mask) * conf_pred,
            (1 - obj_mask) * conf_target
        )

        # ----------------------------
        # CLASS LOSS
        # ----------------------------
        class_pred = predictions[..., :self.C]
        class_target = target[..., :self.C]

        class_loss = self.mse(
            obj_mask * class_pred,
            obj_mask * class_target
        )

        loss = (
            self.lambda_coord * coord_loss
            + obj_loss
            + self.lambda_noobj * noobj_loss
            + class_loss
        )

        return loss / N
