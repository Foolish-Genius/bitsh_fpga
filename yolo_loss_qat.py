import torch
import torch.nn as nn


class YoloLossQAT(nn.Module):
    """
    YOLO loss function tailored for Tiny YOLO + QAT.

    Tensor layout per grid cell (C=1, B=1):
        index 0        : class probability  (aeroplane)
        index 1        : objectness / confidence
        index 2        : x_cell
        index 3        : y_cell
        index 4        : width
        index 5        : height
    """

    def __init__(self, S=7, B=1, C=1):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C

        # Coordinate loss weight (emphasise box accuracy)
        self.lambda_coord = 5.0
        # No-object confidence weight (suppress false positives)
        self.lambda_noobj = 0.4

        self.mse = nn.MSELoss(reduction="mean")
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, predictions, target):
        """
        Args:
            predictions : (batch, S, S, C + 5*B)  — raw (logit) output from the network
            target      : (batch, S, S, C + 5*B)  — encoded ground truth
        """
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        # Binary mask: 1 where an object exists, 0 otherwise
        # Shape: (batch, S, S, 1)
        obj_mask = target[..., self.C].unsqueeze(-1)
        noobj_mask = 1.0 - obj_mask

        # ─────────────────────────────────────────────
        # PART 1 — Bounding-box coordinate loss (MSE)
        # ─────────────────────────────────────────────
        # x_cell, y_cell (indices C+1, C+2)
        pred_xy = obj_mask * predictions[..., self.C + 1 : self.C + 3]
        targ_xy = obj_mask * target[..., self.C + 1 : self.C + 3]

        xy_loss = self.mse(
            pred_xy.flatten(),
            targ_xy.flatten(),
        )

        # width, height (indices C+3, C+4)
        # Use sqrt-space loss (original YOLO trick) so that small boxes
        # are penalised more heavily than large ones.
        pred_wh = obj_mask * predictions[..., self.C + 3 : self.C + 5]
        targ_wh = obj_mask * target[..., self.C + 3 : self.C + 5]

        # Stabilise before sqrt
        pred_wh_safe = torch.sign(pred_wh) * torch.sqrt(torch.abs(pred_wh) + 1e-6)
        targ_wh_safe = torch.sqrt(targ_wh + 1e-6)

        wh_loss = self.mse(
            pred_wh_safe.flatten(),
            targ_wh_safe.flatten(),
        )

        box_loss = xy_loss + wh_loss

        # ─────────────────────────────────────────────
        # PART 2 — Object confidence loss (BCE)
        # ─────────────────────────────────────────────
        pred_conf = predictions[..., self.C : self.C + 1]
        targ_conf = target[..., self.C : self.C + 1]

        obj_conf_loss = self.bce(
            (obj_mask * pred_conf).flatten(),
            (obj_mask * targ_conf).flatten(),
        )

        # ─────────────────────────────────────────────
        # PART 3 — No-object confidence loss (BCE)
        # ─────────────────────────────────────────────
        noobj_conf_loss = self.bce(
            (noobj_mask * pred_conf).flatten(),
            (noobj_mask * targ_conf).flatten(),
        )

        # ─────────────────────────────────────────────
        # PART 4 — Class prediction loss (BCE)
        # ─────────────────────────────────────────────
        pred_class = obj_mask * predictions[..., :self.C]
        targ_class = obj_mask * target[..., :self.C]

        class_loss = self.bce(
            pred_class.flatten(),
            targ_class.flatten(),
        )

        # ─────────────────────────────────────────────
        # TOTAL
        # ─────────────────────────────────────────────
        total_loss = (
            self.lambda_coord * box_loss
            + obj_conf_loss
            + self.lambda_noobj * noobj_conf_loss
            + class_loss
        )

        return total_loss


# ────────────────────────────────────────────────────────────────────── #
#  Quick sanity check                                                   #
# ────────────────────────────────────────────────────────────────────── #
if __name__ == "__main__":
    loss_fn = YoloLossQAT(S=7, B=1, C=1)

    # Fake predictions (raw logits) and target
    preds = torch.randn(4, 7, 7, 6)
    target = torch.zeros(4, 7, 7, 6)

    # Place a dummy object in cell (3, 3) for every sample
    target[:, 3, 3, 0] = 1.0   # class = aeroplane
    target[:, 3, 3, 1] = 1.0   # confidence
    target[:, 3, 3, 2] = 0.5   # x_cell
    target[:, 3, 3, 3] = 0.5   # y_cell
    target[:, 3, 3, 4] = 0.3   # width
    target[:, 3, 3, 5] = 0.2   # height

    loss = loss_fn(preds, target)
    print(f"Test loss: {loss.item():.4f}")
    print("✓ Loss function OK.")
