import torch
import torch.nn as nn
import torch.ao.quantization as quant


class ConvBnReLU(nn.Module):
    """A single fuse-able block: Conv2d → BatchNorm2d → ReLU."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class TinyYOLOQAT(nn.Module):
    """
    Tiny YOLO-style detector with Quantization-Aware Training (QAT) support.
    Designed for FPGA deployment at 64×64 input resolution.

    Grid layout is identical to the spiking model:
        output shape = (batch, S, S, C + 5*B)
    """

    def __init__(self, grid_size=7, num_boxes=1, num_classes=1):
        super().__init__()
        self.S = grid_size
        self.B = num_boxes
        self.C = num_classes
        self.output_dim = self.S * self.S * (self.C + 5 * self.B)

        # QAT stubs ─ these become real quantize/dequantize nodes after prepare_qat
        self.quant = quant.QuantStub()
        self.dequant = quant.DeQuantStub()

        # ── Feature extractor (5 conv blocks, each halving spatial dims) ──
        # 64→32→16→8→4→2
        self.features = nn.Sequential(
            # Block 1: 64×64 → 32×32
            ConvBnReLU(3, 16, 3, 1, 1),
            nn.MaxPool2d(2, 2),

            # Block 2: 32×32 → 16×16
            ConvBnReLU(16, 32, 3, 1, 1),
            nn.MaxPool2d(2, 2),

            # Block 3: 16×16 → 8×8
            ConvBnReLU(32, 64, 3, 1, 1),
            nn.MaxPool2d(2, 2),

            # Block 4: 8×8 → 4×4
            ConvBnReLU(64, 128, 3, 1, 1),
            nn.MaxPool2d(2, 2),

            # Block 5: 4×4 → 2×2
            ConvBnReLU(128, 256, 3, 1, 1),
            nn.MaxPool2d(2, 2),
        )

        # ── Detection head ──
        flat_size = 256 * 2 * 2  # 1024
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.output_dim),
        )

    # ------------------------------------------------------------------ #
    #  Forward pass (no temporal loop — pure feed-forward)                #
    # ------------------------------------------------------------------ #
    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = self.classifier(x)
        x = self.dequant(x)

        # Reshape to YOLO grid: (batch, S, S, C + 5*B)
        x = x.view(-1, self.S, self.S, self.C + 5 * self.B)
        return x

    # ------------------------------------------------------------------ #
    #  QAT helper methods                                                #
    # ------------------------------------------------------------------ #
    def fuse_model(self):
        """Fuse Conv+BN+ReLU inside every ConvBnReLU block for QAT."""
        for module in self.modules():
            if isinstance(module, ConvBnReLU):
                torch.ao.quantization.fuse_modules_qat(
                    module, ["conv", "bn", "act"], inplace=True
                )

    @staticmethod
    def prepare_qat(model):
        """Call after fuse_model() and before training begins."""
        model.qconfig = quant.get_default_qat_qconfig("fbgemm")
        quant.prepare_qat(model, inplace=True)
        return model

    @staticmethod
    def convert_to_quantized(model):
        """Call after training is complete to produce the final INT8 model."""
        model.eval()
        quant.convert(model, inplace=True)
        return model


# ────────────────────────────────────────────────────────────────────── #
#  Quick sanity check                                                   #
# ────────────────────────────────────────────────────────────────────── #
if __name__ == "__main__":
    print("=== Tiny YOLO + QAT — Architecture Test ===\n")

    model = TinyYOLOQAT(grid_size=7, num_boxes=1, num_classes=1)
    print(model)

    # Simulate a single forward pass with a dummy 64×64 image batch
    dummy = torch.randn(4, 3, 64, 64)
    out = model(dummy)
    print(f"\nInput  shape : {dummy.shape}")
    print(f"Output shape : {out.shape}")   # expect (4, 7, 7, 6)

    # Show QAT lifecycle
    print("\n--- Fusing Conv+BN+ReLU ---")
    model.fuse_model()

    print("--- Preparing QAT observers ---")
    TinyYOLOQAT.prepare_qat(model)

    # One calibration forward pass in train mode
    model.train()
    _ = model(dummy)

    print("--- Converting to INT8 ---")
    TinyYOLOQAT.convert_to_quantized(model)

    out_q = model(dummy)
    print(f"Quantized output shape : {out_q.shape}")
    print("\n✓ All checks passed.")