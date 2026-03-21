"""
DeepLabV3+ for Multi-label Semantic Segmentation

FIXES APPLIED:
  [BUG #2] Auxiliary head output is no longer discarded.
           forward() now returns (main_out, aux_out) during training
           so train_multilabel.py can compute aux loss (weight 0.4)
           and feed proper gradient signal into the backbone.
  [BUG #4] Replaced deprecated pretrained=True with the new
           weights=DeepLabV3_ResNet50_Weights.DEFAULT API so
           ImageNet weights are actually loaded on every torchvision version.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ✅ FIX #4: new weights API (works on torchvision ≥ 0.13)
try:
    from torchvision.models.segmentation import (
        deeplabv3_resnet50,
        deeplabv3_resnet101,
        DeepLabV3_ResNet50_Weights,
        DeepLabV3_ResNet101_Weights,
    )
    NEW_WEIGHTS_API = True
except ImportError:
    # torchvision < 0.13 fallback
    from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101
    NEW_WEIGHTS_API = False


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ wrapper for multi-label segmentation.

    forward() returns:
      - Training  : (main_logits [B,C,H,W], aux_logits [B,C,H,W])
      - Inference : main_logits [B,C,H,W]   (call model.eval() first)

    This allows train_multilabel.py to add the auxiliary loss:
        loss = criterion(main) + 0.4 * criterion(aux)
    giving the ResNet backbone a much stronger gradient signal.
    """

    def __init__(self, n_classes=3, backbone='resnet50', pretrained=True):
        super().__init__()
        self.n_classes = n_classes

        # ─── Build backbone ───────────────────────────────────────────────
        if backbone == 'resnet50':
            if NEW_WEIGHTS_API:
                w = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
                self.model = deeplabv3_resnet50(weights=w)
            else:
                self.model = deeplabv3_resnet50(pretrained=pretrained, progress=True)

        elif backbone == 'resnet101':
            if NEW_WEIGHTS_API:
                w = DeepLabV3_ResNet101_Weights.DEFAULT if pretrained else None
                self.model = deeplabv3_resnet101(weights=w)
            else:
                self.model = deeplabv3_resnet101(pretrained=pretrained, progress=True)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # ─── Replace main classifier head ─────────────────────────────────
        # torchvision DeepLabV3 classifier is an ASPP module;
        # the final Conv2d is at index [4].
        in_channels = self.model.classifier[4].in_channels
        self.model.classifier[4] = nn.Conv2d(in_channels, n_classes, kernel_size=1)

        # ─── Replace auxiliary classifier head ────────────────────────────
        # ✅ FIX #2: keep aux_classifier alive and replace its head
        # so that we can actually use it for aux loss during training.
        if hasattr(self.model, 'aux_classifier') and self.model.aux_classifier is not None:
            aux_in = self.model.aux_classifier[4].in_channels
            self.model.aux_classifier[4] = nn.Conv2d(aux_in, n_classes, kernel_size=1)
            self._has_aux = True
        else:
            self._has_aux = False

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W]

        Returns (training mode):
            (main_logits, aux_logits)  — both [B, n_classes, H, W]

        Returns (eval mode):
            main_logits                — [B, n_classes, H, W]
        """
        input_shape = x.shape[-2:]

        output_dict = self.model(x)   # always a dict: {'out': ..., 'aux': ...}

        # ── Main output ───────────────────────────────────────────────────
        main = output_dict['out']
        if main.shape[-2:] != input_shape:
            main = F.interpolate(main, size=input_shape, mode='bilinear', align_corners=False)

        # ── Auxiliary output ──────────────────────────────────────────────
        # ✅ FIX #2: return aux during training instead of discarding it
        if self.training and self._has_aux and 'aux' in output_dict:
            aux = output_dict['aux']
            if aux.shape[-2:] != input_shape:
                aux = F.interpolate(aux, size=input_shape, mode='bilinear', align_corners=False)
            return main, aux

        # Eval mode: return only main logits (standard behaviour)
        return main


# ─────────────────────────────────────────────────────────────────────────────

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(model_type: str, n_classes: int, device):
    """
    Factory used by train_multilabel.py.

    model_type examples: 'deeplabv3_resnet50', 'deeplabv3_resnet101'
    """
    model_type = model_type.lower()

    if 'resnet101' in model_type:
        backbone = 'resnet101'
    else:
        backbone = 'resnet50'

    print(f"📐 Building DeepLabV3+ | backbone={backbone} | classes={n_classes}")

    model = DeepLabV3Plus(n_classes=n_classes, backbone=backbone, pretrained=True)
    n_params = count_parameters(model)
    print(f"📊 Trainable parameters: {n_params:,}  ({n_params / 1e6:.1f}M)")

    model = model.to(device)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== DeepLabV3+ self-test ===\n")

    model = get_model('deeplabv3_resnet50', n_classes=3, device='cpu')

    x = torch.randn(2, 3, 512, 512)
    print(f"Input : {x.shape}")

    # --- Training mode (returns tuple) ---
    model.train()
    with torch.no_grad():
        out = model(x)

    if isinstance(out, tuple):
        main, aux = out
        print(f"Train  → main: {main.shape}  aux: {aux.shape}")
        assert main.shape == torch.Size([2, 3, 512, 512]), "main shape wrong"
        assert aux.shape  == torch.Size([2, 3, 512, 512]), "aux shape wrong"
    else:
        print(f"Train  → {out.shape}  (no aux head)")

    # --- Eval mode (returns tensor) ---
    model.eval()
    with torch.no_grad():
        out = model(x)
    print(f"Eval   → {out.shape}")
    assert out.shape == torch.Size([2, 3, 512, 512]), "eval shape wrong"

    print("\n✅ All checks passed!")