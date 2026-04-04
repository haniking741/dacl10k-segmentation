"""
Multi-label Loss Functions with Focal Loss
FIXES:
  [BUG #1] Numerically stable BCE via F.binary_cross_entropy_with_logits + expand_as
  [BUG #2] pos_weight registered as buffer (auto device movement, safe with AMP)
  [BUG #3] Focal loss uses alpha_t: alpha for positives, (1-alpha) for negatives
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLabelDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs   = probs.contiguous().view(probs.size(0), probs.size(1), -1)
        targets = targets.contiguous().view(targets.size(0), targets.size(1), -1)
        intersection = (probs * targets).sum(dim=2)
        denom = probs.sum(dim=2) + targets.sum(dim=2)
        dice  = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        return (1.0 - dice).mean()


class FocalLoss(nn.Module):
    """
    Sigmoid Focal Loss.
    alpha   → weight for positive pixels  (foreground)
    1-alpha → weight for negative pixels  (background)
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        # Numerically stable per-pixel BCE (no manual sigmoid)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        probs = torch.sigmoid(logits)

        # p_t: probability assigned to the TRUE class
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)

        # FIX #3: alpha_t differs for positive vs negative pixels
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)

        focal = alpha_t * (1.0 - p_t) ** self.gamma * bce
        return focal.mean()


class CombinedLoss(nn.Module):
    def __init__(
        self,
        pos_weight=None,
        smooth: float = 1.0,
        w_bce: float = 1.0,
        w_dice: float = 1.0,
        w_focal: float = 0.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.w_bce  = w_bce
        self.w_dice = w_dice
        self.w_focal = w_focal

        self.dice  = MultiLabelDiceLoss(smooth=smooth)
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

        # FIX #2: register_buffer → auto moves to correct device (GPU/CPU/AMP safe)
        if pos_weight is not None:
            if isinstance(pos_weight, torch.Tensor):
                pos_weight = pos_weight.tolist()
            pw = torch.tensor(pos_weight, dtype=torch.float32).view(1, -1, 1, 1)
            self.register_buffer('pos_weight', pw)
        else:
            self.pos_weight = None

        print(f"✅ CombinedLoss: BCE={w_bce}, Dice={w_dice}, Focal={w_focal}")
        print(f"   pos_weight = {pos_weight}")

    def forward(self, logits, targets):
        # FIX #1: numerically stable BCE, pos_weight broadcast over [N,C,H,W]
        if self.pos_weight is not None:
            pw  = self.pos_weight.expand_as(logits)          # [1,C,1,1] → [N,C,H,W]
            bce = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pw, reduction='mean'
            )
        else:
            bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='mean')

        d = self.dice(logits, targets)
        f = self.focal(logits, targets)

        return self.w_bce * bce + self.w_dice * d + self.w_focal * f
