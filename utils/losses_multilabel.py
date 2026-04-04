"""
Multi-label Loss Functions with Focal Loss
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
        probs = probs.contiguous().view(probs.size(0), probs.size(1), -1)
        targets = targets.contiguous().view(targets.size(0), targets.size(1), -1)
        intersection = (probs * targets).sum(dim=2)
        denom = probs.sum(dim=2) + targets.sum(dim=2)
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        loss = 1.0 - dice
        return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = torch.sigmoid(logits) * targets + (1 - torch.sigmoid(logits)) * (1 - targets)
        focal = self.alpha * (1 - p_t) ** self.gamma * bce
        return focal.mean()


class CombinedLoss(nn.Module):
    def __init__(self, pos_weight=None, smooth=1.0, w_bce=1.0, w_dice=1.0, w_focal=0.5, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.w_bce = w_bce
        self.w_dice = w_dice
        self.w_focal = w_focal
        self.dice = MultiLabelDiceLoss(smooth=smooth)
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

        if pos_weight is not None:
            if isinstance(pos_weight, torch.Tensor):
                pos_weight = pos_weight.tolist()
            self.pos_weight_values = list(pos_weight)
        else:
            self.pos_weight_values = None

        print(f"✅ CombinedLoss: BCE weight={w_bce}, Dice weight={w_dice}, Focal weight={w_focal}")
        print(f"   pos_weight = {self.pos_weight_values}")

    def forward(self, logits, targets):
        if self.pos_weight_values is not None:
            pos_weight = torch.tensor(self.pos_weight_values, dtype=logits.dtype, device=logits.device).view(1, -1, 1, 1)
            sigmoid = torch.sigmoid(logits)
            bce = -(targets * pos_weight * torch.log(sigmoid + 1e-7) + (1 - targets) * torch.log(1 - sigmoid + 1e-7)).mean()
        else:
            bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='mean')

        d = self.dice(logits, targets)
        f = self.focal(logits, targets)
        total = self.w_bce * bce + self.w_dice * d + self.w_focal * f
        return total