"""
Multi-label Loss Functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLabelDiceLoss(nn.Module):
    """Multi-label Dice Loss"""
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = float(smooth)

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        
        probs = probs.contiguous().view(probs.size(0), probs.size(1), -1)
        targets = targets.contiguous().view(targets.size(0), targets.size(1), -1)

        intersection = (probs * targets).sum(dim=2)
        denom = probs.sum(dim=2) + targets.sum(dim=2)

        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        loss = 1.0 - dice

        return loss.mean()


class CombinedBCEDice(nn.Module):
    """Combined BCE + Dice Loss with class weights"""
    def __init__(self, pos_weight=None, smooth: float = 1.0, w_bce: float = 1.0, w_dice: float = 1.0):
        super().__init__()
        self.w_bce = float(w_bce)
        self.w_dice = float(w_dice)
        self.dice = MultiLabelDiceLoss(smooth=smooth)

        if pos_weight is not None:
            if not isinstance(pos_weight, torch.Tensor):
                pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
            self.register_buffer("pos_weight", pos_weight.float())
        else:
            self.pos_weight = None

    def forward(self, logits, targets):
        if self.pos_weight is not None:
            bce = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=self.pos_weight
            )
        else:
            bce = F.binary_cross_entropy_with_logits(logits, targets)

        d = self.dice(logits, targets)
        return self.w_bce * bce + self.w_dice * d