"""
Multi-label Loss Functions
🔥 ULTIMATE FIX: Manual BCE computation - GUARANTEED to work!
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
    """
    Combined BCE + Dice Loss with class weights
    🔥 ULTIMATE FIX: Manual weighted BCE computation
    """
    def __init__(self, pos_weight=None, smooth: float = 1.0, w_bce: float = 1.0, w_dice: float = 1.0):
        super().__init__()
        self.w_bce = float(w_bce)
        self.w_dice = float(w_dice)
        self.dice = MultiLabelDiceLoss(smooth=smooth)

        # Store as Python list
        if pos_weight is not None:
            if isinstance(pos_weight, torch.Tensor):
                pos_weight = pos_weight.tolist()
            self.pos_weight_values = list(pos_weight)
        else:
            self.pos_weight_values = None
        
        print(f"✅ Loss initialized with pos_weight: {self.pos_weight_values}")

    def forward(self, logits, targets):
        """
        Args:
            logits: [B, C, H, W] - raw model outputs
            targets: [B, C, H, W] - binary ground truth masks
        """
        
        # 🔥 ULTIMATE FIX: Compute weighted BCE MANUALLY
        if self.pos_weight_values is not None:
            # Create pos_weight: [C] -> [1, C, 1, 1] for broadcasting
            pos_weight = torch.tensor(
                self.pos_weight_values,
                dtype=logits.dtype,
                device=logits.device
            ).view(1, -1, 1, 1)  # Shape: [1, C, 1, 1]
            
            # Manual BCE computation with weighting
            # BCE formula: -[y*w*log(σ(x)) + (1-y)*log(1-σ(x))]
            sigmoid = torch.sigmoid(logits)
            
            # Weighted BCE (positive samples get pos_weight, negative samples get weight 1.0)
            bce = -(targets * pos_weight * torch.log(sigmoid + 1e-7) + 
                    (1 - targets) * torch.log(1 - sigmoid + 1e-7))
            
            bce = bce.mean()
            
        else:
            # Standard BCE without weighting
            bce = F.binary_cross_entropy_with_logits(
                logits, 
                targets,
                reduction='mean'
            )

        # Compute Dice loss
        d = self.dice(logits, targets)
        
        # Combined loss
        total_loss = self.w_bce * bce + self.w_dice * d
        
        return total_loss