# utils/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss for semantic segmentation.

    logits:  [N, C, H, W]
    targets: [N, H, W] with class ids in [0..C-1]
    """
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        # alpha can be scalar or per-class list/tensor
        if isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        elif isinstance(alpha, torch.Tensor):
            self.alpha = alpha.float()
        else:
            self.alpha = float(alpha)

    def forward(self, logits, targets):
        n, c, h, w = logits.shape

        ce = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            ignore_index=self.ignore_index if self.ignore_index is not None else -100
        )  # [N,H,W]

        pt = torch.exp(-ce)

        # Alpha weighting
        if isinstance(self.alpha, torch.Tensor):
            alpha = self.alpha.to(logits.device)
            # targets might include ignore_index, clamp only for gather safety
            targets_clamped = targets.clamp(0, c - 1)
            at = alpha.gather(0, targets_clamped.view(-1)).view(n, h, w)
        else:
            at = self.alpha

        focal = at * (1 - pt) ** self.gamma * ce

        # Remove ignore pixels manually
        if self.ignore_index is not None:
            valid = (targets != self.ignore_index)
            focal = focal[valid]

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        else:
            return focal


class DiceLoss(nn.Module):
    """
    DirectML-safe multi-class Dice Loss (NO one_hot, NO scatter).

    Works on:
      - CUDA
      - CPU
      - DirectML (AMD/Intel)

    logits:  [N, C, H, W]
    targets: [N, H, W]
    """
    def __init__(self, ignore_index=None, smooth=1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, targets):
        n, c, h, w = logits.shape
        probs = F.softmax(logits, dim=1)  # [N,C,H,W]

        # mask valid pixels if ignore_index set
        if self.ignore_index is not None:
            valid = (targets != self.ignore_index).float()  # [N,H,W]
        else:
            valid = torch.ones_like(targets, dtype=torch.float32)

        # compute dice per class (loop is fine; C=20)
        dice_sum = 0.0
        count = 0

        for cls in range(c):
            if self.ignore_index is not None and cls == self.ignore_index:
                continue

            p = probs[:, cls, :, :]                 # [N,H,W]
            t = (targets == cls).float()            # [N,H,W]

            # apply valid mask
            p = p * valid
            t = t * valid

            intersection = (p * t).sum()
            denom = p.sum() + t.sum()

            dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
            dice_sum += (1.0 - dice)
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=logits.device)

        return dice_sum / count


class CombinedLoss(nn.Module):
    """
    Combine two losses: w1*loss1 + w2*loss2
    Example: CombinedLoss(FocalLoss(...), DiceLoss(...), w1=1.0, w2=1.0)
    """
    def __init__(self, loss1: nn.Module, loss2: nn.Module, w1=1.0, w2=1.0):
        super().__init__()
        self.loss1 = loss1
        self.loss2 = loss2
        self.w1 = float(w1)
        self.w2 = float(w2)

    def forward(self, logits, targets):
        return self.w1 * self.loss1(logits, targets) + self.w2 * self.loss2(logits, targets)