# utils/losses.py

import torch.nn as nn


def get_loss(ignore_index=None, class_weights=None):
    """
    Single-label multi-class loss for semantic segmentation.

    logits:  [N, C, H, W]
    targets: [N, H, W]  (values 0..C-1)

    Uses CrossEntropyLoss.
    """

    if class_weights is not None:
        return nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index
        )

    return nn.CrossEntropyLoss(
        ignore_index=ignore_index
    )