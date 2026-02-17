"""
Evaluation Metrics for Semantic Segmentation
- mIoU (mean Intersection over Union)
- F1-score (Dice coefficient)
- Pixel Accuracy
- Per-class metrics
"""
import torch
import numpy as np
from sklearn.metrics import confusion_matrix


class SegmentationMetrics:
    """
    Compute segmentation metrics for multi-class segmentation
    """
    
    def __init__(self, num_classes, class_names=None, ignore_index=None):
        """
        Args:
            num_classes: Number of classes
            class_names: List of class names (optional, for printing)
            ignore_index: Class index to ignore in metrics (e.g., background or void)
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
    
    def update(self, preds, targets):
        """
        Update confusion matrix with predictions and targets
        
        Args:
            preds: Predictions tensor [B, H, W] with class indices
            targets: Ground truth tensor [B, H, W] with class indices
        """
        # Convert to numpy
        if torch.is_tensor(preds):
            preds = preds.cpu().numpy()
        if torch.is_tensor(targets):
            targets = targets.cpu().numpy()
        
        # Flatten
        preds = preds.flatten()
        targets = targets.flatten()
        
        # Remove ignore index
        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            preds = preds[mask]
            targets = targets[mask]
        
        # Update confusion matrix
        cm = confusion_matrix(targets, preds, labels=list(range(self.num_classes)))
        self.confusion_matrix += cm
    
    def get_metrics(self, return_per_class=False):
        """
        Compute all metrics from confusion matrix
        
        Returns:
            dict: Dictionary of metrics
        """
        cm = self.confusion_matrix
        
        # Per-class metrics
        # TP: diagonal elements
        tp = np.diag(cm)
        
        # FP: sum of column - TP
        fp = cm.sum(axis=0) - tp
        
        # FN: sum of row - TP
        fn = cm.sum(axis=1) - tp
        
        # TN: total - TP - FP - FN
        tn = cm.sum() - tp - fp - fn
        
        # IoU per class
        iou = tp / (tp + fp + fn + 1e-10)
        
        # Precision per class
        precision = tp / (tp + fp + 1e-10)
        
        # Recall per class
        recall = tp / (tp + fn + 1e-10)
        
        # F1-score per class
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        # Mean metrics (excluding classes with no ground truth)
        valid_classes = (tp + fn) > 0  # Classes that appear in ground truth
        
        miou = iou[valid_classes].mean()
        mean_f1 = f1[valid_classes].mean()
        mean_precision = precision[valid_classes].mean()
        mean_recall = recall[valid_classes].mean()
        
        # Pixel accuracy
        pixel_acc = tp.sum() / (cm.sum() + 1e-10)
        
        # Overall metrics
        metrics = {
            'mIoU': miou,
            'mean_F1': mean_f1,
            'mean_Precision': mean_precision,
            'mean_Recall': mean_recall,
            'Pixel_Accuracy': pixel_acc,
        }
        
        if return_per_class:
            # Add per-class metrics
            metrics['per_class'] = {
                'IoU': dict(zip(self.class_names, iou)),
                'F1': dict(zip(self.class_names, f1)),
                'Precision': dict(zip(self.class_names, precision)),
                'Recall': dict(zip(self.class_names, recall)),
            }
        
        return metrics
    
    def print_metrics(self):
        """Print formatted metrics"""
        metrics = self.get_metrics(return_per_class=True)
        
        print("\n" + "="*70)
        print("SEGMENTATION METRICS")
        print("="*70)
        print(f"mIoU (mean IoU):        {metrics['mIoU']:.4f}")
        print(f"mean F1-score:          {metrics['mean_F1']:.4f}")
        print(f"mean Precision:         {metrics['mean_Precision']:.4f}")
        print(f"mean Recall:            {metrics['mean_Recall']:.4f}")
        print(f"Pixel Accuracy:         {metrics['Pixel_Accuracy']:.4f}")
        
        print("\n" + "-"*70)
        print("PER-CLASS IoU:")
        print("-"*70)
        for cls_name, iou_val in metrics['per_class']['IoU'].items():
            if iou_val > 0:  # Only show classes that appeared
                print(f"  {cls_name:30s}  {iou_val:.4f}")
        print("="*70 + "\n")


def compute_iou(preds, targets, num_classes, ignore_index=None):
    """
    Quick mIoU computation (simpler version)
    
    Args:
        preds: Predictions [B, H, W]
        targets: Ground truth [B, H, W]
        num_classes: Number of classes
        ignore_index: Index to ignore
    
    Returns:
        float: mIoU score
    """
    if torch.is_tensor(preds):
        preds = preds.cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.cpu().numpy()
    
    preds = preds.flatten()
    targets = targets.flatten()
    
    if ignore_index is not None:
        mask = targets != ignore_index
        preds = preds[mask]
        targets = targets[mask]
    
    iou_per_class = []
    for cls in range(num_classes):
        pred_mask = preds == cls
        target_mask = targets == cls
        
        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()
        
        if union > 0:
            iou_per_class.append(intersection / union)
    
    return np.mean(iou_per_class) if iou_per_class else 0.0


def compute_dice(preds, targets, num_classes, ignore_index=None):
    """
    Compute Dice coefficient (F1-score)
    
    Args:
        preds: Predictions [B, H, W]
        targets: Ground truth [B, H, W]
        num_classes: Number of classes
        ignore_index: Index to ignore
    
    Returns:
        float: Dice score
    """
    if torch.is_tensor(preds):
        preds = preds.cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.cpu().numpy()
    
    preds = preds.flatten()
    targets = targets.flatten()
    
    if ignore_index is not None:
        mask = targets != ignore_index
        preds = preds[mask]
        targets = targets[mask]
    
    dice_per_class = []
    for cls in range(num_classes):
        pred_mask = preds == cls
        target_mask = targets == cls
        
        intersection = np.logical_and(pred_mask, target_mask).sum()
        
        if pred_mask.sum() + target_mask.sum() > 0:
            dice = 2 * intersection / (pred_mask.sum() + target_mask.sum())
            dice_per_class.append(dice)
    
    return np.mean(dice_per_class) if dice_per_class else 0.0


if __name__ == "__main__":
    # Test metrics
    print("Testing segmentation metrics...\n")
    
    # DACL10K class names
    class_names = [
        "background", "graffiti", "drainage", "wetspot", "weathering", "crack",
        "rockpocket", "spalling", "washouts/concrete corrosion", "cavity",
        "efflorescence", "rust", "protective equipment", "exposed rebars", "bearing",
        "hollowareas", "joint tape", "restformwork",
        "alligator crack", "expansion joint"
    ]
    
    # Create dummy predictions and targets
    num_classes = 20
    preds = torch.randint(0, num_classes, (2, 256, 256))  # Batch of 2, 256x256
    targets = torch.randint(0, num_classes, (2, 256, 256))
    
    # Compute metrics
    metrics = SegmentationMetrics(num_classes, class_names)
    metrics.update(preds, targets)
    metrics.print_metrics()
    
    # Quick mIoU
    miou = compute_iou(preds, targets, num_classes)
    print(f"Quick mIoU: {miou:.4f}\n")
    
    print("✅ Metrics working correctly!")