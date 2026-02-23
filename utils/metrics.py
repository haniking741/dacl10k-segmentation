import torch
import numpy as np

class SegmentationMetrics:
    def __init__(self, num_classes, class_names=None, ignore_index=None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, preds, targets):
        """
        preds, targets: [B,H,W] integer class ids
        """
        if torch.is_tensor(preds):
            preds = preds.detach().cpu().numpy()
        if torch.is_tensor(targets):
            targets = targets.detach().cpu().numpy()

        preds = preds.astype(np.int64).reshape(-1)
        targets = targets.astype(np.int64).reshape(-1)

        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            preds = preds[mask]
            targets = targets[mask]

        # keep only valid range
        valid = (targets >= 0) & (targets < self.num_classes) & (preds >= 0) & (preds < self.num_classes)
        preds = preds[valid]
        targets = targets[valid]

        # fast confusion matrix via bincount
        idx = targets * self.num_classes + preds
        cm = np.bincount(idx, minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)
        self.confusion_matrix += cm

    def get_metrics(self, return_per_class=False):
        cm = self.confusion_matrix.astype(np.float64)

        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp

        iou = tp / (tp + fp + fn + 1e-10)
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        valid_classes = (tp + fn) > 0  # only classes that appear in GT
        miou = iou[valid_classes].mean() if np.any(valid_classes) else 0.0
        mean_f1 = f1[valid_classes].mean() if np.any(valid_classes) else 0.0
        mean_precision = precision[valid_classes].mean() if np.any(valid_classes) else 0.0
        mean_recall = recall[valid_classes].mean() if np.any(valid_classes) else 0.0

        pixel_acc = tp.sum() / (cm.sum() + 1e-10)

        metrics = {
            "mIoU": float(miou),
            "mean_F1": float(mean_f1),
            "mean_Precision": float(mean_precision),
            "mean_Recall": float(mean_recall),
            "Pixel_Accuracy": float(pixel_acc),
        }

        if return_per_class:
            metrics["per_class"] = {
                "IoU": dict(zip(self.class_names, iou)),
                "F1": dict(zip(self.class_names, f1)),
                "Precision": dict(zip(self.class_names, precision)),
                "Recall": dict(zip(self.class_names, recall)),
            }

        return metrics