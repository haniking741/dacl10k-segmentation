"""
Multi-label Segmentation Metrics
"""

import torch


class MultiLabelSegmentationMetrics:
    """Multi-label semantic segmentation metrics"""

    def __init__(
        self,
        num_classes: int,
        class_names=None,
        threshold: float = 0.25,
        eps: float = 1e-7,
        ignore_empty: bool = True,
    ):
        self.num_classes = int(num_classes)
        self.class_names = class_names if class_names is not None else [
            f"class_{i:02d}" for i in range(num_classes)
        ]
        self.threshold = float(threshold)
        self.eps = float(eps)
        self.ignore_empty = bool(ignore_empty)

        self.reset()

    def reset(self):
        device = torch.device("cpu")
        self.tp = torch.zeros(self.num_classes, dtype=torch.float64, device=device)
        self.fp = torch.zeros(self.num_classes, dtype=torch.float64, device=device)
        self.fn = torch.zeros(self.num_classes, dtype=torch.float64, device=device)
        self.tn = torch.zeros(self.num_classes, dtype=torch.float64, device=device)
        self.gt_pos_pixels = torch.zeros(self.num_classes, dtype=torch.float64, device=device)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """Update confusion stats"""
        if logits.ndim != 4 or targets.ndim != 4:
            raise ValueError("logits and targets must be [N,C,H,W]")

        logits = logits.detach().float().cpu()
        targets = targets.detach().float().cpu()

        probs = torch.sigmoid(logits)
        preds = (probs >= self.threshold).to(torch.uint8)
        t = (targets >= 0.5).to(torch.uint8)

        C = preds.shape[1]
        preds_f = preds.permute(1, 0, 2, 3).contiguous().view(C, -1)
        t_f = t.permute(1, 0, 2, 3).contiguous().view(C, -1)

        tp = (preds_f & t_f).sum(dim=1).to(torch.float64)
        fp = (preds_f & (1 - t_f)).sum(dim=1).to(torch.float64)
        fn = ((1 - preds_f) & t_f).sum(dim=1).to(torch.float64)
        tn = ((1 - preds_f) & (1 - t_f)).sum(dim=1).to(torch.float64)

        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += tn
        self.gt_pos_pixels += t_f.sum(dim=1).to(torch.float64)

    def _safe_div(self, num, den):
        return num / (den + self.eps)

    def get_metrics(self):
        """Return metrics dictionary"""
        tp = self.tp
        fp = self.fp
        fn = self.fn

        precision = self._safe_div(tp, tp + fp)
        recall = self._safe_div(tp, tp + fn)
        f1 = self._safe_div(2 * tp, 2 * tp + fp + fn)
        iou = self._safe_div(tp, tp + fp + fn)

        if self.ignore_empty:
            valid = (self.gt_pos_pixels > 0)
        else:
            valid = torch.ones_like(self.gt_pos_pixels, dtype=torch.bool)

        if valid.sum().item() == 0:
            mean_precision = 0.0
            mean_recall = 0.0
            mean_f1 = 0.0
            mean_iou = 0.0
        else:
            mean_precision = precision[valid].mean().item()
            mean_recall = recall[valid].mean().item()
            mean_f1 = f1[valid].mean().item()
            mean_iou = iou[valid].mean().item()

        per_class = []
        for i in range(self.num_classes):
            per_class.append({
                "id": i,
                "name": self.class_names[i] if i < len(self.class_names) else f"class_{i:02d}",
                "IoU": float(iou[i].item()),
                "F1": float(f1[i].item()),
                "Precision": float(precision[i].item()),
                "Recall": float(recall[i].item()),
                "TP": int(tp[i].item()),
                "FP": int(fp[i].item()),
                "FN": int(fn[i].item()),
                "GT_pos_pixels": int(self.gt_pos_pixels[i].item()),
                "valid": bool(valid[i].item()),
            })

        return {
            "mean_IoU": mean_iou,
            "mean_F1": mean_f1,
            "mean_Precision": mean_precision,
            "mean_Recall": mean_recall,
            "threshold": self.threshold,
            "ignore_empty": self.ignore_empty,
            "per_class": per_class,
        }