# utils/metrics_multilabel.py
import torch


class MultiLabelSegmentationMetrics:
    """
    Multi-label semantic segmentation metrics (one-vs-rest per class).

    Expected:
      logits: [N, C, H, W] (raw outputs from model)
      targets: [N, C, H, W] (binary masks 0/1 per class)

    We compute per-class:
      - IoU (Jaccard)
      - F1 (Dice)
      - Precision
      - Recall

    And also means across classes (excluding empty classes optionally).
    """

    def __init__(
        self,
        num_classes: int,
        class_names=None,
        threshold: float = 0.3,
        eps: float = 1e-7,
        ignore_empty: bool = True,
    ):
        self.num_classes = int(num_classes)
        self.class_names = class_names if class_names is not None else [f"class_{i:02d}" for i in range(num_classes)]
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

        # for reporting how many classes had any positive pixels in GT
        self.gt_pos_pixels = torch.zeros(self.num_classes, dtype=torch.float64, device=device)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Update confusion stats.
        """
        if logits.ndim != 4:
            raise ValueError(f"logits must be [N,C,H,W], got {tuple(logits.shape)}")
        if targets.ndim != 4:
            raise ValueError(f"targets must be [N,C,H,W], got {tuple(targets.shape)}")
        if logits.shape[:2] != targets.shape[:2] or logits.shape[2:] != targets.shape[2:]:
            raise ValueError(f"Shape mismatch: logits {tuple(logits.shape)} vs targets {tuple(targets.shape)}")

        # move to CPU for stable accumulation (also DirectML safe)
        logits = logits.detach().float().cpu()
        targets = targets.detach().float().cpu()

        # sigmoid -> probabilities
        probs = torch.sigmoid(logits)
        preds = (probs >= self.threshold).to(torch.uint8) # 0/1
        t = (targets >= 0.5).to(torch.uint8) # ensure binary 0/1

        # flatten per class: [C, N*H*W]
        C = preds.shape[1]
        preds_f = preds.permute(1, 0, 2, 3).contiguous().view(C, -1)
        t_f = t.permute(1, 0, 2, 3).contiguous().view(C, -1)

        # counts
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
        """
        Return dict with:
          - per_class: list of dicts
          - mean_IoU, mean_F1, mean_Precision, mean_Recall
        """
        tp = self.tp
        fp = self.fp
        fn = self.fn
        tn = self.tn

        precision = self._safe_div(tp, tp + fp)
        recall = self._safe_div(tp, tp + fn)
        f1 = self._safe_div(2 * tp, 2 * tp + fp + fn)
        iou = self._safe_div(tp, tp + fp + fn)

        # optionally ignore classes that have NO positives in GT across all seen batches
        if self.ignore_empty:
            valid = (self.gt_pos_pixels > 0)
        else:
            valid = torch.ones_like(self.gt_pos_pixels, dtype=torch.bool)

        # avoid empty valid set
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