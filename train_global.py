#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  DACL10K  Multi-Label  Semantic  Segmentation  –  FINAL  PIPELINE          ║
║                                                                            ║
║  Self-contained: dataset · augmentation · model · loss · metrics           ║
║                  training · validation (TTA) · test / inference            ║
║                                                                            ║
║  Target:  3 defect classes  →  crack (1)  ·  spalling (7)  ·  rust (11)   ║
║  Model :  DeepLabV3+  ResNet-101  (aux head weight 0.4)                    ║
║  Loss  :  BCE (label-smooth 0.05) + Dice + Focal   (combined)             ║
║  Sched :  OneCycleLR                                                       ║
║  AMP   :  ✔  (CUDA only)                                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os, sys, time, random, math, json, warnings
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms.functional as TF

# AMP imports (PyTorch ≥ 2.0 preferred)
try:
    from torch.amp import autocast, GradScaler
    _NEW_AMP = True
except ImportError:
    from torch.cuda.amp import autocast, GradScaler  # type: ignore
    _NEW_AMP = False

from tqdm import tqdm

# Optional albumentations (for stronger spatial augments)
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUM = True
except ImportError:
    HAS_ALBUM = False
    print("⚠️  albumentations not installed – GridDistortion / ElasticTransform disabled")

# torchvision model weights API
try:
    from torchvision.models.segmentation import (
        deeplabv3_resnet50,
        deeplabv3_resnet101,
        DeepLabV3_ResNet50_Weights,
        DeepLabV3_ResNet101_Weights,
    )
    _NEW_WEIGHTS = True
except ImportError:
    from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101
    _NEW_WEIGHTS = False

warnings.filterwarnings("ignore", category=UserWarning)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CONFIG  (edit these paths / hyper-params for your environment)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONFIG = {
    # ── Paths ──────────────────────────────────────────────────────────────
    "DATA_ROOT":       "/kaggle/input/datasets/hanihafnaoui/hani-dataset/dataset2",
    "IMAGES_SUBDIR":   "images",
    "MASKS_SUBDIR":    "masks_multilabel",
    "SAVE_DIR":        "/kaggle/working/checkpoints_final",
    "LOG_DIR":         "/kaggle/working/logs_final",
    "TEST_IMG_DIR":    "/kaggle/input/datasets/hanihafnaoui/hani-dataset/dataset2/images/val",
    "TEST_OVERLAY_DIR":"/kaggle/working/test_overlays",

    # ── Dataset ────────────────────────────────────────────────────────────
    "CLASSES_TO_LOAD": [1, 7, 11],             # crack, spalling, rust
    "CLASS_NAMES":     ["crack", "spalling", "rust"],
    "NUM_LABELS":      3,

    # ── Model ──────────────────────────────────────────────────────────────
    "BACKBONE":        "resnet101",             # upgraded from resnet50
    "AUX_LOSS_WEIGHT": 0.4,

    # ── Training ───────────────────────────────────────────────────────────
    "IMG_SIZE":        (512, 512),
    "BATCH_SIZE":      8,
    "NUM_WORKERS":     4,
    "NUM_EPOCHS":      50,
    "RANDOM_SEED":     42,

    # ── Optimizer / Scheduler ──────────────────────────────────────────────
    "LEARNING_RATE":   1e-4,
    "MAX_LR":          6e-4,                    # OneCycleLR peak
    "WEIGHT_DECAY":    1e-4,

    # ── Loss ───────────────────────────────────────────────────────────────
    "BCE_POS_WEIGHT":  [9.67, 6.16, 6.95],     # crack, spalling, rust
    "DICE_SMOOTH":     1.0,
    "LABEL_SMOOTHING": 0.05,
    "FOCAL_ALPHA":     0.25,
    "FOCAL_GAMMA":     2.0,
    "W_BCE":           1.0,
    "W_DICE":          1.0,
    "W_FOCAL":         0.5,

    # ── Augmentation ───────────────────────────────────────────────────────
    "DEFECT_CROP_PROB":   0.7,
    "CROP_RATIO":         0.60,
    "CROP_TRIES":         10,
    "MIN_DEFECT_RATIO":   0.01,
    "COLOR_JITTER":       True,
    "RANDOM_BLUR":        True,
    "RANDOM_NOISE":       True,
    "NOISE_STD":          0.02,
    "NOISE_PROB":         0.2,
    "BLUR_PROB":          0.3,
    "USE_ALBUM_SPATIAL":  True,   # GridDistortion + ElasticTransform

    # ── Validation / Inference ─────────────────────────────────────────────
    "THRESHOLD":       0.25,
    "USE_TTA":         True,
    "EARLY_STOP_PAT":  5,

    # ── AMP ────────────────────────────────────────────────────────────────
    "USE_AMP":         True,

    # ── Overlay colours (R, G, B) ──────────────────────────────────────────
    "CLASS_COLORS": {
        "crack":    (255,   0,   0),   # red
        "spalling": (  0,   0, 255),   # blue
        "rust":     (255, 255,   0),   # yellow
    },
    "OVERLAY_ALPHA": 0.45,
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  1.  DATASET
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DACL10KMultiLabel(Dataset):
    """Multi-label dataset: one binary mask per class."""

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        classes_to_load: List[int],
        img_size: Tuple[int, int] = (512, 512),
        is_train: bool = True,
        cfg: dict = CONFIG,
    ):
        self.img_dir  = img_dir
        self.mask_dir = mask_dir
        self.classes  = list(classes_to_load)
        self.C        = len(self.classes)
        self.img_size = tuple(img_size)
        self.train    = is_train
        self.cfg      = cfg

        self.images = sorted(
            f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png"))
        )
        print(f"  📂 {'Train' if is_train else 'Val'}: {len(self.images)} images  "
              f"({img_dir})")

        # Build albumentations spatial pipeline (train only)
        self.album_spatial = None
        if is_train and HAS_ALBUM and cfg.get("USE_ALBUM_SPATIAL", False):
            self.album_spatial = A.Compose([
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.4),
                A.ElasticTransform(alpha=80, sigma=80 * 0.05,
                                   p=0.3),
            ], additional_targets={f"mask{i}": "mask" for i in range(self.C)})

    def __len__(self):
        return len(self.images)

    # ── mask loading ──────────────────────────────────────────────────────
    def _load_masks(self, base: str, fallback_size: Tuple[int, int]):
        masks = []
        for cid in self.classes:
            fp = os.path.join(self.mask_dir, f"{base}_class{cid:02d}.png")
            if os.path.exists(fp):
                m = Image.open(fp).convert("L")
            else:
                m = Image.new("L", fallback_size, 0)
            masks.append(m)
        return masks

    # ── cropping helpers ──────────────────────────────────────────────────
    @staticmethod
    def _random_crop(img, masks, ch, cw):
        w, h = img.size
        if h <= ch or w <= cw:
            return img, masks
        top  = random.randint(0, h - ch)
        left = random.randint(0, w - cw)
        img  = TF.crop(img, top, left, ch, cw)
        masks = [TF.crop(m, top, left, ch, cw) for m in masks]
        return img, masks

    def _defect_crop(self, img, masks, ch, cw, tries=10, min_ratio=0.01):
        w, h = img.size
        if h <= ch or w <= cw:
            return img, masks

        # low-res union for speed
        low = (256, 256)
        union = np.zeros(low[::-1], dtype=np.uint8)
        for m in masks:
            ms = m.resize(low, resample=Image.NEAREST)
            union = np.maximum(union, np.array(ms, dtype=np.uint8))

        ys, xs = np.where(union > 0)
        if ys.size == 0:
            return self._random_crop(img, masks, ch, cw)

        for _ in range(tries):
            idx = random.randint(0, ys.size - 1)
            y = int(ys[idx] * (h / low[1]))
            x = int(xs[idx] * (w / low[0]))
            top  = max(0, min(y - ch // 2, h - ch))
            left = max(0, min(x - cw // 2, w - cw))

            img_c = TF.crop(img, top, left, ch, cw)
            ms_c  = [TF.crop(m, top, left, ch, cw) for m in masks]

            u = np.zeros((ch, cw), dtype=np.uint8)
            for mc in ms_c:
                u = np.maximum(u, np.array(mc, dtype=np.uint8))
            if float((u > 0).mean()) >= min_ratio:
                return img_c, ms_c

        return self._random_crop(img, masks, ch, cw)

    # ── transforms ────────────────────────────────────────────────────────
    def _train_transforms(self, img, masks):
        cfg = self.cfg
        ch = max(64, int(self.img_size[0] * cfg["CROP_RATIO"]))
        cw = max(64, int(self.img_size[1] * cfg["CROP_RATIO"]))

        # 1) defect-focused crop
        if random.random() < cfg["DEFECT_CROP_PROB"]:
            img, masks = self._defect_crop(
                img, masks, ch, cw,
                tries=cfg["CROP_TRIES"],
                min_ratio=cfg["MIN_DEFECT_RATIO"],
            )
        else:
            img, masks = self._random_crop(img, masks, ch, cw)

        # 2) resize
        img   = TF.resize(img, self.img_size)
        masks = [TF.resize(m, self.img_size, interpolation=Image.NEAREST) for m in masks]

        # 3) random flips
        if random.random() > 0.5:
            img   = TF.hflip(img)
            masks = [TF.hflip(m) for m in masks]
        if random.random() > 0.5:
            img   = TF.vflip(img)
            masks = [TF.vflip(m) for m in masks]

        # 4) rotation
        if random.random() > 0.5:
            a = random.uniform(-15, 15)
            img   = TF.rotate(img, a)
            masks = [TF.rotate(m, a, interpolation=Image.NEAREST) for m in masks]

        # 5) albumentations spatial (GridDistortion, ElasticTransform)
        if self.album_spatial is not None and random.random() < 0.5:
            img_np   = np.array(img)
            masks_np = {f"mask{i}": np.array(masks[i]) for i in range(self.C)}
            masks_np["image"] = img_np
            masks_np["mask"]  = np.array(masks[0])  # primary target
            # We do not use the "mask" key directly; use additional_targets
            result = self.album_spatial(
                image=img_np,
                **{f"mask{i}": np.array(masks[i]) for i in range(self.C)},
            )
            img   = Image.fromarray(result["image"])
            masks = [Image.fromarray(result[f"mask{i}"]) for i in range(self.C)]

        # 6) colour jitter (PIL image only)
        if cfg.get("COLOR_JITTER", False) and random.random() > 0.5:
            import torchvision.transforms as T
            img = T.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
            )(img)

        # 7) gaussian blur (PIL)
        if cfg.get("RANDOM_BLUR", False) and random.random() < cfg.get("BLUR_PROB", 0.3):
            ks = random.choice([3, 5, 7])
            img = TF.gaussian_blur(img, ks)

        return img, masks

    # ── __getitem__ ───────────────────────────────────────────────────────
    def __getitem__(self, idx):
        cfg      = self.cfg
        img_name = self.images[idx]
        img      = Image.open(os.path.join(self.img_dir, img_name)).convert("RGB")
        base     = os.path.splitext(img_name)[0]
        masks    = self._load_masks(base, img.size)

        if self.train:
            img, masks = self._train_transforms(img, masks)
        else:
            img   = TF.resize(img, self.img_size)
            masks = [TF.resize(m, self.img_size, interpolation=Image.NEAREST) for m in masks]

        # to tensor [0,1]
        img_t = TF.to_tensor(img)

        # noise BEFORE normalize
        if self.train and cfg.get("RANDOM_NOISE", False):
            if random.random() < cfg.get("NOISE_PROB", 0.2):
                noise = torch.randn_like(img_t) * cfg.get("NOISE_STD", 0.02)
                img_t = torch.clamp(img_t + noise, 0.0, 1.0)

        # ImageNet normalize
        img_t = TF.normalize(img_t, self.IMAGENET_MEAN, self.IMAGENET_STD)

        # masks → [C,H,W] float32 binary
        mask_t = torch.stack(
            [torch.from_numpy((np.array(m, dtype=np.uint8) > 0).astype(np.float32))
             for m in masks],
            dim=0,
        )
        return img_t, mask_t


def build_dataloaders(cfg: dict):
    root = cfg["DATA_ROOT"]
    bs   = cfg["BATCH_SIZE"]
    nw   = cfg["NUM_WORKERS"]
    sz   = cfg["IMG_SIZE"]

    train_ds = DACL10KMultiLabel(
        os.path.join(root, cfg["IMAGES_SUBDIR"], "train"),
        os.path.join(root, cfg["MASKS_SUBDIR"],  "train"),
        cfg["CLASSES_TO_LOAD"], sz, is_train=True, cfg=cfg,
    )
    val_ds = DACL10KMultiLabel(
        os.path.join(root, cfg["IMAGES_SUBDIR"], "val"),
        os.path.join(root, cfg["MASKS_SUBDIR"],  "val"),
        cfg["CLASSES_TO_LOAD"], sz, is_train=False, cfg=cfg,
    )

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=True, drop_last=False)

    print(f"  ✅ Train: {len(train_ds)} imgs / {len(train_loader)} batches")
    print(f"  ✅ Val:   {len(val_ds)} imgs / {len(val_loader)} batches")
    return train_loader, val_loader


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  2.  MODEL  –  DeepLabV3+
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DeepLabV3Plus(nn.Module):
    """
    Wrapper around torchvision DeepLabV3.
    Training  →  returns (main_logits, aux_logits)
    Eval      →  returns main_logits
    """
    def __init__(self, n_classes: int = 3, backbone: str = "resnet101",
                 pretrained: bool = True):
        super().__init__()
        self.n_classes = n_classes

        if backbone == "resnet101":
            if _NEW_WEIGHTS:
                w = DeepLabV3_ResNet101_Weights.DEFAULT if pretrained else None
                self.model = deeplabv3_resnet101(weights=w)
            else:
                self.model = deeplabv3_resnet101(pretrained=pretrained)
        else:
            if _NEW_WEIGHTS:
                w = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
                self.model = deeplabv3_resnet50(weights=w)
            else:
                self.model = deeplabv3_resnet50(pretrained=pretrained)

        # Replace main head
        in_ch = self.model.classifier[4].in_channels
        self.model.classifier[4] = nn.Conv2d(in_ch, n_classes, 1)

        # Replace aux head
        self._has_aux = False
        if hasattr(self.model, "aux_classifier") and self.model.aux_classifier is not None:
            aux_in = self.model.aux_classifier[4].in_channels
            self.model.aux_classifier[4] = nn.Conv2d(aux_in, n_classes, 1)
            self._has_aux = True

    def forward(self, x):
        h, w = x.shape[-2:]
        out  = self.model(x)
        main = out["out"]
        if main.shape[-2:] != (h, w):
            main = F.interpolate(main, (h, w), mode="bilinear", align_corners=False)

        if self.training and self._has_aux and "aux" in out:
            aux = out["aux"]
            if aux.shape[-2:] != (h, w):
                aux = F.interpolate(aux, (h, w), mode="bilinear", align_corners=False)
            return main, aux
        return main


def build_model(cfg: dict, device):
    bb = cfg["BACKBONE"]
    nc = cfg["NUM_LABELS"]
    model = DeepLabV3Plus(n_classes=nc, backbone=bb, pretrained=True)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  📐 DeepLabV3+ | backbone={bb} | classes={nc} | params={n/1e6:.1f}M")
    return model.to(device)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  3.  LOSSES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MultiLabelDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        p = probs.flatten(2)     # [B, C, H*W]
        t = targets.flatten(2)
        inter = (p * t).sum(2)
        denom = p.sum(2) + t.sum(2)
        dice  = (2.0 * inter + self.smooth) / (denom + self.smooth)
        return (1.0 - dice).mean()


class FocalLoss(nn.Module):
    """Sigmoid focal loss for multi-label segmentation."""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.sigmoid(logits) * targets + (1 - torch.sigmoid(logits)) * (1 - targets)
        focal = self.alpha * (1 - p_t) ** self.gamma * bce
        return focal.mean()


class CombinedLoss(nn.Module):
    """
    BCE (label-smoothed, class-weighted) + Dice + Focal
    """
    def __init__(self, cfg: dict):
        super().__init__()
        self.w_bce   = cfg["W_BCE"]
        self.w_dice  = cfg["W_DICE"]
        self.w_focal = cfg["W_FOCAL"]
        self.smooth  = cfg["LABEL_SMOOTHING"]

        # pos_weight stored as raw list (created as tensor during forward)
        self.pw_vals = list(cfg["BCE_POS_WEIGHT"])
        self.dice    = MultiLabelDiceLoss(smooth=cfg["DICE_SMOOTH"])
        self.focal   = FocalLoss(alpha=cfg["FOCAL_ALPHA"], gamma=cfg["FOCAL_GAMMA"])

        print(f"  📊 Loss: BCE(smooth={self.smooth}, pw={self.pw_vals}) "
              f"+ Dice + Focal(α={cfg['FOCAL_ALPHA']}, γ={cfg['FOCAL_GAMMA']})")

    def forward(self, logits, targets):
        # label smoothing
        t_smooth = targets * (1.0 - self.smooth) + self.smooth * 0.5

        # weighted BCE (manual for device safety)
        pw = torch.tensor(self.pw_vals, dtype=logits.dtype,
                          device=logits.device).view(1, -1, 1, 1)
        sig = torch.sigmoid(logits)
        bce = -(t_smooth * pw * torch.log(sig + 1e-7)
                + (1 - t_smooth) * torch.log(1 - sig + 1e-7))
        bce = bce.mean()

        d = self.dice(logits, targets)
        f = self.focal(logits, targets)

        return self.w_bce * bce + self.w_dice * d + self.w_focal * f


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  4.  METRICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MultiLabelMetrics:
    """Accumulates TP/FP/FN across batches → precision, recall, F1, IoU."""

    def __init__(self, num_classes: int, class_names: List[str],
                 threshold: float = 0.25, ignore_empty: bool = True):
        self.C     = num_classes
        self.names = class_names
        self.thr   = threshold
        self.ignore_empty = ignore_empty
        self.eps   = 1e-7
        self.reset()

    def reset(self):
        z = lambda: torch.zeros(self.C, dtype=torch.float64)
        self.tp = z(); self.fp = z(); self.fn = z(); self.gt_pos = z()

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        logits  = logits.detach().float().cpu()
        targets = targets.detach().float().cpu()
        preds   = (torch.sigmoid(logits) >= self.thr).to(torch.uint8)
        gt      = (targets >= 0.5).to(torch.uint8)

        C = preds.shape[1]
        pf = preds.permute(1, 0, 2, 3).reshape(C, -1)
        tf = gt.permute(1, 0, 2, 3).reshape(C, -1)

        self.tp     += (pf & tf).sum(1).to(torch.float64)
        self.fp     += (pf & (1 - tf)).sum(1).to(torch.float64)
        self.fn     += ((1 - pf) & tf).sum(1).to(torch.float64)
        self.gt_pos += tf.sum(1).to(torch.float64)

    def compute(self) -> Dict:
        tp, fp, fn = self.tp, self.fp, self.fn
        prec  = tp / (tp + fp + self.eps)
        rec   = tp / (tp + fn + self.eps)
        f1    = 2 * tp / (2 * tp + fp + fn + self.eps)
        iou   = tp / (tp + fp + fn + self.eps)

        valid = (self.gt_pos > 0) if self.ignore_empty else torch.ones(self.C, dtype=torch.bool)
        nv    = max(1, int(valid.sum().item()))

        per_class = []
        for i in range(self.C):
            per_class.append({
                "name":      self.names[i] if i < len(self.names) else f"cls{i}",
                "IoU":       float(iou[i]),
                "F1":        float(f1[i]),
                "Precision": float(prec[i]),
                "Recall":    float(rec[i]),
                "TP":        int(tp[i]),
                "FP":        int(fp[i]),
                "FN":        int(fn[i]),
                "valid":     bool(valid[i]),
            })
        return {
            "mean_IoU":       float(iou[valid].sum() / nv),
            "mean_F1":        float(f1[valid].sum() / nv),
            "mean_Precision": float(prec[valid].sum() / nv),
            "mean_Recall":    float(rec[valid].sum() / nv),
            "per_class":      per_class,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  5.  TRAINER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Trainer:
    def __init__(self, cfg: dict = CONFIG):
        self.cfg = cfg
        torch.manual_seed(cfg["RANDOM_SEED"])
        np.random.seed(cfg["RANDOM_SEED"])
        random.seed(cfg["RANDOM_SEED"])

        os.makedirs(cfg["SAVE_DIR"], exist_ok=True)
        os.makedirs(cfg["LOG_DIR"],  exist_ok=True)

        # device
        self.device, self.dev_type = self._pick_device()

        # AMP
        self.use_amp = cfg["USE_AMP"] and self.dev_type == "cuda"
        if self.use_amp:
            self.scaler = GradScaler("cuda") if _NEW_AMP else GradScaler()
        else:
            self.scaler = None

        self._print_banner()

        # data
        print("\n📂 DATASET")
        self.train_loader, self.val_loader = build_dataloaders(cfg)

        # model
        print("\n📐 MODEL")
        self.model = build_model(cfg, self.device)

        # loss
        print("\n📊 LOSS")
        self.criterion = CombinedLoss(cfg)

        # optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=cfg["LEARNING_RATE"],
            weight_decay=cfg["WEIGHT_DECAY"],
        )
        print(f"  ⚙️  AdamW  lr={cfg['LEARNING_RATE']}  wd={cfg['WEIGHT_DECAY']}")

        # scheduler – OneCycleLR
        steps_per_epoch = len(self.train_loader)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=cfg["MAX_LR"],
            epochs=cfg["NUM_EPOCHS"],
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
            anneal_strategy="cos",
        )
        print(f"  📈 OneCycleLR  max_lr={cfg['MAX_LR']}  "
              f"steps/epoch={steps_per_epoch}")

        # metrics
        self.metrics = MultiLabelMetrics(
            cfg["NUM_LABELS"], cfg["CLASS_NAMES"],
            threshold=cfg["THRESHOLD"], ignore_empty=True,
        )

        # state
        self.best_miou = 0.0
        self.start_epoch = 0

        # attempt resume
        ckpt = os.path.join(cfg["SAVE_DIR"], "checkpoint_best.pth")
        if os.path.exists(ckpt):
            self._load(ckpt)

        amp_str = "enabled ✔" if self.use_amp else "disabled"
        print(f"\n  ⚡ AMP: {amp_str}")
        print(f"  🔀 Aux loss weight: {cfg['AUX_LOSS_WEIGHT']}")
        print(f"  🎯 TTA: {'enabled ✔' if cfg['USE_TTA'] else 'disabled'}")
        print(f"  ⏱  EarlyStopping patience: {cfg['EARLY_STOP_PAT']}")

    # ── device ────────────────────────────────────────────────────────────
    @staticmethod
    def _pick_device():
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"  🚀 CUDA: {name}")
            return torch.device("cuda"), "cuda"
        print("  🐌 CPU")
        return torch.device("cpu"), "cpu"

    # ── banner ────────────────────────────────────────────────────────────
    def _print_banner(self):
        c = self.cfg
        print("\n" + "═" * 72)
        print("  DACL10K  FINAL  TRAINING  PIPELINE")
        print("═" * 72)
        for k, v in [
            ("Device",       f"{self.device} ({self.dev_type})"),
            ("Backbone",     c["BACKBONE"]),
            ("Classes",      c["CLASS_NAMES"]),
            ("Image size",   c["IMG_SIZE"]),
            ("Batch",        c["BATCH_SIZE"]),
            ("Epochs",       c["NUM_EPOCHS"]),
            ("LR / Max LR",  f"{c['LEARNING_RATE']} / {c['MAX_LR']}"),
            ("Loss",         f"BCE(ls={c['LABEL_SMOOTHING']}) + Dice + Focal"),
            ("AMP",          c["USE_AMP"]),
        ]:
            print(f"  {k:<14}: {v}")
        print("═" * 72)

    # ── checkpoint I/O ────────────────────────────────────────────────────
    def _save(self, epoch, miou):
        path = os.path.join(self.cfg["SAVE_DIR"], "checkpoint_best.pth")
        torch.save({
            "epoch":              epoch,
            "model_state_dict":   self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_miou":          float(miou),
            "config":             self.cfg,
        }, path)
        print(f"  💾 Saved BEST → {path}  (mIoU={miou:.4f})")

    def _load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        key  = "model_state_dict" if "model_state_dict" in ckpt else "model"
        self.model.load_state_dict(ckpt[key])
        if "optimizer_state_dict" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception:
                pass  # optimizer mismatch is OK on resume with new scheduler
        self.best_miou   = ckpt.get("best_miou", 0.0)
        self.start_epoch = ckpt.get("epoch", 0) + 1
        print(f"  🔄 Resumed from epoch {self.start_epoch-1}  "
              f"(best mIoU={self.best_miou:.4f})")

    # ── forward + loss (handles aux head) ─────────────────────────────────
    def _forward_loss(self, imgs, masks):
        out = self.model(imgs)
        if isinstance(out, tuple):
            main, aux = out
            loss = (self.criterion(main, masks)
                    + self.cfg["AUX_LOSS_WEIGHT"] * self.criterion(aux, masks))
            return main, loss
        return out, self.criterion(out, masks)

    # ── train epoch ───────────────────────────────────────────────────────
    def _train_epoch(self, epoch):
        self.model.train()
        total = 0.0
        pbar = tqdm(self.train_loader,
                    desc=f"Epoch {epoch+1}/{self.cfg['NUM_EPOCHS']} [TRAIN]",
                    leave=False)
        for imgs, masks in pbar:
            imgs  = imgs.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                ctx = autocast("cuda") if _NEW_AMP else autocast()
                with ctx:
                    _, loss = self._forward_loss(imgs, masks)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                _, loss = self._forward_loss(imgs, masks)
                loss.backward()
                self.optimizer.step()

            self.scheduler.step()  # OneCycleLR steps every batch
            total += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             lr=f"{self.optimizer.param_groups[0]['lr']:.2e}")
        return total / max(1, len(self.train_loader))

    # ── validate (with optional TTA) ──────────────────────────────────────
    @torch.no_grad()
    def _validate(self, epoch):
        self.model.eval()
        self.metrics.reset()
        total = 0.0
        use_tta = self.cfg["USE_TTA"]

        pbar = tqdm(self.val_loader,
                    desc=f"Epoch {epoch+1}/{self.cfg['NUM_EPOCHS']} [VAL]",
                    leave=False)
        for imgs, masks in pbar:
            imgs  = imgs.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            logits = self.model(imgs)
            loss   = self.criterion(logits, masks)
            total += loss.item()

            if use_tta:
                # horizontal flip TTA
                imgs_flip   = torch.flip(imgs, dims=[-1])
                logits_flip = self.model(imgs_flip)
                logits_flip = torch.flip(logits_flip, dims=[-1])
                logits      = (logits + logits_flip) / 2.0

            self.metrics.update(logits, masks)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        met = self.metrics.compute()
        return total / max(1, len(self.val_loader)), met

    # ── main training loop ────────────────────────────────────────────────
    def train(self):
        print("\n" + "═" * 72)
        print(f"  🚀 STARTING TRAINING FROM EPOCH {self.start_epoch}")
        print("═" * 72 + "\n")

        no_improve = 0
        patience   = self.cfg["EARLY_STOP_PAT"]

        for epoch in range(self.start_epoch, self.cfg["NUM_EPOCHS"]):
            t0 = time.time()
            tr_loss        = self._train_epoch(epoch)
            val_loss, met  = self._validate(epoch)
            miou           = met["mean_IoU"]
            dt             = time.time() - t0

            print(f"\n  📊 Epoch {epoch+1}/{self.cfg['NUM_EPOCHS']}  "
                  f"({dt:.0f}s)")
            print(f"     Train Loss : {tr_loss:.4f}  │  "
                  f"Val Loss : {val_loss:.4f}  │  "
                  f"mIoU : {miou:.4f}")
            print(f"     LR : {self.optimizer.param_groups[0]['lr']:.2e}")
            for c in met["per_class"]:
                tag = "✓" if c["valid"] else "·"
                print(f"     {tag} {c['name']:12s}  IoU={c['IoU']:.4f}  "
                      f"F1={c['F1']:.4f}  "
                      f"P={c['Precision']:.4f}  R={c['Recall']:.4f}")

            if miou > self.best_miou:
                self.best_miou = miou
                no_improve = 0
                self._save(epoch, miou)
            else:
                no_improve += 1

            if no_improve >= patience:
                print(f"\n  ⚠️  Early stopping – no improvement for "
                      f"{patience} epochs.")
                break

        print("\n" + "═" * 72)
        print(f"  ✅ TRAINING COMPLETE   Best mIoU = {self.best_miou:.4f}")
        print("═" * 72 + "\n")
        return self.best_miou


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  6.  TEST / INFERENCE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_test(cfg: dict = CONFIG):
    """
    Load best checkpoint → run inference on TEST_IMG_DIR → save colour
    overlays → print per-class report.
    """
    ckpt_path = os.path.join(cfg["SAVE_DIR"], "checkpoint_best.pth")
    if not os.path.exists(ckpt_path):
        print("❌ No checkpoint found – skipping test.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "═" * 72)
    print("  🧪 TEST / INFERENCE  PHASE")
    print("═" * 72)
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Test images: {cfg['TEST_IMG_DIR']}")
    print(f"  Overlays   : {cfg['TEST_OVERLAY_DIR']}")

    # load model
    model = DeepLabV3Plus(n_classes=cfg["NUM_LABELS"],
                          backbone=cfg["BACKBONE"], pretrained=False)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    key  = "model_state_dict" if "model_state_dict" in ckpt else "model"
    model.load_state_dict(ckpt[key])
    model.to(device).eval()
    print(f"  ✅ Model loaded  (best mIoU = {ckpt.get('best_miou', '?')})")

    # prepare test images
    test_dir = cfg["TEST_IMG_DIR"]
    test_imgs = sorted(
        f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".png"))
    )
    print(f"  Found {len(test_imgs)} test images")

    os.makedirs(cfg["TEST_OVERLAY_DIR"], exist_ok=True)

    # optional: if masks exist alongside test images we can compute metrics
    mask_dir = os.path.join(cfg["DATA_ROOT"], cfg["MASKS_SUBDIR"], "val")
    has_masks = os.path.isdir(mask_dir)
    if has_masks:
        metrics = MultiLabelMetrics(
            cfg["NUM_LABELS"], cfg["CLASS_NAMES"],
            threshold=cfg["THRESHOLD"], ignore_empty=True,
        )
    else:
        metrics = None
        print("  ℹ️  No mask directory found – overlay-only mode (no metrics)")

    sz       = cfg["IMG_SIZE"]
    thr      = cfg["THRESHOLD"]
    colours  = cfg["CLASS_COLORS"]
    alpha    = cfg["OVERLAY_ALPHA"]
    names    = cfg["CLASS_NAMES"]
    cls_ids  = cfg["CLASSES_TO_LOAD"]
    MEAN     = DACL10KMultiLabel.IMAGENET_MEAN
    STD      = DACL10KMultiLabel.IMAGENET_STD

    with torch.no_grad():
        for fname in tqdm(test_imgs, desc="  Inference"):
            img_pil = Image.open(os.path.join(test_dir, fname)).convert("RGB")
            orig_w, orig_h = img_pil.size

            # preprocess
            img_r = TF.resize(img_pil, sz)
            img_t = TF.to_tensor(img_r)
            img_t = TF.normalize(img_t, MEAN, STD).unsqueeze(0).to(device)

            # forward (+ TTA)
            logits = model(img_t)
            if cfg["USE_TTA"]:
                logits_f = model(torch.flip(img_t, [-1]))
                logits   = (logits + torch.flip(logits_f, [-1])) / 2.0

            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()   # [C,H,W]

            # load GT mask for metrics if available
            if metrics is not None:
                base = os.path.splitext(fname)[0]
                mask_list = []
                for cid in cls_ids:
                    fp = os.path.join(mask_dir, f"{base}_class{cid:02d}.png")
                    if os.path.exists(fp):
                        m = Image.open(fp).convert("L")
                    else:
                        m = Image.new("L", img_pil.size, 0)
                    m = TF.resize(m, sz, interpolation=Image.NEAREST)
                    mask_list.append(
                        torch.from_numpy(
                            (np.array(m, dtype=np.uint8) > 0).astype(np.float32)
                        )
                    )
                gt = torch.stack(mask_list).unsqueeze(0)   # [1,C,H,W]
                metrics.update(logits.cpu(), gt)

            # build overlay
            overlay = np.array(TF.resize(img_pil, sz)).copy()
            for i, name in enumerate(names):
                mask = (probs[i] > thr).astype(np.uint8)
                r, g, b = colours[name]
                colour_layer = np.zeros_like(overlay)
                colour_layer[..., 0] = r
                colour_layer[..., 1] = g
                colour_layer[..., 2] = b
                mask3 = np.stack([mask]*3, axis=-1)
                overlay = np.where(
                    mask3,
                    (overlay * (1 - alpha) + colour_layer * alpha).astype(np.uint8),
                    overlay,
                )

            out_name = os.path.splitext(fname)[0] + "_overlay.png"
            Image.fromarray(overlay).save(
                os.path.join(cfg["TEST_OVERLAY_DIR"], out_name)
            )

    # ── Final report ──────────────────────────────────────────────────────
    if metrics is not None:
        met = metrics.compute()
        print("\n" + "═" * 72)
        print("  📋 TEST  RESULTS  (Per-Class Report)")
        print("═" * 72)
        print(f"  {'Class':12s}  {'Precision':>10}  {'Recall':>10}  "
              f"{'F1':>10}  {'IoU':>10}")
        print("  " + "─" * 58)
        for c in met["per_class"]:
            print(f"  {c['name']:12s}  {c['Precision']:10.4f}  "
                  f"{c['Recall']:10.4f}  {c['F1']:10.4f}  {c['IoU']:10.4f}")
        print("  " + "─" * 58)
        print(f"  {'MEAN':12s}  {met['mean_Precision']:10.4f}  "
              f"{met['mean_Recall']:10.4f}  {met['mean_F1']:10.4f}  "
              f"{met['mean_IoU']:10.4f}")
        print("═" * 72)
    else:
        print(f"\n  ✅ Overlays saved to {cfg['TEST_OVERLAY_DIR']}")

    print("\n  🎨 Colour Legend:")
    for name, col in colours.items():
        print(f"     {name:12s} → RGB{col}")
    print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  7.  MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║     DACL10K  MULTI-LABEL  SEGMENTATION  –  FINAL  PIPELINE     ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    # ── PHASE 1: Train + Validate ─────────────────────────────────────────
    trainer = Trainer(CONFIG)
    best = trainer.train()

    # ── PHASE 2: Test / Inference ─────────────────────────────────────────
    run_test(CONFIG)

    print("🏁 ALL DONE.")
