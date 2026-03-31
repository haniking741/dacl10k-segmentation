#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  DACL10K  Multi-Label  Semantic  Segmentation  –  FIXED  PIPELINE          ║
║                                                                            ║
║  Fixes applied vs original:                                                ║
║   1. Removed pos_weight (was causing all-positive prediction)              ║
║   2. Threshold 0.25 → 0.50                                                ║
║   3. Removed label smoothing (hurts binary segmentation)                   ║
║   4. OneCycleLR → CosineAnnealingWarmRestarts (resume-safe)               ║
║   5. Differential LR: backbone 1e-5, head 1e-4                            ║
║   6. Aux loss weight 0.4 → 0.2                                            ║
║   7. Gradient clipping (max_norm=1.0)                                      ║
║   8. Crop ratio 0.60 → 0.80 (preserve context)                            ║
║   9. Batch size comment: use 16 if GPU memory allows                       ║
║  10. Loss: Dice-only warmup (2 epochs) then full combined                  ║
║  11. Early stopping patience 5 → 8                                         ║
║  12. TTA includes scale TTA (0.75×, 1.0×, 1.25×)                         ║
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

try:
    from torch.amp import autocast, GradScaler
    _NEW_AMP = True
except ImportError:
    from torch.cuda.amp import autocast, GradScaler  # type: ignore
    _NEW_AMP = False

from tqdm import tqdm

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUM = True
except ImportError:
    HAS_ALBUM = False
    print("⚠️  albumentations not installed – GridDistortion / ElasticTransform disabled")

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
#  CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONFIG = {
    # ── Paths ──────────────────────────────────────────────────────────────
    "DATA_ROOT":        "/kaggle/input/datasets/hanihafnaoui/hani-dataset/dataset2",
    "IMAGES_SUBDIR":    "images",
    "MASKS_SUBDIR":     "masks_multilabel",
    "SAVE_DIR":         "/kaggle/working/checkpoints_final",
    "LOG_DIR":          "/kaggle/working/logs_final",
    "TEST_IMG_DIR":     "/kaggle/input/datasets/hanihafnaoui/hani-dataset/dataset2/images/val",
    "TEST_OVERLAY_DIR": "/kaggle/working/test_overlays",

    # ── Dataset ────────────────────────────────────────────────────────────
    "CLASSES_TO_LOAD":  [1, 7, 11],           # crack, spalling, rust
    "CLASS_NAMES":      ["crack", "spalling", "rust"],
    "NUM_LABELS":       3,

    # ── Model ──────────────────────────────────────────────────────────────
    "BACKBONE":         "resnet101",
    # FIX #6: reduced aux weight from 0.4 → 0.2
    "AUX_LOSS_WEIGHT":  0.2,

    # ── Training ───────────────────────────────────────────────────────────
    "IMG_SIZE":         (512, 512),
    # FIX #9: use 16 if VRAM allows; T4 16GB can handle 16 with AMP
    "BATCH_SIZE":       16,
    "NUM_WORKERS":      4,
    "NUM_EPOCHS":       60,
    "RANDOM_SEED":      42,

    # ── Optimizer ──────────────────────────────────────────────────────────
    # FIX #5: differential LR – backbone is pre-trained so it needs a much
    # smaller LR than the freshly-initialised segmentation head.
    "BACKBONE_LR":      1e-5,
    "HEAD_LR":          1e-4,
    "WEIGHT_DECAY":     1e-4,

    # ── Scheduler ─────────────────────────────────────────────────────────
    # FIX #4: CosineAnnealingWarmRestarts is resume-safe and avoids the
    # "single cycle burned" problem that makes OneCycleLR dangerous with
    # early stopping and checkpoint resuming.
    "SCHEDULER":        "cosine_warm",       # "cosine_warm" | "plateau"
    "COSINE_T0":        15,                  # restart every 15 epochs
    "COSINE_T_MULT":    2,                   # periods double after each restart
    "WARMUP_EPOCHS":    2,                   # linear warm-up before cosine

    # ── Loss ───────────────────────────────────────────────────────────────
    # FIX #1: pos_weight removed – it was overwhelming BCE and causing the
    # model to predict everything as positive (recall~1, precision~0.03).
    # FIX #3: label smoothing removed – it blurs the binary decision boundary
    # and makes the model hedge instead of committing.
    # FIX #10: DICE_WARMUP_EPOCHS: first N epochs train with Dice only (stable
    # early signal), then full combined loss kicks in.
    "DICE_SMOOTH":      1.0,
    "FOCAL_ALPHA":      0.25,
    "FOCAL_GAMMA":      2.0,
    "W_BCE":            1.0,
    "W_DICE":           1.5,               # upweighted – Dice is more robust
    "W_FOCAL":          0.5,
    "DICE_WARMUP_EPOCHS": 2,              # train dice-only for first 2 epochs

    # ── Augmentation ───────────────────────────────────────────────────────
    "DEFECT_CROP_PROB":   0.65,
    # FIX #8: crop ratio 0.60 → 0.80 to preserve spatial context
    "CROP_RATIO":         0.80,
    "CROP_TRIES":         10,
    "MIN_DEFECT_RATIO":   0.005,           # slightly lower → finds more crops
    "COLOR_JITTER":       True,
    "RANDOM_BLUR":        True,
    "RANDOM_NOISE":       True,
    "NOISE_STD":          0.015,
    "NOISE_PROB":         0.15,
    "BLUR_PROB":          0.25,
    "USE_ALBUM_SPATIAL":  True,

    # ── Validation / Inference ─────────────────────────────────────────────
    # FIX #2: threshold 0.25 → 0.50 – a lower threshold was accepting enormous
    # false-positive regions, tanking precision to ~0.03.
    "THRESHOLD":        0.50,
    "USE_TTA":          True,

    # FIX #11: patience 5 → 8 (cosine restarts need more room)
    "EARLY_STOP_PAT":   8,

    # ── Gradient clipping (FIX #7) ─────────────────────────────────────────
    "GRAD_CLIP":        1.0,

    # ── AMP ────────────────────────────────────────────────────────────────
    "USE_AMP":          True,

    # ── Overlay colours ────────────────────────────────────────────────────
    "CLASS_COLORS": {
        "crack":    (255,   0,   0),
        "spalling": (  0,   0, 255),
        "rust":     (255, 255,   0),
    },
    "OVERLAY_ALPHA": 0.45,
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  1.  DATASET
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DACL10KMultiLabel(Dataset):
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

        self.album_spatial = None
        if is_train and HAS_ALBUM and cfg.get("USE_ALBUM_SPATIAL", False):
            self.album_spatial = A.Compose([
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.4),
                A.ElasticTransform(alpha=80, sigma=80 * 0.05, p=0.3),
            ], additional_targets={f"mask{i}": "mask" for i in range(self.C)})

    def __len__(self):
        return len(self.images)

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

    @staticmethod
    def _random_crop(img, masks, ch, cw):
        w, h = img.size
        if h <= ch or w <= cw:
            return img, masks
        top  = random.randint(0, h - ch)
        left = random.randint(0, w - cw)
        img   = TF.crop(img, top, left, ch, cw)
        masks = [TF.crop(m, top, left, ch, cw) for m in masks]
        return img, masks

    def _defect_crop(self, img, masks, ch, cw, tries=10, min_ratio=0.005):
        w, h = img.size
        if h <= ch or w <= cw:
            return img, masks

        low = (256, 256)
        union = np.zeros(low[::-1], dtype=np.uint8)
        for m in masks:
            ms = m.resize(low, resample=Image.NEAREST)
            union = np.maximum(union, np.array(ms, dtype=np.uint8))

        ys, xs = np.where(union > 0)
        if ys.size == 0:
            return self._random_crop(img, masks, ch, cw)

        for _ in range(tries):
            idx  = random.randint(0, ys.size - 1)
            y    = int(ys[idx] * (h / low[1]))
            x    = int(xs[idx] * (w / low[0]))
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

    def _train_transforms(self, img, masks):
        cfg = self.cfg
        # FIX #8: larger crop ratio = 0.80 preserves more spatial context
        ch = max(64, int(self.img_size[0] * cfg["CROP_RATIO"]))
        cw = max(64, int(self.img_size[1] * cfg["CROP_RATIO"]))

        if random.random() < cfg["DEFECT_CROP_PROB"]:
            img, masks = self._defect_crop(
                img, masks, ch, cw,
                tries=cfg["CROP_TRIES"],
                min_ratio=cfg["MIN_DEFECT_RATIO"],
            )
        else:
            img, masks = self._random_crop(img, masks, ch, cw)

        img   = TF.resize(img, self.img_size)
        masks = [TF.resize(m, self.img_size, interpolation=Image.NEAREST) for m in masks]

        if random.random() > 0.5:
            img   = TF.hflip(img)
            masks = [TF.hflip(m) for m in masks]
        if random.random() > 0.5:
            img   = TF.vflip(img)
            masks = [TF.vflip(m) for m in masks]

        if random.random() > 0.5:
            a     = random.uniform(-15, 15)
            img   = TF.rotate(img, a)
            masks = [TF.rotate(m, a, interpolation=Image.NEAREST) for m in masks]

        if self.album_spatial is not None and random.random() < 0.5:
            img_np = np.array(img)
            result = self.album_spatial(
                image=img_np,
                **{f"mask{i}": np.array(masks[i]) for i in range(self.C)},
            )
            img   = Image.fromarray(result["image"])
            masks = [Image.fromarray(result[f"mask{i}"]) for i in range(self.C)]

        if cfg.get("COLOR_JITTER", False) and random.random() > 0.5:
            import torchvision.transforms as T
            img = T.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
            )(img)

        if cfg.get("RANDOM_BLUR", False) and random.random() < cfg.get("BLUR_PROB", 0.25):
            ks  = random.choice([3, 5])
            img = TF.gaussian_blur(img, ks)

        return img, masks

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
            masks = [TF.resize(m, self.img_size, interpolation=Image.NEAREST)
                     for m in masks]

        img_t = TF.to_tensor(img)

        if self.train and cfg.get("RANDOM_NOISE", False):
            if random.random() < cfg.get("NOISE_PROB", 0.15):
                noise = torch.randn_like(img_t) * cfg.get("NOISE_STD", 0.015)
                img_t = torch.clamp(img_t + noise, 0.0, 1.0)

        img_t = TF.normalize(img_t, self.IMAGENET_MEAN, self.IMAGENET_STD)

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

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=nw, pin_memory=True, drop_last=True,
        persistent_workers=(nw > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=True, drop_last=False,
        persistent_workers=(nw > 0),
    )

    print(f"  ✅ Train: {len(train_ds)} imgs / {len(train_loader)} batches")
    print(f"  ✅ Val:   {len(val_ds)} imgs / {len(val_loader)} batches")
    return train_loader, val_loader


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  2.  MODEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DeepLabV3Plus(nn.Module):
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

        in_ch = self.model.classifier[4].in_channels
        self.model.classifier[4] = nn.Conv2d(in_ch, n_classes, 1)

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

    def get_param_groups(self, backbone_lr: float, head_lr: float):
        """
        FIX #5: differential LR parameter groups.
        The backbone (ResNet101) is pre-trained and should be fine-tuned
        cautiously. The segmentation head is randomly initialised and needs
        a higher LR to learn quickly.
        """
        backbone_params = []
        head_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # Everything under model.backbone is the ResNet encoder
            if "model.backbone" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        return [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params,     "lr": head_lr},
        ]


def build_model(cfg: dict, device):
    bb    = cfg["BACKBONE"]
    nc    = cfg["NUM_LABELS"]
    model = DeepLabV3Plus(n_classes=nc, backbone=bb, pretrained=True)
    n     = sum(p.numel() for p in model.parameters() if p.requires_grad)
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
        p = probs.flatten(2)
        t = targets.flatten(2)
        inter = (p * t).sum(2)
        denom = p.sum(2) + t.sum(2)
        dice  = (2.0 * inter + self.smooth) / (denom + self.smooth)
        return (1.0 - dice).mean()


class FocalLoss(nn.Module):
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
    FIX #1 & #3: No pos_weight, no label smoothing.
    Plain BCE + Dice + Focal with correct balance.
    """
    def __init__(self, cfg: dict):
        super().__init__()
        self.w_bce   = cfg["W_BCE"]
        self.w_dice  = cfg["W_DICE"]
        self.w_focal = cfg["W_FOCAL"]
        self.dice    = MultiLabelDiceLoss(smooth=cfg["DICE_SMOOTH"])
        self.focal   = FocalLoss(alpha=cfg["FOCAL_ALPHA"], gamma=cfg["FOCAL_GAMMA"])
        print(f"  📊 Loss: BCE + Dice(w={self.w_dice}) + Focal(α={cfg['FOCAL_ALPHA']}, "
              f"γ={cfg['FOCAL_GAMMA']}, w={self.w_focal})")

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        d   = self.dice(logits, targets)
        f   = self.focal(logits, targets)
        return self.w_bce * bce + self.w_dice * d + self.w_focal * f


class DiceOnlyLoss(nn.Module):
    """Used during warmup epochs (FIX #10) – Dice alone is a clean, stable signal."""
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.dice = MultiLabelDiceLoss(smooth=smooth)

    def forward(self, logits, targets):
        return self.dice(logits, targets)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  4.  METRICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MultiLabelMetrics:
    def __init__(self, num_classes: int, class_names: List[str],
                 threshold: float = 0.50, ignore_empty: bool = True):
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

        C  = preds.shape[1]
        pf = preds.permute(1, 0, 2, 3).reshape(C, -1)
        tf = gt.permute(1, 0, 2, 3).reshape(C, -1)

        self.tp     += (pf & tf).sum(1).to(torch.float64)
        self.fp     += (pf & (1 - tf)).sum(1).to(torch.float64)
        self.fn     += ((1 - pf) & tf).sum(1).to(torch.float64)
        self.gt_pos += tf.sum(1).to(torch.float64)

    def compute(self) -> Dict:
        tp, fp, fn = self.tp, self.fp, self.fn
        prec = tp / (tp + fp + self.eps)
        rec  = tp / (tp + fn + self.eps)
        f1   = 2 * tp / (2 * tp + fp + fn + self.eps)
        iou  = tp / (tp + fp + fn + self.eps)

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
#  5.  WARMUP LR SCHEDULER HELPER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class LinearWarmupScheduler:
    """
    FIX #4: Linear LR warm-up for the first WARMUP_EPOCHS, then hands off to
    CosineAnnealingWarmRestarts.  Operates per-epoch (not per-step).
    """
    def __init__(self, optimizer, warmup_epochs: int, target_lrs: List[float]):
        self.optimizer     = optimizer
        self.warmup_epochs = warmup_epochs
        self.target_lrs    = target_lrs          # one per param group

    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            scale = (epoch + 1) / max(1, self.warmup_epochs)
            for pg, tgt in zip(self.optimizer.param_groups, self.target_lrs):
                pg["lr"] = tgt * scale


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  6.  TRAINER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Trainer:
    def __init__(self, cfg: dict = CONFIG):
        self.cfg = cfg
        torch.manual_seed(cfg["RANDOM_SEED"])
        np.random.seed(cfg["RANDOM_SEED"])
        random.seed(cfg["RANDOM_SEED"])

        os.makedirs(cfg["SAVE_DIR"], exist_ok=True)
        os.makedirs(cfg["LOG_DIR"],  exist_ok=True)

        self.device, self.dev_type = self._pick_device()

        self.use_amp = cfg["USE_AMP"] and self.dev_type == "cuda"
        if self.use_amp:
            self.scaler = GradScaler("cuda") if _NEW_AMP else GradScaler()
        else:
            self.scaler = None

        self._print_banner()

        print("\n📂 DATASET")
        self.train_loader, self.val_loader = build_dataloaders(cfg)

        print("\n📐 MODEL")
        self.model = build_model(cfg, self.device)

        print("\n📊 LOSS")
        self.criterion_warmup = DiceOnlyLoss(smooth=cfg["DICE_SMOOTH"])
        self.criterion_full   = CombinedLoss(cfg)

        # FIX #5: differential LR via param groups
        param_groups = self.model.get_param_groups(
            backbone_lr=cfg["BACKBONE_LR"],
            head_lr=cfg["HEAD_LR"],
        )
        self.optimizer = optim.AdamW(param_groups, weight_decay=cfg["WEIGHT_DECAY"])
        print(f"  ⚙️  AdamW  backbone_lr={cfg['BACKBONE_LR']}  "
              f"head_lr={cfg['HEAD_LR']}  wd={cfg['WEIGHT_DECAY']}")

        # FIX #4: CosineAnnealingWarmRestarts – safe to resume, no burned cycle
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=cfg["COSINE_T0"],
            T_mult=cfg["COSINE_T_MULT"],
            eta_min=1e-7,
        )
        target_lrs = [cfg["BACKBONE_LR"], cfg["HEAD_LR"]]
        self.warmup = LinearWarmupScheduler(
            self.optimizer, cfg["WARMUP_EPOCHS"], target_lrs
        )
        print(f"  📈 CosineAnnealingWarmRestarts  T0={cfg['COSINE_T0']}  "
              f"T_mult={cfg['COSINE_T_MULT']}  warmup={cfg['WARMUP_EPOCHS']}ep")

        self.metrics = MultiLabelMetrics(
            cfg["NUM_LABELS"], cfg["CLASS_NAMES"],
            threshold=cfg["THRESHOLD"], ignore_empty=True,
        )

        self.best_miou   = 0.0
        self.start_epoch = 0

        ckpt = os.path.join(cfg["SAVE_DIR"], "checkpoint_best.pth")
        if os.path.exists(ckpt):
            self._load(ckpt)

        amp_str = "enabled ✔" if self.use_amp else "disabled"
        print(f"\n  ⚡ AMP: {amp_str}")
        print(f"  🔀 Aux loss weight: {cfg['AUX_LOSS_WEIGHT']}")
        print(f"  ✂️  Grad clip: {cfg['GRAD_CLIP']}")
        print(f"  🎯 TTA: {'enabled ✔' if cfg['USE_TTA'] else 'disabled'}")
        print(f"  🎯 Threshold: {cfg['THRESHOLD']}")
        print(f"  ⏱  EarlyStopping patience: {cfg['EARLY_STOP_PAT']}")
        print(f"  🌡  Dice warmup: {cfg['DICE_WARMUP_EPOCHS']} epochs")

    @staticmethod
    def _pick_device():
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"  🚀 CUDA: {name}")
            return torch.device("cuda"), "cuda"
        print("  🐌 CPU")
        return torch.device("cpu"), "cpu"

    def _print_banner(self):
        c = self.cfg
        print("\n" + "═" * 72)
        print("  DACL10K  FIXED  TRAINING  PIPELINE")
        print("═" * 72)
        for k, v in [
            ("Device",        f"{self.device} ({self.dev_type})"),
            ("Backbone",      c["BACKBONE"]),
            ("Classes",       c["CLASS_NAMES"]),
            ("Image size",    c["IMG_SIZE"]),
            ("Batch",         c["BATCH_SIZE"]),
            ("Epochs",        c["NUM_EPOCHS"]),
            ("Backbone LR",   c["BACKBONE_LR"]),
            ("Head LR",       c["HEAD_LR"]),
            ("Threshold",     c["THRESHOLD"]),
            ("Loss",          "BCE + Dice(1.5×) + Focal — no label smoothing"),
            ("pos_weight",    "REMOVED"),
            ("AMP",           c["USE_AMP"]),
        ]:
            print(f"  {k:<14}: {v}")
        print("═" * 72)

    def _save(self, epoch, miou):
        path = os.path.join(self.cfg["SAVE_DIR"], "checkpoint_best.pth")
        torch.save({
            "epoch":               epoch,
            "model_state_dict":    self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_miou":           float(miou),
            "config":              self.cfg,
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
                pass
        if "scheduler_state_dict" in ckpt:
            try:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except Exception:
                pass
        self.best_miou   = ckpt.get("best_miou", 0.0)
        self.start_epoch = ckpt.get("epoch", 0) + 1
        print(f"  🔄 Resumed from epoch {self.start_epoch-1}  "
              f"(best mIoU={self.best_miou:.4f})")

    def _get_criterion(self, epoch: int):
        """FIX #10: use dice-only for warmup epochs, then full loss."""
        if epoch < self.cfg["DICE_WARMUP_EPOCHS"]:
            return self.criterion_warmup
        return self.criterion_full

    def _forward_loss(self, imgs, masks, epoch: int):
        criterion = self._get_criterion(epoch)
        out = self.model(imgs)
        if isinstance(out, tuple):
            main, aux = out
            loss = (criterion(main, masks)
                    + self.cfg["AUX_LOSS_WEIGHT"] * criterion(aux, masks))
            return main, loss
        return out, criterion(out, masks)

    def _train_epoch(self, epoch: int):
        self.model.train()

        # Apply linear warm-up for first WARMUP_EPOCHS
        self.warmup.step(epoch)

        warmup_active = epoch < self.cfg["WARMUP_EPOCHS"]
        loss_label    = "Dice-only" if epoch < self.cfg["DICE_WARMUP_EPOCHS"] else "Combined"
        total = 0.0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.cfg['NUM_EPOCHS']} "
                 f"[TRAIN · {loss_label}{'· warmup' if warmup_active else ''}]",
            leave=False,
        )
        for imgs, masks in pbar:
            imgs  = imgs.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                ctx = autocast("cuda") if _NEW_AMP else autocast()
                with ctx:
                    _, loss = self._forward_loss(imgs, masks, epoch)
                self.scaler.scale(loss).backward()
                # FIX #7: gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg["GRAD_CLIP"]
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                _, loss = self._forward_loss(imgs, masks, epoch)
                loss.backward()
                # FIX #7: gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg["GRAD_CLIP"]
                )
                self.optimizer.step()

            total += loss.item()
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                bb_lr=f"{self.optimizer.param_groups[0]['lr']:.1e}",
                hd_lr=f"{self.optimizer.param_groups[1]['lr']:.1e}",
            )

        # Step cosine scheduler once per epoch (after warmup)
        if epoch >= self.cfg["WARMUP_EPOCHS"]:
            self.scheduler.step(epoch - self.cfg["WARMUP_EPOCHS"])

        return total / max(1, len(self.train_loader))

    # FIX #12: scale TTA + flip TTA
    @torch.no_grad()
    def _tta_forward(self, imgs):
        """
        Returns averaged logits from:
          - original scale
          - horizontal flip
          - 75% scale + back
          - 125% scale + back
        """
        H, W = imgs.shape[-2:]

        def _fwd(x):
            return self.model(x)

        logits = _fwd(imgs)

        # horizontal flip
        lf = _fwd(torch.flip(imgs, [-1]))
        logits = logits + torch.flip(lf, [-1])

        # 75% scale
        s75 = F.interpolate(imgs, scale_factor=0.75, mode="bilinear", align_corners=False)
        l75 = _fwd(s75)
        logits = logits + F.interpolate(l75, (H, W), mode="bilinear", align_corners=False)

        # 125% scale
        s125 = F.interpolate(imgs, scale_factor=1.25, mode="bilinear", align_corners=False)
        l125 = _fwd(s125)
        logits = logits + F.interpolate(l125, (H, W), mode="bilinear", align_corners=False)

        return logits / 4.0

    @torch.no_grad()
    def _validate(self, epoch: int):
        self.model.eval()
        self.metrics.reset()
        total    = 0.0
        use_tta  = self.cfg["USE_TTA"]
        criterion = self._get_criterion(epoch)

        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch+1}/{self.cfg['NUM_EPOCHS']} [VAL]",
            leave=False,
        )
        for imgs, masks in pbar:
            imgs  = imgs.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            if use_tta:
                logits = self._tta_forward(imgs)
            else:
                logits = self.model(imgs)

            loss   = criterion(logits, masks)
            total += loss.item()
            self.metrics.update(logits, masks)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        met = self.metrics.compute()
        return total / max(1, len(self.val_loader)), met

    def train(self):
        print("\n" + "═" * 72)
        print(f"  🚀 STARTING TRAINING FROM EPOCH {self.start_epoch}")
        print("═" * 72 + "\n")

        no_improve = 0
        patience   = self.cfg["EARLY_STOP_PAT"]
        log_path   = os.path.join(self.cfg["LOG_DIR"], "training_log.json")
        history    = []

        for epoch in range(self.start_epoch, self.cfg["NUM_EPOCHS"]):
            t0            = time.time()
            tr_loss       = self._train_epoch(epoch)
            val_loss, met = self._validate(epoch)
            miou          = met["mean_IoU"]
            dt            = time.time() - t0

            bb_lr = self.optimizer.param_groups[0]["lr"]
            hd_lr = self.optimizer.param_groups[1]["lr"]

            print(f"\n  📊 Epoch {epoch+1}/{self.cfg['NUM_EPOCHS']}  ({dt:.0f}s)")
            print(f"     Train Loss : {tr_loss:.4f}  │  Val Loss : {val_loss:.4f}  "
                  f"│  mIoU : {miou:.4f}  │  mF1 : {met['mean_F1']:.4f}")
            print(f"     BB-LR : {bb_lr:.2e}  │  HD-LR : {hd_lr:.2e}")
            print(f"     {'Class':12s}  {'IoU':>8}  {'F1':>8}  {'Prec':>8}  {'Rec':>8}")
            for c in met["per_class"]:
                tag = "✓" if c["valid"] else "·"
                print(f"     {tag} {c['name']:12s}  {c['IoU']:8.4f}  {c['F1']:8.4f}  "
                      f"{c['Precision']:8.4f}  {c['Recall']:8.4f}")

            # Log to JSON
            history.append({
                "epoch": epoch + 1,
                "train_loss": tr_loss,
                "val_loss": val_loss,
                **{k: met[k] for k in ["mean_IoU", "mean_F1", "mean_Precision", "mean_Recall"]},
            })
            with open(log_path, "w") as f:
                json.dump(history, f, indent=2)

            if miou > self.best_miou:
                self.best_miou = miou
                no_improve     = 0
                self._save(epoch, miou)
            else:
                no_improve += 1
                print(f"     ⏳ No improvement  ({no_improve}/{patience})")

            if no_improve >= patience:
                print(f"\n  ⚠️  Early stopping – no improvement for {patience} epochs.")
                break

        print("\n" + "═" * 72)
        print(f"  ✅ TRAINING COMPLETE   Best mIoU = {self.best_miou:.4f}")
        print("═" * 72 + "\n")
        return self.best_miou


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  7.  TEST / INFERENCE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_test(cfg: dict = CONFIG):
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

    model = DeepLabV3Plus(n_classes=cfg["NUM_LABELS"],
                          backbone=cfg["BACKBONE"], pretrained=False)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    key   = "model_state_dict" if "model_state_dict" in ckpt else "model"
    model.load_state_dict(ckpt[key])
    model.to(device).eval()
    print(f"  ✅ Model loaded  (best mIoU = {ckpt.get('best_miou', '?')})")

    test_dir  = cfg["TEST_IMG_DIR"]
    test_imgs = sorted(
        f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".png"))
    )
    print(f"  Found {len(test_imgs)} test images")
    os.makedirs(cfg["TEST_OVERLAY_DIR"], exist_ok=True)

    mask_dir  = os.path.join(cfg["DATA_ROOT"], cfg["MASKS_SUBDIR"], "val")
    has_masks = os.path.isdir(mask_dir)
    if has_masks:
        metrics = MultiLabelMetrics(
            cfg["NUM_LABELS"], cfg["CLASS_NAMES"],
            threshold=cfg["THRESHOLD"], ignore_empty=True,
        )
    else:
        metrics = None
        print("  ℹ️  No mask directory – overlay-only mode")

    sz      = cfg["IMG_SIZE"]
    thr     = cfg["THRESHOLD"]
    colours = cfg["CLASS_COLORS"]
    alpha   = cfg["OVERLAY_ALPHA"]
    names   = cfg["CLASS_NAMES"]
    cls_ids = cfg["CLASSES_TO_LOAD"]
    MEAN    = DACL10KMultiLabel.IMAGENET_MEAN
    STD     = DACL10KMultiLabel.IMAGENET_STD

    with torch.no_grad():
        for fname in tqdm(test_imgs, desc="  Inference"):
            img_pil = Image.open(os.path.join(test_dir, fname)).convert("RGB")

            img_r = TF.resize(img_pil, sz)
            img_t = TF.normalize(TF.to_tensor(img_r), MEAN, STD).unsqueeze(0).to(device)

            # scale + flip TTA at inference
            H, W   = img_t.shape[-2:]
            logits = model(img_t)
            lf     = model(torch.flip(img_t, [-1]))
            logits = logits + torch.flip(lf, [-1])
            if cfg["USE_TTA"]:
                s75  = F.interpolate(img_t, scale_factor=0.75, mode="bilinear", align_corners=False)
                l75  = model(s75)
                logits = logits + F.interpolate(l75, (H, W), mode="bilinear", align_corners=False)
                s125 = F.interpolate(img_t, scale_factor=1.25, mode="bilinear", align_corners=False)
                l125 = model(s125)
                logits = logits + F.interpolate(l125, (H, W), mode="bilinear", align_corners=False)
                logits = logits / 4.0
            else:
                logits = logits / 2.0

            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

            if metrics is not None:
                base = os.path.splitext(fname)[0]
                mask_list = []
                for cid in cls_ids:
                    fp = os.path.join(mask_dir, f"{base}_class{cid:02d}.png")
                    m  = Image.open(fp).convert("L") if os.path.exists(fp) \
                         else Image.new("L", img_pil.size, 0)
                    m  = TF.resize(m, sz, interpolation=Image.NEAREST)
                    mask_list.append(
                        torch.from_numpy(
                            (np.array(m, dtype=np.uint8) > 0).astype(np.float32)
                        )
                    )
                gt = torch.stack(mask_list).unsqueeze(0)
                metrics.update(logits.cpu(), gt)

            overlay = np.array(TF.resize(img_pil, sz)).copy()
            for i, name in enumerate(names):
                mask = (probs[i] > thr).astype(np.uint8)
                r, g, b = colours[name]
                colour_layer = np.zeros_like(overlay)
                colour_layer[..., 0] = r
                colour_layer[..., 1] = g
                colour_layer[..., 2] = b
                mask3 = np.stack([mask] * 3, axis=-1)
                overlay = np.where(
                    mask3,
                    (overlay * (1 - alpha) + colour_layer * alpha).astype(np.uint8),
                    overlay,
                )

            out_name = os.path.splitext(fname)[0] + "_overlay.png"
            Image.fromarray(overlay).save(
                os.path.join(cfg["TEST_OVERLAY_DIR"], out_name)
            )

    if metrics is not None:
        met = metrics.compute()
        print("\n" + "═" * 72)
        print("  📋 TEST  RESULTS")
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
#  8.  MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║   DACL10K  MULTI-LABEL  SEGMENTATION  –  FIXED  PIPELINE       ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    trainer = Trainer(CONFIG)
    best    = trainer.train()
    run_test(CONFIG)
    print("🏁 ALL DONE.")
