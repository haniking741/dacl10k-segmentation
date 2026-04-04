"""
Training Script (MULTI-LABEL) for DACL10K - 3 Classes
Optimized for RTX 4070 / Kaggle T4 with AMP

Uses CombinedLoss = BCE + Dice + Focal
"""

import os
import time
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm

try:
    from torch.amp import autocast, GradScaler
    USE_NEW_AMP_API = True
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    USE_NEW_AMP_API = False

import config
from models.deeplabv3 import get_model
from data.dataset_multilabel import get_dataloaders_multilabel
from utils.losses_multilabel import CombinedLoss   # <-- UPDATED
from utils.metrics_multilabel import MultiLabelSegmentationMetrics

AUX_LOSS_WEIGHT = 0.4

class Trainer:
    def __init__(self):
        torch.manual_seed(config.RANDOM_SEED)
        np.random.seed(config.RANDOM_SEED)

        self.device, self.device_type = self._get_device()
        self.use_amp = bool(getattr(config, "USE_AMP", False)) and (self.device_type == "cuda")
        if self.use_amp:
            self.scaler = GradScaler('cuda') if USE_NEW_AMP_API else GradScaler()
        else:
            self.scaler = None

        os.makedirs(config.SAVE_DIR, exist_ok=True)
        os.makedirs(getattr(config, "LOG_DIR", "logs"), exist_ok=True)

        self._print_config_summary()

        print("\n📂 Loading dataset (MULTI-LABEL - 3 CLASSES)...")
        self.train_loader, self.val_loader, self.num_labels = get_dataloaders_multilabel(
            data_root=config.DATA_ROOT,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            img_size=config.IMG_SIZE,
            images_subdir=config.IMAGES_SUBDIR,
            masks_subdir=config.MASKS_SUBDIR,
            cpu_mode=config.CPU_MODE,
            defect_crop_prob=config.DEFECT_CROP_PROB,
            crop_ratio=config.CROP_RATIO,
            crop_tries=config.CROP_TRIES,
            min_defect_ratio=config.MIN_DEFECT_RATIO,
        )

        print("\n📐 Creating model...")
        self.model = get_model(config.MODEL_TYPE, self.num_labels, self.device)

        self.criterion = self._get_criterion()
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler() if config.USE_SCHEDULER else None

        self.metrics = MultiLabelSegmentationMetrics(
            num_classes=self.num_labels,
            class_names=getattr(config, "CLASS_NAMES", None),
            threshold=getattr(config, "THRESHOLD", 0.5),
            ignore_empty=True,
        )

        self.best_miou = 0.0
        self.start_epoch = 0
        checkpoint_path = os.path.join(config.SAVE_DIR, "checkpoint_best_multilabel.pth")
        self.start_epoch = self.load_checkpoint(checkpoint_path)

        amp_label = ("enabled (PyTorch 2.x)" if USE_NEW_AMP_API else "enabled (PyTorch 1.x)") \
                    if self.use_amp else "disabled"
        print(f"⚡ AMP {amp_label}")
        print(f"🔀 Auxiliary loss weight: {AUX_LOSS_WEIGHT}")

    def load_checkpoint(self, path):
        if os.path.exists(path):
            print(f"🔄 Loading checkpoint: {path}")
            ckpt = torch.load(path, map_location=self.device)
            key = 'model_state_dict' if 'model_state_dict' in ckpt else 'model'
            self.model.load_state_dict(ckpt[key])
            if 'optimizer_state_dict' in ckpt:
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.best_miou = ckpt.get('best_miou', 0.0)
            epoch = ckpt.get('epoch', 0)
            print(f"✅ Resumed from epoch {epoch}  (best mIoU: {self.best_miou:.4f})")
            return epoch + 1
        print("🆕 No checkpoint found — training from scratch.")
        return 0

    def _get_device(self):
        if getattr(config, "CPU_MODE", False):
            print("🐌 CPU MODE forced")
            return torch.device("cpu"), "cpu"
        if torch.cuda.is_available():
            print("🚀 CUDA:", torch.cuda.get_device_name(0))
            return torch.device("cuda"), "cuda"
        return torch.device("cpu"), "cpu"

    def _print_config_summary(self):
        print("\n" + "=" * 70)
        print("TRAINING CONFIGURATION (MULTI-LABEL — 3 CLASSES)")
        print("=" * 70)
        for k, v in [
            ("Device",       f"{self.device} ({self.device_type})"),
            ("Model",        config.MODEL_TYPE),
            ("Classes",      config.CLASS_NAMES),
            ("IMG_SIZE",     config.IMG_SIZE),
            ("Batch",        config.BATCH_SIZE),
            ("Epochs",       config.NUM_EPOCHS),
            ("LR",           config.LEARNING_RATE),
            ("Optimizer",    config.OPTIMIZER),
            ("Loss",         "Combined (BCE + Dice + Focal)"),
            ("AMP",          self.use_amp),
            ("Class Weights",config.BCE_POS_WEIGHT),
            ("Workers",      config.NUM_WORKERS),
        ]:
            print(f"  {k:<14}: {v}")
        print("=" * 70)

    def _get_criterion(self):
        print("📊 Loss: Combined (BCE + Dice + Focal)")
        return CombinedLoss(
            pos_weight=config.BCE_POS_WEIGHT,
            smooth=config.DICE_SMOOTH,
            w_bce=getattr(config, 'W_BCE', 1.0),
            w_dice=getattr(config, 'W_DICE', 1.0),
            w_focal=getattr(config, 'W_FOCAL', 0.5),
            focal_alpha=getattr(config, 'FOCAL_ALPHA', 0.25),
            focal_gamma=getattr(config, 'FOCAL_GAMMA', 2.0),
        )

    def _get_optimizer(self):
        name = config.OPTIMIZER.lower()
        lr = config.LEARNING_RATE
        wd = config.WEIGHT_DECAY
        if name == "adam":
            opt = optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        elif name == "adamw":
            opt = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        elif name == "sgd":
            opt = optim.SGD(self.model.parameters(), lr=lr,
                            momentum=getattr(config, "MOMENTUM", 0.9), weight_decay=wd)
        else:
            raise ValueError(f"Unknown optimizer: {config.OPTIMIZER}")
        print(f"⚙️  Optimizer: {config.OPTIMIZER.upper()}  LR={lr}")
        return opt

    def _get_scheduler(self):
        st = config.SCHEDULER_TYPE.lower()
        if st == "cosine":
            print("📈 Scheduler: CosineAnnealingLR")
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.NUM_EPOCHS)
        if st == "step":
            print("📈 Scheduler: StepLR")
            return optim.lr_scheduler.StepLR(
                self.optimizer, step_size=config.SCHEDULER_STEP_SIZE, gamma=config.SCHEDULER_FACTOR)
        if st == "plateau":
            print("📈 Scheduler: ReduceLROnPlateau")
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max",
                factor=config.SCHEDULER_FACTOR, patience=config.SCHEDULER_PATIENCE)
        raise ValueError(f"Unknown scheduler: {config.SCHEDULER_TYPE}")

    def _forward_and_loss(self, imgs, masks):
        out = self.model(imgs)
        if isinstance(out, tuple):
            main, aux = out
            main_loss = self.criterion(main, masks)
            aux_loss = self.criterion(aux, masks)
            loss = main_loss + AUX_LOSS_WEIGHT * aux_loss
            return main, loss
        else:
            loss = self.criterion(out, masks)
            return out, loss

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{config.NUM_EPOCHS} [TRAIN]")
        for imgs, masks in pbar:
            imgs = imgs.to(self.device)
            masks = masks.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            if self.use_amp:
                ctx = autocast('cuda') if USE_NEW_AMP_API else autocast()
                with ctx:
                    _, loss = self._forward_and_loss(imgs, masks)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                _, loss = self._forward_and_loss(imgs, masks)
                loss.backward()
                self.optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        return total_loss / max(1, len(self.train_loader))

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        self.metrics.reset()
        total_loss = 0.0
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{config.NUM_EPOCHS} [VAL]")
        for imgs, masks in pbar:
            imgs = imgs.to(self.device)
            masks = masks.to(self.device)
            logits = self.model(imgs)
            loss = self.criterion(logits, masks)
            total_loss += loss.item()
            self.metrics.update(logits, masks)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        metrics = self.metrics.get_metrics()
        return total_loss / max(1, len(self.val_loader)), metrics

    def save_best(self, epoch, miou):
        path = os.path.join(config.SAVE_DIR, "checkpoint_best_multilabel.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_miou": float(miou),
            "num_labels": self.num_labels,
        }, path)
        print(f"💾 Saved BEST checkpoint (mIoU={miou:.4f}) → {path}")

    def train(self):
        print("\n" + "=" * 70)
        print(f"🚀 START TRAINING FROM EPOCH {self.start_epoch}")
        print("=" * 70)
        no_improve = 0
        for epoch in range(self.start_epoch, config.NUM_EPOCHS):
            t0 = time.time()
            tr_loss = self.train_epoch(epoch)
            val_loss, metrics = self.validate(epoch)
            miou = metrics.get("mean_IoU", 0.0)
            mf1 = metrics.get("mean_F1", 0.0)
            print(f"\n📊 Epoch {epoch} Summary:")
            print(f"   Train Loss : {tr_loss:.4f}  |  Val Loss : {val_loss:.4f}")
            print(f"   mIoU       : {miou:.4f}  |  mF1 : {mf1:.4f}")
            print(f"   LR         : {self.optimizer.param_groups[0]['lr']:.6f}  |  Time : {time.time()-t0:.1f}s")
            print("   Per-class IoU & F1:")
            for cls in metrics['per_class']:
                if cls['valid']:
                    print(f"     {cls['name']:12s}: IoU={cls['IoU']:.4f}  F1={cls['F1']:.4f}")
            if miou > self.best_miou:
                self.best_miou = miou
                no_improve = 0
                self.save_best(epoch, miou)
            else:
                no_improve += 1
            if self.scheduler is not None:
                if config.SCHEDULER_TYPE.lower() == "plateau":
                    self.scheduler.step(miou)
                else:
                    self.scheduler.step()
            if no_improve >= config.EARLY_STOPPING_PATIENCE:
                print(f"\n⚠️  Early stopping — no improvement for {no_improve} epochs.")
                break
        print(f"\n✅ Training complete.  Best mIoU: {self.best_miou:.4f}")

if __name__ == "__main__":
    Trainer().train()