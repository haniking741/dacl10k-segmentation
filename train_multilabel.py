"""
Training Script (MULTI-LABEL) for DACL10K U-Net
- Uses dataset2/images + dataset2/masks_multilabel
- Output channels = 19 (defects only, no background channel)
- Supports CPU / CUDA (NVIDIA) / DirectML (AMD/Intel on Windows)
"""

import os
import time
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from torch.amp import autocast, GradScaler

import config
from models.unet import get_model
from data.dataset_multilabel import get_dataloaders_multilabel
from utils.losses_multilabel import MultiLabelDiceLoss
from utils.metrics_multilabel import MultiLabelSegmentationMetrics


class Trainer:
    def __init__(self):
        torch.manual_seed(config.RANDOM_SEED)
        np.random.seed(config.RANDOM_SEED)

        # ---------------------------
        # Device selection
        # ---------------------------
        self.device, self.device_type = self._get_device()

        # ---------------------------
        # AMP: only effective on CUDA
        # ---------------------------
        self.use_amp = bool(getattr(config, "USE_AMP", False)) and (self.device_type == "cuda")
        self.scaler = GradScaler(enabled=self.use_amp)

        # Create dirs
        os.makedirs(config.SAVE_DIR, exist_ok=True)
        os.makedirs(getattr(config, "LOG_DIR", "logs"), exist_ok=True)

        # Print config summary (NOW safe: use_amp exists)
        self._print_config_summary()

        # ---------------------------
        # Dataloaders
        # ---------------------------
        print("\n📂 Loading dataset (MULTI-LABEL)...")
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

        # ---------------------------
        # Model
        # ---------------------------
        print("\n📐 Creating model...")
        self.model = get_model(config.MODEL_TYPE, self.num_labels, self.device)

        # ---------------------------
        # Loss / Optim / Scheduler
        # ---------------------------
        self.criterion = self._get_criterion()
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler() if config.USE_SCHEDULER else None

        # Metrics
        # class_names should match NUM_LABELS (19) (defects only)
        self.metrics = MultiLabelSegmentationMetrics(
            num_classes=self.num_labels,
            class_names=getattr(config, "LABEL_NAMES", None), # optional
        )

        self.best_miou = 0.0

        if self.use_amp:
            print("⚡ AMP enabled (CUDA)")
        else:
            print("ℹ️ AMP disabled (CPU/DirectML)")

    # ==========================================================
    # DEVICE
    # ==========================================================
    def _get_device(self):
        # CPU mode forced
        if getattr(config, "CPU_MODE", False):
            print("🐌 CPU MODE forced")
            return torch.device("cpu"), "cpu"

        # CUDA (NVIDIA)
        if torch.cuda.is_available():
            print("🚀 Using CUDA:", torch.cuda.get_device_name(0))
            return torch.device("cuda"), "cuda"

        # DirectML (AMD/Intel on Windows)
        try:
            import torch_directml
            if torch_directml.is_available():
                n = torch_directml.device_count()
                print(f"🔍 Found {n} DirectML device(s)")
                dev = torch_directml.device(getattr(config, "GPU_ID", 0))
                print(f"🎮 Using DirectML GPU {getattr(config, 'GPU_ID', 0)}")
                return dev, "directml"
        except Exception as e:
            print(f"⚠️ torch_directml not available ({e})")

        print("⚠️ No CUDA/DirectML found, using CPU")
        return torch.device("cpu"), "cpu"

    # ==========================================================
    # PRINT CONFIG
    # ==========================================================
    def _print_config_summary(self):
        print("\n" + "=" * 70)
        print("TRAINING CONFIGURATION (MULTI-LABEL)")
        print("=" * 70)
        print(f"Device: {self.device} (type={self.device_type})")
        print(f"Model: {config.MODEL_TYPE}")
        print(f"IMG_SIZE: {config.IMG_SIZE}")
        print(f"BATCH: {config.BATCH_SIZE}")
        print(f"EPOCHS: {config.NUM_EPOCHS}")
        print(f"LR: {config.LEARNING_RATE}")
        print(f"OPT: {config.OPTIMIZER}")
        print(f"LOSS: {config.LOSS_TYPE}")
        print(f"AMP: {getattr(config, 'USE_AMP', False)} (effective={self.use_amp})")
        print(f"NUM_LABELS: {getattr(config, 'NUM_LABELS', 'auto')}")
        print(f"NUM_WORKERS: {config.NUM_WORKERS}")
        print(f"DATA_ROOT: {config.DATA_ROOT}")
        print("=" * 70)

    # ==========================================================
    # LOSS
    # ==========================================================
    def _get_criterion(self):
        lt = config.LOSS_TYPE.lower()

        # BCE helper (supports pos_weight)
        def bce_loss(logits, targets):
            posw = getattr(config, "BCE_POS_WEIGHT", None)
            posw_t = None
            if posw is not None:
                posw_t = torch.tensor(posw, dtype=torch.float32, device=logits.device)
            return torch.nn.functional.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=posw_t
            )

        if lt == "bce":
            print("📊 Loss: BCEWithLogits")
            return bce_loss

        if lt == "dice":
            print("📊 Loss: MultiLabel Dice")
            return MultiLabelDiceLoss()

        if lt == "bce_dice":
            print("📊 Loss: BCE + Dice")

            dice = MultiLabelDiceLoss()

            def loss_fn(logits, targets):
                return 1.0 * bce_loss(logits, targets) + 1.0 * dice(logits, targets)

            return loss_fn

        raise ValueError(f"LOSS_TYPE must be: bce | dice | bce_dice, got {config.LOSS_TYPE}")

    # ==========================================================
    # OPTIMIZER
    # ==========================================================
    def _get_optimizer(self):
        if config.OPTIMIZER.lower() == "adam":
            opt = optim.Adam(
                self.model.parameters(),
                lr=config.LEARNING_RATE,
                weight_decay=config.WEIGHT_DECAY
            )
        elif config.OPTIMIZER.lower() == "sgd":
            opt = optim.SGD(
                self.model.parameters(),
                lr=config.LEARNING_RATE,
                momentum=getattr(config, "MOMENTUM", 0.9),
                weight_decay=config.WEIGHT_DECAY
            )
        else:
            raise ValueError(f"Unknown OPTIMIZER: {config.OPTIMIZER}")

        print(f"⚙️ Optimizer: {config.OPTIMIZER.upper()}, LR={config.LEARNING_RATE}")
        return opt

    # ==========================================================
    # SCHEDULER
    # ==========================================================
    def _get_scheduler(self):
        st = config.SCHEDULER_TYPE.lower()
        if st == "cosine":
            print("📈 Scheduler: cosine")
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.NUM_EPOCHS)
        if st == "step":
            print("📈 Scheduler: step")
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.SCHEDULER_STEP_SIZE,
                gamma=config.SCHEDULER_FACTOR
            )
        if st == "plateau":
            print("📈 Scheduler: plateau")
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=config.SCHEDULER_FACTOR,
                patience=config.SCHEDULER_PATIENCE
            )
        raise ValueError(f"Unknown scheduler type: {config.SCHEDULER_TYPE}")

    # ==========================================================
    # TRAIN / VAL
    # ==========================================================
    def train_epoch(self, epoch):
        self.model.train()
        total = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{config.NUM_EPOCHS} [TRAIN]")

        for imgs, masks in pbar:
            imgs = imgs.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with autocast(device_type="cuda", enabled=True):
                    logits = self.model(imgs)
                    loss = self.criterion(logits, masks)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(imgs)
                loss = self.criterion(logits, masks)
                loss.backward()
                self.optimizer.step()

            total += float(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total / max(1, len(self.train_loader))

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        self.metrics.reset()
        total = 0.0

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{config.NUM_EPOCHS} [VAL]")
        for imgs, masks in pbar:
            imgs = imgs.to(self.device)
            masks = masks.to(self.device)

            logits = self.model(imgs)
            loss = self.criterion(logits, masks)
            total += float(loss.item())

            self.metrics.update(logits, masks)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        metrics = self.metrics.get_metrics()
        return total / max(1, len(self.val_loader)), metrics

    # ==========================================================
    # CHECKPOINT
    # ==========================================================
    def save_best(self, epoch, miou):
        path = os.path.join(config.SAVE_DIR, "checkpoint_best_multilabel.pth")
        torch.save(
            {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "best_miou": self.best_miou,
            },
            path
        )
        print(f"💾 Saved BEST multilabel model (mIoU={miou:.4f}) -> {path}")

    # ==========================================================
    # MAIN LOOP
    # ==========================================================
    def train(self):
        print("\n" + "=" * 70)
        print("🚀 START MULTI-LABEL TRAINING")
        print("=" * 70)

        no_imp = 0

        for epoch in range(config.NUM_EPOCHS):
            t0 = time.time()
            tr_loss = self.train_epoch(epoch)

            val_loss, metrics = self.validate(epoch)
            miou = metrics.get("mIoU", 0.0)

            print(f"\n📊 Epoch {epoch} Summary:")
            print(f" Train Loss: {tr_loss:.4f}")
            print(f" Val Loss: {val_loss:.4f}")
            print(f" mIoU: {miou:.4f}")
            if "mean_F1" in metrics:
                print(f" mean_F1: {metrics['mean_F1']:.4f}")
            print(f" Time: {time.time() - t0:.1f}s")
            print(f" LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            improved = miou > self.best_miou
            if improved:
                self.best_miou = miou
                no_imp = 0
                self.save_best(epoch, miou)
            else:
                no_imp += 1

            if self.scheduler is not None:
                if config.SCHEDULER_TYPE.lower() == "plateau":
                    self.scheduler.step(miou)
                else:
                    self.scheduler.step()

            if no_imp >= config.EARLY_STOPPING_PATIENCE:
                print("⚠️ Early stopping (no improvement)")
                break

        print("\n✅ DONE. Best mIoU:", self.best_miou)


if __name__ == "__main__":
    Trainer().train()