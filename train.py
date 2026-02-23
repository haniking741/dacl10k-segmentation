"""
Main Training Script for U-Net on DACL10K
Supports CPU, CUDA (NVIDIA), and DirectML (AMD/Intel) training
"""
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

# ✅ Device-agnostic AMP (works best for CUDA; we will disable AMP for DirectML/CPU)
from torch.amp import autocast, GradScaler

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from models.unet import get_model
from data.dataset import get_dataloaders
from utils.metrics import SegmentationMetrics


class Trainer:
    def __init__(self):
        """Initialize trainer"""
        # Set random seed
        torch.manual_seed(config.RANDOM_SEED)
        np.random.seed(config.RANDOM_SEED)

        # ---------------------------
        # Device selection (CPU/CUDA/DirectML)
        # ---------------------------
        self.device, self.device_type = self._get_device()

        # Create directories
        os.makedirs(config.SAVE_DIR, exist_ok=True)
        os.makedirs(config.LOG_DIR, exist_ok=True)

        # Print configuration
        config.get_config_summary()

        # Load data
        print("📂 Loading dataset...")
        self.train_loader, self.val_loader, self.num_classes = get_dataloaders(
            data_root=config.DATA_ROOT,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            img_size=config.IMG_SIZE,
            cpu_mode=config.CPU_MODE
        )

        # Create model
        print("\n📐 Creating model...")
        self.model = get_model(
            model_type=config.MODEL_TYPE,
            n_classes=self.num_classes,
            device=self.device
        )

        # Loss function
        self.criterion = self._get_criterion()

        # Optimizer
        self.optimizer = self._get_optimizer()

        # Learning rate scheduler
        self.scheduler = self._get_scheduler() if config.USE_SCHEDULER else None

        # ---------------------------
        # AMP: enable only on CUDA
        # ---------------------------
        self.use_amp = bool(config.USE_AMP) and (self.device_type == "cuda")
        self.scaler = GradScaler(enabled=self.use_amp)

        if self.use_amp:
            print("⚡ AMP enabled (CUDA)")
        else:
            print("ℹ️  AMP disabled (CPU/DirectML)")

        # Metrics
        self.metrics = SegmentationMetrics(
            num_classes=self.num_classes,
            class_names=config.CLASS_NAMES
        )

        # Training state
        self.start_epoch = 0
        self.best_miou = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_mious = []

        # Resume from checkpoint if specified
        if config.RESUME_CHECKPOINT and os.path.exists(config.RESUME_CHECKPOINT):
            self._load_checkpoint(config.RESUME_CHECKPOINT)

    def _get_device(self):
        """
        Return (device, device_type_string). 
        device_type_string in {'cpu','cuda','directml'}
        
        ✅ FIXED: Now tries to use GPU 1 (RX 6600) instead of GPU 0 (integrated)
        """
        if config.CPU_MODE:
            print("🐌 Running on CPU (slow, for testing only)")
            return torch.device("cpu"), "cpu"

        # 1) CUDA (NVIDIA)
        if torch.cuda.is_available():
            dev = torch.device("cuda")
            print(f"🚀 Running on CUDA GPU: {torch.cuda.get_device_name(0)}")
            return dev, "cuda"

        # 2) DirectML (AMD / Intel)
        try:
            import torch_directml
            if torch_directml.is_available():
                # ✅ FIX: Try to use GPU 1 (RX 6600) instead of GPU 0 (integrated)
                device_count = torch_directml.device_count()
                print(f"🔍 Found {device_count} DirectML device(s)")
                
                if device_count > 1:
                    # Multiple GPUs detected, prefer GPU 1 (RX 6600)
                    dev = torch_directml.device(1)
                    print("🎮 Using GPU 1 (RX 6600 - Discrete GPU)")
                else:
                    # Only one GPU, use it
                    dev = torch_directml.device(0)
                    print("🚀 Running on DirectML GPU 0")
                
                return dev, "directml"
            else:
                print("⚠️ DirectML not available, falling back to CPU")
                return torch.device("cpu"), "cpu"
        except Exception as e:
            print(f"⚠️ torch_directml failed ({e}), falling back to CPU")
            return torch.device("cpu"), "cpu"

    def _get_criterion(self):
        """Create loss function (supports CE / Weighted CE / Dice / Focal + combos)"""

        loss_type = config.LOSS_TYPE.lower()

        # Import losses
        from utils.losses import FocalLoss, DiceLoss, CombinedLoss

        # ---------
        # Helpers
        # ---------
        def make_ce(weighted: bool):
            if weighted and (config.CLASS_WEIGHTS is not None):
                weights = torch.tensor(config.CLASS_WEIGHTS, dtype=torch.float32).to(self.device)
                return nn.CrossEntropyLoss(weight=weights)
            return nn.CrossEntropyLoss()

        def make_focal():
            # If you want per-class alpha, set FOCAL_ALPHA in config as list of length C
            return FocalLoss(alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA, ignore_index=None)

        def make_dice():
            return DiceLoss(ignore_index=None)

        # -------------------
        # Single-loss options
        # -------------------
        if loss_type == "ce":
            criterion = make_ce(weighted=False)
            print("📊 Using CrossEntropyLoss")

        elif loss_type == "wce":
            criterion = make_ce(weighted=True)
            print("📊 Using Weighted CrossEntropyLoss")

        elif loss_type == "dice":
            criterion = make_dice()
            print("📊 Using Dice Loss")

        elif loss_type == "focal":
            criterion = make_focal()
            print("📊 Using Focal Loss")

        # -------------------
        # Combined-loss options
        # -------------------
        elif loss_type == "ce_dice":
            ce = make_ce(weighted=False)
            dice = make_dice()
            criterion = CombinedLoss(ce, dice, w1=1.0, w2=1.0)
            print("📊 Using CE + Dice Loss")

        elif loss_type == "wce_dice":
            ce = make_ce(weighted=True)
            dice = make_dice()
            criterion = CombinedLoss(ce, dice, w1=1.0, w2=1.0)
            print("📊 Using Weighted CE + Dice Loss")

        elif loss_type == "focal_dice":
            focal = make_focal()
            dice = make_dice()
            criterion = CombinedLoss(focal, dice, w1=1.0, w2=1.0)
            print("📊 Using Focal + Dice Loss")
        elif loss_type == "wce":
            criterion = make_ce(weighted=True)  # ✅ This uses CLASS_WEIGHTS
            print("📊 Using Weighted CrossEntropyLoss")
        else:
            raise ValueError(
                f"Unknown LOSS_TYPE: {config.LOSS_TYPE}. "
                "Use one of: ce, wce, dice, focal, ce_dice, wce_dice, focal_dice"
            )

        return criterion

    def _get_optimizer(self):
        """Create optimizer"""
        if config.OPTIMIZER == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=config.LEARNING_RATE,
                weight_decay=config.WEIGHT_DECAY
            )
        elif config.OPTIMIZER == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=config.LEARNING_RATE,
                momentum=config.MOMENTUM,
                weight_decay=config.WEIGHT_DECAY
            )
        else:
            raise ValueError(f"Unknown optimizer: {config.OPTIMIZER}")

        print(f"⚙️  Optimizer: {config.OPTIMIZER.upper()}, LR={config.LEARNING_RATE}")
        return optimizer

    def _get_scheduler(self):
        """Create learning rate scheduler"""
        if config.SCHEDULER_TYPE == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.NUM_EPOCHS)
        elif config.SCHEDULER_TYPE == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.SCHEDULER_STEP_SIZE,
                gamma=config.SCHEDULER_FACTOR
            )
        elif config.SCHEDULER_TYPE == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=config.SCHEDULER_FACTOR,
                patience=config.SCHEDULER_PATIENCE
            )
        else:
            raise ValueError(f"Unknown scheduler type: {config.SCHEDULER_TYPE}")

        print(f"📈 Scheduler: {config.SCHEDULER_TYPE}")
        return scheduler

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{config.NUM_EPOCHS} [TRAIN]")

        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()

            # Forward + loss
            if self.use_amp:
                with autocast(device_type="cuda", enabled=True):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        return epoch_loss / max(1, len(self.train_loader))

    @torch.no_grad()
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
        self.metrics.reset()

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{config.NUM_EPOCHS} [VAL]")

        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            # ================= DEBUG CHECK (FIRST BATCH ONLY) =================
# ================= DEBUG CHECK (FIRST BATCH ONLY) =================
            if pbar.n == 0:
                 preds_cpu = preds.detach().cpu()
                 masks_cpu = masks.detach().cpu()
                 bg_pred = (preds_cpu == 0).float().mean().item()
                 bg_true = (masks_cpu == 0).float().mean().item()
                 print("\n========== DEBUG ==========")
                 print(f"% predicted background: {bg_pred*100:.2f}%")
                 print(f"% true background:      {bg_true*100:.2f}%")
                 print("Unique predicted labels:", torch.unique(preds_cpu).numpy())
                 print("Unique true labels:     ", torch.unique(masks_cpu).numpy())
                 print("================================\n")
# ================================================================

            self.metrics.update(preds, masks)
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        metrics = self.metrics.get_metrics()
        avg_loss = val_loss / max(1, len(self.val_loader))
        return avg_loss, metrics

    def save_checkpoint(self, epoch, miou, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_miou': self.best_miou,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_mious': self.val_mious,
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        latest_path = os.path.join(config.SAVE_DIR, 'checkpoint_latest.pth')
        torch.save(checkpoint, latest_path)

        if is_best:
            best_path = os.path.join(config.SAVE_DIR, 'checkpoint_best.pth')
            torch.save(checkpoint, best_path)
            print(f"💾 Saved best model (mIoU: {miou:.4f})")

    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint"""
        print(f"📥 Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.start_epoch = checkpoint['epoch'] + 1
        self.best_miou = checkpoint.get('best_miou', 0.0)
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_mious = checkpoint.get('val_mious', [])

        print(f"✅ Resumed from epoch {self.start_epoch}, best mIoU: {self.best_miou:.4f}")

    def train(self):
        """Main training loop"""
        print("\n" + "="*70)
        print("🚀 STARTING TRAINING")
        print("="*70 + "\n")

        no_improvement = 0

        for epoch in range(self.start_epoch, config.NUM_EPOCHS):
            epoch_start = time.time()

            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)

            if (epoch + 1) % config.VAL_FREQUENCY == 0:
                val_loss, metrics = self.validate(epoch)
                self.val_losses.append(val_loss)
                self.val_mious.append(metrics['mIoU'])

                print(f"\n📊 Epoch {epoch}/{config.NUM_EPOCHS} Summary:")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss:   {val_loss:.4f}")
                print(f"  mIoU:       {metrics['mIoU']:.4f}")
                print(f"  F1-score:   {metrics['mean_F1']:.4f}")
                print(f"  Pixel Acc:  {metrics['Pixel_Accuracy']:.4f}")
                print(f"  Time:       {time.time() - epoch_start:.1f}s")

                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"  LR:         {current_lr:.6f}\n")

                is_best = metrics['mIoU'] > self.best_miou
                if is_best:
                    self.best_miou = metrics['mIoU']
                    no_improvement = 0
                else:
                    no_improvement += 1

                if config.SAVE_BEST_ONLY:
                    if is_best:
                        self.save_checkpoint(epoch, metrics['mIoU'], is_best=True)
                else:
                    self.save_checkpoint(epoch, metrics['mIoU'], is_best=is_best)

                if no_improvement >= config.EARLY_STOPPING_PATIENCE:
                    print(f"\n⚠️ Early stopping: No improvement for {config.EARLY_STOPPING_PATIENCE} epochs")
                    break

                if self.scheduler is not None:
                    if config.SCHEDULER_TYPE == 'plateau':
                        self.scheduler.step(metrics['mIoU'])
                    else:
                        self.scheduler.step()

        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE!")
        print("="*70)
        print(f"Best mIoU: {self.best_miou:.4f}")
        print(f"Checkpoints saved in: {config.SAVE_DIR}")
        print("="*70 + "\n")


def main():
    trainer = Trainer()
    trainer.train()


if __name__ == "__main__":
    main()