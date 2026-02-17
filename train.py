"""
Main Training Script for U-Net on DACL10K
Supports both CPU and GPU training
"""
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from models.unet import get_model
from data.dataset import get_dataloaders
from utils.metrics import SegmentationMetrics, compute_iou


class Trainer:
    def __init__(self):
        """Initialize trainer"""
        # Set random seed
        torch.manual_seed(config.RANDOM_SEED)
        np.random.seed(config.RANDOM_SEED)
        
        # Device
        if config.CPU_MODE:
            self.device = torch.device('cpu')
            print("🐌 Running on CPU (slow, for testing only)")
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if self.device.type == 'cuda':
                print(f"🚀 Running on GPU: {torch.cuda.get_device_name(0)}")
            else:
                print("⚠️  No GPU found, falling back to CPU")
        
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
        
        # Mixed precision scaler
        self.scaler = GradScaler() if config.USE_AMP else None
        
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
    
    def _get_criterion(self):
        """Create loss function"""
        if config.LOSS_TYPE == 'ce':
            # Standard Cross Entropy
            if config.CLASS_WEIGHTS is not None:
                weights = torch.tensor(config.CLASS_WEIGHTS, dtype=torch.float32).to(self.device)
                criterion = nn.CrossEntropyLoss(weight=weights)
            else:
                criterion = nn.CrossEntropyLoss()
            print("📊 Using CrossEntropyLoss")
        
        elif config.LOSS_TYPE == 'focal':
            # Focal Loss (for class imbalance)
            from utils.losses import FocalLoss
            criterion = FocalLoss(alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA)
            print("📊 Using Focal Loss")
        
        elif config.LOSS_TYPE == 'dice':
            # Dice Loss
            from utils.losses import DiceLoss
            criterion = DiceLoss()
            print("📊 Using Dice Loss")
        
        else:
            raise ValueError(f"Unknown loss type: {config.LOSS_TYPE}")
        
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
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.NUM_EPOCHS
            )
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
            raise ValueError(f"Unknown scheduler: {config.SCHEDULER_TYPE}")
        
        print(f"📈 Scheduler: {config.SCHEDULER_TYPE}")
        return scheduler
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{config.NUM_EPOCHS} [TRAIN]")
        
        for batch_idx, (images, masks) in enumerate(pbar):
            # Move to device
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            if config.USE_AMP:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if config.USE_AMP:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / len(self.train_loader)
        return avg_loss
    
    @torch.no_grad()
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
        self.metrics.reset()
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{config.NUM_EPOCHS} [VAL]  ")
        
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            val_loss += loss.item()
            
            # Get predictions
            preds = torch.argmax(outputs, dim=1)
            
            # Update metrics
            self.metrics.update(preds, masks)
            
            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Compute metrics
        metrics = self.metrics.get_metrics()
        avg_loss = val_loss / len(self.val_loader)
        
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
        
        # Save latest checkpoint
        latest_path = os.path.join(config.SAVE_DIR, 'checkpoint_latest.pth')
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
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
            
            # Train
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            if (epoch + 1) % config.VAL_FREQUENCY == 0:
                val_loss, metrics = self.validate(epoch)
                self.val_losses.append(val_loss)
                self.val_mious.append(metrics['mIoU'])
                
                # Print results
                print(f"\n📊 Epoch {epoch}/{config.NUM_EPOCHS} Summary:")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss:   {val_loss:.4f}")
                print(f"  mIoU:       {metrics['mIoU']:.4f}")
                print(f"  F1-score:   {metrics['mean_F1']:.4f}")
                print(f"  Pixel Acc:  {metrics['Pixel_Accuracy']:.4f}")
                print(f"  Time:       {time.time() - epoch_start:.1f}s")
                
                # Learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"  LR:         {current_lr:.6f}\n")
                
                # Check if best model
                is_best = metrics['mIoU'] > self.best_miou
                if is_best:
                    self.best_miou = metrics['mIoU']
                    no_improvement = 0
                else:
                    no_improvement += 1
                
                # Save checkpoint
                if config.SAVE_BEST_ONLY:
                    if is_best:
                        self.save_checkpoint(epoch, metrics['mIoU'], is_best=True)
                else:
                    self.save_checkpoint(epoch, metrics['mIoU'], is_best=is_best)
                
                # Early stopping
                if no_improvement >= config.EARLY_STOPPING_PATIENCE:
                    print(f"\n⚠️  Early stopping: No improvement for {config.EARLY_STOPPING_PATIENCE} epochs")
                    break
                
                # Update scheduler
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
    """Main function"""
    trainer = Trainer()
    trainer.train()


if __name__ == "__main__":
    main()