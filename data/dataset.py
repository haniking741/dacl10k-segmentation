"""
DACL10K Dataset Loader for Semantic Segmentation
Loads images and corresponding masks for U-Net training
"""
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch
import random


class DACL10KDataset(Dataset):
    """
    DACL10K Dataset for semantic segmentation
    
    Args:
        img_dir: Path to images folder (e.g., 'dataset/images/train')
        mask_dir: Path to masks folder (e.g., 'dataset/masks/train')
        num_classes: Number of classes (19 for DACL10K + background = 20)
        transform: Data augmentation (True for training, False for validation)
        img_size: Input image size (height, width)
    """
    
    def __init__(self, img_dir, mask_dir, num_classes=20, transform=True, img_size=(512, 512)):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.num_classes = num_classes
        self.transform = transform
        self.img_size = img_size
        
        # Get all image files
        self.images = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
        
        print(f"📂 Found {len(self.images)} images in {img_dir}")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        # Load mask (PNG with same name)
        mask_name = os.path.splitext(img_name)[0] + '.png'
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = Image.open(mask_path).convert("L")
  # Single channel, values 0-19
        
        # Apply transforms
        if self.transform:
            image, mask = self._apply_transforms(image, mask)
        else:
            # Just resize for validation
            image = TF.resize(image, self.img_size)
            mask = TF.resize(mask, self.img_size, interpolation=Image.NEAREST)
        
        # Convert to tensors
        image = TF.to_tensor(image)  # [3, H, W], range [0, 1]
        mask = torch.from_numpy(np.array(mask)).long()  # [H, W], range [0, num_classes-1]
        
        # Normalize image (ImageNet stats)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        return image, mask
    
    def _apply_transforms(self, image, mask):
        """
        Apply synchronized transforms to image and mask
        Data augmentation for training
        """
        # Resize
        image = TF.resize(image, self.img_size)
        mask = TF.resize(mask, self.img_size, interpolation=Image.NEAREST)
        
        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        
        # Random vertical flip
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        
        # Random rotation (-15 to +15 degrees)
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle, interpolation=Image.NEAREST)
        
        # Color jitter (only on image, not mask)
        if random.random() > 0.5:
            image = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)(image)
        
        return image, mask


class DACL10KDatasetCPU(DACL10KDataset):
    """
    CPU-optimized version with smaller image size and fewer augmentations
    Use this for local testing on CPU
    """
    
    def __init__(self, img_dir, mask_dir, num_classes=20, transform=True, img_size=(256, 256)):
        super().__init__(img_dir, mask_dir, num_classes, transform, img_size)
        print("⚡ Using CPU-optimized dataset (256x256)")
    
    def _apply_transforms(self, image, mask):
        """Minimal augmentation for CPU training"""
        # Resize
        image = TF.resize(image, self.img_size)
        mask = TF.resize(mask, self.img_size, interpolation=Image.NEAREST)
        
        # Only horizontal flip (fastest augmentation)
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        
        return image, mask


def get_dataloaders(data_root, batch_size=8, num_workers=4, img_size=(512, 512), cpu_mode=False):
    """
    Create train and validation dataloaders
    
    Args:
        data_root: Path to dataset root (contains images/ and masks/ folders)
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        img_size: Input image size (height, width)
        cpu_mode: Use CPU-optimized smaller images (256x256) and batch_size=2
    
    Returns:
        train_loader, val_loader, num_classes
    """
    
    if cpu_mode:
        img_size = (256, 256)
        batch_size = 2
        num_workers = 2
        DatasetClass = DACL10KDatasetCPU
        print("🐌 CPU MODE: Using 256x256 images, batch_size=2")
    else:
        DatasetClass = DACL10KDataset
        print(f"🚀 GPU MODE: Using {img_size} images, batch_size={batch_size}")
    
    train_img_dir = os.path.join(data_root, "images", "train")
    train_mask_dir = os.path.join(data_root, "masks", "train")
    val_img_dir = os.path.join(data_root, "images", "val")
    val_mask_dir = os.path.join(data_root, "masks", "val")
    
    # Create datasets
    train_dataset = DatasetClass(train_img_dir, train_mask_dir, 
                                  num_classes=20, transform=True, img_size=img_size)
    val_dataset = DatasetClass(val_img_dir, val_mask_dir, 
                                num_classes=20, transform=False, img_size=img_size)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=False if cpu_mode else True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=False if cpu_mode else True
    )
    
    print(f"✅ Train: {len(train_dataset)} images, {len(train_loader)} batches")
    print(f"✅ Val:   {len(val_dataset)} images, {len(val_loader)} batches")
    
    return train_loader, val_loader, train_dataset.num_classes


if __name__ == "__main__":
    # Test the dataset
    print("Testing dataset loader...")
    
    # Test with your actual paths
    train_loader, val_loader, num_classes = get_dataloaders(
        data_root=r"C:\Users\Informatics\Desktop\dataset_mémoire\segmentation_project\dataset",
        batch_size=2,
        num_workers=0,  # Use 0 for testing
        img_size=(256, 256),
        cpu_mode=True
    )
    
    # Load one batch
    images, masks = next(iter(train_loader))
    print(f"\n📊 Batch shapes:")
    print(f"  Images: {images.shape}  (batch, channels, height, width)")
    print(f"  Masks:  {masks.shape}   (batch, height, width)")
    print(f"  Mask values: min={masks.min()}, max={masks.max()}")
    print(f"  Num classes: {num_classes}")
    print("\n✅ Dataset loader working correctly!")