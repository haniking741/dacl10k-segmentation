"""
DACL10K Dataset Loader for Semantic Segmentation
Loads images and corresponding masks for U-Net training
+ Defect-focused cropping to fight background collapse

KEY FIX:
✅ Crop BEFORE resize (crop on original resolution)
✅ Stronger crop_ratio (default 0.60)
✅ Stronger min_defect_ratio (default 0.01)
✅ After crop -> resize to img_size -> then augmentations
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
        num_classes: Number of classes (19 defects + background = 20)
        transform: Data augmentation (True for training, False for validation)
        img_size: Input image size (height, width)
        defect_crop_prob: probability to do defect-focused crop (e.g. 0.7)
        crop_ratio: crop size ratio vs img_size (e.g. 0.6 -> crop smaller then resize up)
        crop_tries: number of tries to find a defect crop
        min_defect_ratio: minimum defect pixels ratio inside crop (e.g. 0.01 = 1%)
    """

    def __init__(
        self,
        img_dir,
        mask_dir,
        num_classes=20,
        transform=True,
        img_size=(512, 512),
        defect_crop_prob=0.7,
        crop_ratio=0.60,
        crop_tries=10,
        min_defect_ratio=0.01,
    ):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.num_classes = num_classes
        self.transform = transform
        self.img_size = img_size

        # Defect-focused cropping config (TRAIN ONLY)
        self.defect_crop_prob = float(defect_crop_prob)
        self.crop_ratio = float(crop_ratio)
        self.crop_tries = int(crop_tries)
        self.min_defect_ratio = float(min_defect_ratio)

        # Get all image files
        self.images = sorted([f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png"))])
        print(f"📂 Found {len(self.images)} images in {img_dir}")

    def __len__(self):
        return len(self.images)

    # ---------------------------
    # Cropping helpers (PIL)
    # ---------------------------
    def _random_crop_pil(self, image, mask, crop_h, crop_w):
        w, h = image.size # PIL: (W,H)
        if h <= crop_h or w <= crop_w:
            return image, mask

        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)

        image_c = TF.crop(image, top, left, crop_h, crop_w)
        mask_c = TF.crop(mask, top, left, crop_h, crop_w)
        return image_c, mask_c

    def _defect_focused_crop_pil(self, image, mask, crop_h, crop_w, tries=10, min_defect_ratio=0.01):
        """
        Try to crop around defect pixels (mask != 0).
        min_defect_ratio: minimum fraction of defect pixels inside crop
        """
        w, h = image.size
        if h <= crop_h or w <= crop_w:
            return image, mask

        mask_np = np.array(mask, dtype=np.int64)
        ys, xs = np.where(mask_np != 0)

        if ys.size == 0:
            # no defects -> fallback to random crop
            return self._random_crop_pil(image, mask, crop_h, crop_w)

        for _ in range(tries):
            i = random.randint(0, ys.size - 1)
            y, x = int(ys[i]), int(xs[i])

            top = y - crop_h // 2
            left = x - crop_w // 2

            top = max(0, min(top, h - crop_h))
            left = max(0, min(left, w - crop_w))

            image_c = TF.crop(image, top, left, crop_h, crop_w)
            mask_c = TF.crop(mask, top, left, crop_h, crop_w)

            mask_c_np = np.array(mask_c, dtype=np.int64)
            defect_ratio = float((mask_c_np != 0).mean())

            if defect_ratio >= min_defect_ratio:
                return image_c, mask_c

        return self._random_crop_pil(image, mask, crop_h, crop_w)

    def __getitem__(self, idx):
        # Load image
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # Load mask (PNG with same name)
        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = Image.open(mask_path).convert("L") # values 0..19

        if self.transform:
            image, mask = self._apply_transforms(image, mask)
        else:
            # Validation: just resize (NO crop, NO aug)
            image = TF.resize(image, self.img_size)
            mask = TF.resize(mask, self.img_size, interpolation=Image.NEAREST)

        # Convert to tensors
        image = TF.to_tensor(image) # [3,H,W] float
        mask = torch.from_numpy(np.array(mask, dtype=np.int64)).long() # [H,W]

        # Normalize image (ImageNet stats)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return image, mask

    def _apply_transforms(self, image, mask):
        """
        TRAIN pipeline:

        ✅ 1) Crop on ORIGINAL size (defect-focused 70%)
        ✅ 2) Resize to img_size
        ✅ 3) Augmentations (flip/rotate/jitter) on final size
        """
        # -----------------------------------
        # 1) Defect-focused crop on ORIGINAL
        # -----------------------------------
        crop_h = max(64, int(self.img_size[0] * self.crop_ratio))
        crop_w = max(64, int(self.img_size[1] * self.crop_ratio))

        if random.random() < self.defect_crop_prob:
            image, mask = self._defect_focused_crop_pil(
                image,
                mask,
                crop_h,
                crop_w,
                tries=self.crop_tries,
                min_defect_ratio=self.min_defect_ratio,
            )
        else:
            image, mask = self._random_crop_pil(image, mask, crop_h, crop_w)

        # ----------------
        # 2) Resize to model size
        # ----------------
        image = TF.resize(image, self.img_size)
        mask = TF.resize(mask, self.img_size, interpolation=Image.NEAREST)

        # ----------------
        # 3) Augmentations
        # ----------------
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

        # Color jitter (image only)
        if random.random() > 0.5:
            image = T.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
            )(image)

        return image, mask


class DACL10KDatasetCPU(DACL10KDataset):
    """
    CPU-optimized version
    """

    def __init__(self, img_dir, mask_dir, num_classes=20, transform=True, img_size=(256, 256)):
        super().__init__(
            img_dir,
            mask_dir,
            num_classes=num_classes,
            transform=transform,
            img_size=img_size,
            defect_crop_prob=0.7,
            crop_ratio=0.60,
            crop_tries=10,
            min_defect_ratio=0.01,
        )
        print("⚡ Using CPU-optimized dataset")

    def _apply_transforms(self, image, mask):
        """
        CPU: keep it light (crop + resize + hflip only)
        """
        crop_h = max(64, int(self.img_size[0] * self.crop_ratio))
        crop_w = max(64, int(self.img_size[1] * self.crop_ratio))

        if random.random() < self.defect_crop_prob:
            image, mask = self._defect_focused_crop_pil(
                image, mask, crop_h, crop_w, tries=self.crop_tries, min_defect_ratio=self.min_defect_ratio
            )
        else:
            image, mask = self._random_crop_pil(image, mask, crop_h, crop_w)

        image = TF.resize(image, self.img_size)
        mask = TF.resize(mask, self.img_size, interpolation=Image.NEAREST)

        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        return image, mask


def get_dataloaders(data_root, batch_size=8, num_workers=4, img_size=(512, 512), cpu_mode=False):
    """
    Create train and validation dataloaders
    """
    if cpu_mode:
        img_size = (256, 256)
        batch_size = 2
        num_workers = 0
        DatasetClass = DACL10KDatasetCPU
        print("🐌 CPU MODE: Using 256x256 images, batch_size=2")
    else:
        DatasetClass = DACL10KDataset
        print(f"🚀 GPU MODE: Using {img_size} images, batch_size={batch_size}")

    train_img_dir = os.path.join(data_root, "images", "train")
    train_mask_dir = os.path.join(data_root, "masks", "train")
    val_img_dir = os.path.join(data_root, "images", "val")
    val_mask_dir = os.path.join(data_root, "masks", "val")

    # TRAIN: enable crop
    train_dataset = DatasetClass(
        train_img_dir,
        train_mask_dir,
        num_classes=20,
        transform=True,
        img_size=img_size,
        defect_crop_prob=0.7,
        crop_ratio=0.60, # ✅ stronger crop
        crop_tries=10,
        min_defect_ratio=0.01, # ✅ stronger minimum defect
    )

    # VAL: no crop / no aug
    val_dataset = DatasetClass(
        val_img_dir,
        val_mask_dir,
        num_classes=20,
        transform=False,
        img_size=img_size,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False if cpu_mode else True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False if cpu_mode else True,
    )

    print(f"✅ Train: {len(train_dataset)} images, {len(train_loader)} batches")
    print(f"✅ Val: {len(val_dataset)} images, {len(val_loader)} batches")

    return train_loader, val_loader, train_dataset.num_classes


if __name__ == "__main__":
    print("Testing dataset loader...")
    train_loader, val_loader, num_classes = get_dataloaders(
        data_root=r"C:\Users\Informatics\Desktop\dataset_mémoire\segmentation_project\dataset",
        batch_size=2,
        num_workers=0,
        img_size=(256, 256),
        cpu_mode=True,
    )
    images, masks = next(iter(train_loader))
    print(f"\n📊 Batch shapes:")
    print(f" Images: {images.shape}")
    print(f" Masks: {masks.shape}")
    print(f" Mask values: min={masks.min()}, max={masks.max()}")
    print(f" Num classes: {num_classes}")
    print("\n✅ Dataset loader working correctly!")