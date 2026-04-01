"""
Multi-label Dataset Loader for DACL10K (3 CLASSES)
Loads only: crack (1), spalling (7), rust (11)

FIXES APPLIED:
  [BUG #1] Noise is now applied BEFORE normalization, clamped to [0,1]
           → Previously, clamp(0,1) after normalize() was destroying the
             ImageNet-normalized range (~[-2.1, 2.6]), corrupting 20% of batches.
  [BUG #6] Empty fallback mask now uses actual image.size instead of
           hardcoded (512,512) → prevents shape mismatch before resize.
  [CRACK FIX] Replaced NEAREST with BILINEAR interpolation for masks to prevent
              thin lines from vanishing during resize/rotate.
  [CRACK FIX] Added morphological dilation (max_pool2d) to thicken the crack mask
              by 1 pixel before returning it to the training loop.
"""

import os
import random
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class DACL10KMultiLabelDataset(Dataset):
    def __init__(
        self,
        img_dir,
        mask_dir,
        classes_to_load=None,
        transform=True,
        img_size=(512, 512),
        defect_crop_prob=0.7,
        crop_ratio=0.60,
        crop_tries=10,
        min_defect_ratio=0.01,
        low_res_union=(256, 256),
    ):
        self.img_dir = img_dir
        self.mask_dir = mask_dir

        import config
        if classes_to_load is None:
            classes_to_load = getattr(config, 'CLASSES_TO_LOAD', [1, 7, 11])
        self.classes_to_load = list(classes_to_load)
        self.num_labels = len(self.classes_to_load)

        self.transform = bool(transform)
        self.img_size = tuple(img_size)
        self.defect_crop_prob = float(defect_crop_prob)
        self.crop_ratio = float(crop_ratio)
        self.crop_tries = int(crop_tries)
        self.min_defect_ratio = float(min_defect_ratio)
        self.low_res_union = tuple(low_res_union)

        self.images = sorted(
            [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png"))]
        )
        print(f"📂 Found {len(self.images)} images in {img_dir}")
        print(f"✅ Loading {self.num_labels} classes: {self.classes_to_load}")

    def __len__(self):
        return len(self.images)

    def _load_multilabel_masks(self, base_name, image_size):
        """
        Load only the specified classes.
        image_size: (W, H) from image.size — used for fallback empty mask.
        """
        masks = []
        for class_id in self.classes_to_load:
            fn = f"{base_name}_class{class_id:02d}.png"
            fp = os.path.join(self.mask_dir, fn)

            if os.path.exists(fp):
                m = Image.open(fp).convert("L")
            else:
                m = Image.new('L', image_size, 0)

            masks.append(m)
        return masks

    def _random_crop(self, image, masks, crop_h, crop_w):
        w, h = image.size
        if h <= crop_h or w <= crop_w:
            return image, masks
        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)
        image_c = TF.crop(image, top, left, crop_h, crop_w)
        masks_c = [TF.crop(m, top, left, crop_h, crop_w) for m in masks]
        return image_c, masks_c

    def _defect_focused_crop(self, image, masks, crop_h, crop_w, tries=10, min_defect_ratio=0.01):
        w, h = image.size
        if h <= crop_h or w <= crop_w:
            return image, masks

        low_w, low_h = self.low_res_union
        union_small = np.zeros((low_h, low_w), dtype=np.uint8)

        for m in masks:
            # ✅ CRACK FIX: Use BILINEAR so thin cracks aren't dropped in low-res search
            ms = m.resize((low_w, low_h), resample=Image.BILINEAR)
            union_small = np.maximum(union_small, np.array(ms, dtype=np.uint8))

        ys, xs = np.where(union_small > 0)
        if ys.size == 0:
            return self._random_crop(image, masks, crop_h, crop_w)

        for _ in range(tries):
            i = random.randint(0, ys.size - 1)
            y_s, x_s = int(ys[i]), int(xs[i])

            y = int(y_s * (h / low_h))
            x = int(x_s * (w / low_w))

            top = y - crop_h // 2
            left = x - crop_w // 2
            top = max(0, min(top, h - crop_h))
            left = max(0, min(left, w - crop_w))

            img_c = TF.crop(image, top, left, crop_h, crop_w)
            masks_c = [TF.crop(m, top, left, crop_h, crop_w) for m in masks]

            union_crop = np.zeros((crop_h, crop_w), dtype=np.uint8)
            for mc in masks_c:
                union_crop = np.maximum(union_crop, np.array(mc, dtype=np.uint8))
            defect_ratio = float((union_crop > 0).mean())

            if defect_ratio >= min_defect_ratio:
                return img_c, masks_c

        return self._random_crop(image, masks, crop_h, crop_w)

    def _apply_transforms(self, image, masks):
        import config

        crop_h = max(64, int(self.img_size[0] * self.crop_ratio))
        crop_w = max(64, int(self.img_size[1] * self.crop_ratio))

        # Crop
        if random.random() < self.defect_crop_prob:
            image, masks = self._defect_focused_crop(
                image, masks, crop_h, crop_w,
                tries=self.crop_tries,
                min_defect_ratio=self.min_defect_ratio,
            )
        else:
            image, masks = self._random_crop(image, masks, crop_h, crop_w)

        # Resize
        image = TF.resize(image, self.img_size)
        # ✅ CRACK FIX: Use BILINEAR to preserve thin structures during resize
        masks = [TF.resize(m, self.img_size, interpolation=Image.BILINEAR) for m in masks]

        # Horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            masks = [TF.hflip(m) for m in masks]

        # Vertical flip
        if random.random() > 0.5:
            image = TF.vflip(image)
            masks = [TF.vflip(m) for m in masks]

        # Rotation
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle)
            # ✅ CRACK FIX: Use BILINEAR to preserve thin lines during rotation
            masks = [TF.rotate(m, angle, interpolation=Image.BILINEAR) for m in masks]

        # Color Jitter (applied while still PIL Image)
        if getattr(config, 'USE_COLOR_JITTER', False):
            if random.random() > 0.5:
                image = T.ColorJitter(
                    brightness=getattr(config, 'COLOR_JITTER_BRIGHTNESS', 0.3),
                    contrast=getattr(config, 'COLOR_JITTER_CONTRAST', 0.3),
                    saturation=getattr(config, 'COLOR_JITTER_SATURATION', 0.3),
                    hue=getattr(config, 'COLOR_JITTER_HUE', 0.1),
                )(image)

        # Gaussian Blur (still PIL)
        if getattr(config, 'USE_RANDOM_BLUR', False):
            if random.random() < getattr(config, 'BLUR_PROB', 0.3):
                kernel_sizes = getattr(config, 'BLUR_KERNEL_SIZES', [3, 5, 7])
                kernel_size = random.choice(kernel_sizes)
                image = TF.gaussian_blur(image, kernel_size)

        return image, masks

    def __getitem__(self, idx):
        import config

        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        base = os.path.splitext(img_name)[0]

        masks = self._load_multilabel_masks(base, image.size)

        if self.transform:
            image, masks = self._apply_transforms(image, masks)
        else:
            image = TF.resize(image, self.img_size)
            # ✅ CRACK FIX: Use BILINEAR for validation set too
            masks = [TF.resize(m, self.img_size, interpolation=Image.BILINEAR) for m in masks]

        # Convert to tensor first (values in [0.0, 1.0])
        image = TF.to_tensor(image)

        if self.transform and getattr(config, 'USE_RANDOM_NOISE', False):
            if random.random() < getattr(config, 'NOISE_PROB', 0.2):
                noise = torch.randn_like(image) * getattr(config, 'NOISE_STD', 0.02)
                image = torch.clamp(image + noise, 0.0, 1.0)

        image = TF.normalize(
            image,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        # Masks → [C, H, W] float32 binary tensor
        masks_t = []
        for m in masks:
            arr = np.array(m, dtype=np.uint8)
            # Re-binarize here because BILINEAR introduces gray values
            masks_t.append(torch.from_numpy((arr > 0).astype(np.float32)))
        masks_t = torch.stack(masks_t, dim=0)

        # ─────────────────────────────────────────────────────────────────
        # ✅ CRACK FIX: MORPHOLOGICAL DILATION
        # Thickens the crack mask slightly so the network can see it.
        # Assuming crack is the 1st class in your config (index 0).
        # ─────────────────────────────────────────────────────────────────
        crack_idx = 0 
        
        # We only apply dilation during training to help the model learn,
        # but you can remove `if self.transform:` to apply it during validation too.
        if self.transform:
            crack_mask = masks_t[crack_idx].unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
            thick_crack = F.max_pool2d(crack_mask, kernel_size=3, stride=1, padding=1)
            masks_t[crack_idx] = thick_crack.squeeze()
        # ─────────────────────────────────────────────────────────────────

        return image, masks_t


# ──────────────────────────────────────────────────────────────────────────────

def get_dataloaders_multilabel(
    data_root,
    batch_size=8,
    num_workers=8,
    img_size=(512, 512),
    images_subdir="images",
    masks_subdir="masks_multilabel",
    cpu_mode=False,
    defect_crop_prob=0.7,
    crop_ratio=0.60,
    crop_tries=10,
    min_defect_ratio=0.01,
):
    import config

    if cpu_mode:
        img_size = (256, 256)
        batch_size = 1
        num_workers = 0
        print("🐌 CPU MODE: 256x256, batch_size=1")

    train_img_dir  = os.path.join(data_root, images_subdir, "train")
    val_img_dir    = os.path.join(data_root, images_subdir, "val")
    train_mask_dir = os.path.join(data_root, masks_subdir, "train")
    val_mask_dir   = os.path.join(data_root, masks_subdir, "val")

    classes_to_load = getattr(config, 'CLASSES_TO_LOAD', [1, 7, 11])
    num_labels = len(classes_to_load)

    train_ds = DACL10KMultiLabelDataset(
        img_dir=train_img_dir,
        mask_dir=train_mask_dir,
        classes_to_load=classes_to_load,
        transform=True,
        img_size=img_size,
        defect_crop_prob=defect_crop_prob,
        crop_ratio=crop_ratio,
        crop_tries=crop_tries,
        min_defect_ratio=min_defect_ratio,
    )

    val_ds = DACL10KMultiLabelDataset(
        img_dir=val_img_dir,
        mask_dir=val_mask_dir,
        classes_to_load=classes_to_load,
        transform=False,
        img_size=img_size,
    )

    pin_mem = not cpu_mode

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=False,
    )

    print(f"✅ Train : {len(train_ds)} images | {len(train_loader)} batches")
    print(f"✅ Val   : {len(val_ds)} images | {len(val_loader)} batches")
    print(f"✅ Image size : {img_size}")
    print(f"✅ Num labels : {num_labels}  (classes: {classes_to_load})")

    return train_loader, val_loader, num_labels
