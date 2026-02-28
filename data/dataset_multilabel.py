"""
Multi-label Dataset Loader for DACL10K
Outputs:
  image: [3,H,W]
  masks: [19,H,W] float {0,1}

Files expected:
  masks_multilabel/<split>/<basename>_class01.png ... _class19.png
"""

import os
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class DACL10KMultiLabelDataset(Dataset):
    def __init__(
        self,
        img_dir,
        mask_dir,
        num_labels=19,
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
        self.num_labels = int(num_labels)
        self.transform = bool(transform)
        self.img_size = tuple(img_size)

        # Crop config
        self.defect_crop_prob = float(defect_crop_prob)
        self.crop_ratio = float(crop_ratio)
        self.crop_tries = int(crop_tries)
        self.min_defect_ratio = float(min_defect_ratio)

        # Low-res union settings (memory safe)
        self.low_res_union = tuple(low_res_union)

        self.images = sorted(
            [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png"))]
        )
        print(f"📂 Found {len(self.images)} images in {img_dir}")

    def __len__(self):
        return len(self.images)

    # ---------------------------
    # Cropping helpers
    # ---------------------------
    def _random_crop(self, image, masks, crop_h, crop_w):
        w, h = image.size # PIL: (W,H)
        if h <= crop_h or w <= crop_w:
            return image, masks

        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)

        image_c = TF.crop(image, top, left, crop_h, crop_w)
        masks_c = [TF.crop(m, top, left, crop_h, crop_w) for m in masks]
        return image_c, masks_c

    def _defect_focused_crop(self, image, masks, crop_h, crop_w, tries=10, min_defect_ratio=0.01):
        """
        Memory-safe defect-focused crop for multi-label masks.
        - Build a LOW-RES union mask (e.g. 256x256) to sample defect locations
        - Map sampled point back to original resolution
        - Crop image + all masks
        - Validate defect_ratio on the CROPPED area (small) only
        """
        w, h = image.size # PIL (W,H)
        if h <= crop_h or w <= crop_w:
            return image, masks

        low_w, low_h = self.low_res_union
        union_small = np.zeros((low_h, low_w), dtype=np.uint8)

        # Build low-res union mask
        for m in masks:
            ms = m.resize((low_w, low_h), resample=Image.NEAREST)
            union_small = np.maximum(union_small, np.array(ms, dtype=np.uint8))

        ys, xs = np.where(union_small > 0)
        if ys.size == 0:
            return self._random_crop(image, masks, crop_h, crop_w)

        for _ in range(tries):
            i = random.randint(0, ys.size - 1)
            y_s, x_s = int(ys[i]), int(xs[i])

            # Map low-res coords -> original coords
            y = int(y_s * (h / low_h))
            x = int(x_s * (w / low_w))

            top = y - crop_h // 2
            left = x - crop_w // 2
            top = max(0, min(top, h - crop_h))
            left = max(0, min(left, w - crop_w))

            img_c = TF.crop(image, top, left, crop_h, crop_w)
            masks_c = [TF.crop(m, top, left, crop_h, crop_w) for m in masks]

            # Compute defect ratio on cropped union
            union_crop = np.zeros((crop_h, crop_w), dtype=np.uint8)
            for mc in masks_c:
                union_crop = np.maximum(union_crop, np.array(mc, dtype=np.uint8))
            defect_ratio = float((union_crop > 0).mean())

            if defect_ratio >= min_defect_ratio:
                return img_c, masks_c

        return self._random_crop(image, masks, crop_h, crop_w)

    def _load_multilabel_masks(self, base_name):
        """
        base_name = file stem without extension
        returns list of PIL masks length=num_labels, each is L (0 or 255)
        """
        masks = []
        for k in range(1, self.num_labels + 1):
            fn = f"{base_name}_class{k:02d}.png"
            fp = os.path.join(self.mask_dir, fn)
            m = Image.open(fp).convert("L")
            masks.append(m)
        return masks

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        base = os.path.splitext(img_name)[0]
        masks = self._load_multilabel_masks(base)

        if self.transform:
            image, masks = self._apply_transforms(image, masks)
        else:
            image = TF.resize(image, self.img_size)
            masks = [TF.resize(m, self.img_size, interpolation=Image.NEAREST) for m in masks]

        # image -> tensor
        image = TF.to_tensor(image)
        image = TF.normalize(
            image,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        # masks -> [C,H,W] float {0,1}
        masks_t = []
        for m in masks:
            arr = np.array(m, dtype=np.uint8)
            masks_t.append(torch.from_numpy((arr > 0).astype(np.float32)))
        masks_t = torch.stack(masks_t, dim=0) # [C,H,W]

        return image, masks_t

    def _apply_transforms(self, image, masks):
        """
        TRAIN pipeline:
        1) crop on original resolution (defect-focused prob)
        2) resize to img_size
        3) augmentations (flip/rotate/jitter) on final size
        """
        crop_h = max(64, int(self.img_size[0] * self.crop_ratio))
        crop_w = max(64, int(self.img_size[1] * self.crop_ratio))

        if random.random() < self.defect_crop_prob:
            image, masks = self._defect_focused_crop(
                image,
                masks,
                crop_h,
                crop_w,
                tries=self.crop_tries,
                min_defect_ratio=self.min_defect_ratio,
            )
        else:
            image, masks = self._random_crop(image, masks, crop_h, crop_w)

        # resize
        image = TF.resize(image, self.img_size)
        masks = [TF.resize(m, self.img_size, interpolation=Image.NEAREST) for m in masks]

        # flips
        if random.random() > 0.5:
            image = TF.hflip(image)
            masks = [TF.hflip(m) for m in masks]

        if random.random() > 0.5:
            image = TF.vflip(image)
            masks = [TF.vflip(m) for m in masks]

        # rotate
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle)
            masks = [TF.rotate(m, angle, interpolation=Image.NEAREST) for m in masks]

        # color jitter (image only)
        if random.random() > 0.5:
            image = T.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
            )(image)

        return image, masks


def get_dataloaders_multilabel(
    data_root,
    batch_size=2,
    num_workers=4,
    img_size=(512, 512),
    images_subdir="images",
    masks_subdir="masks_multilabel",
    cpu_mode=False,
    defect_crop_prob=0.7,
    crop_ratio=0.60,
    crop_tries=10,
    min_defect_ratio=0.01,
    num_labels=19,
):
    if cpu_mode:
        img_size = (256, 256)
        batch_size = 1
        num_workers = 0
        print("🐌 CPU MODE (multilabel): 256x256, batch_size=1")

    train_img_dir = os.path.join(data_root, images_subdir, "train")
    val_img_dir = os.path.join(data_root, images_subdir, "val")

    train_mask_dir = os.path.join(data_root, masks_subdir, "train")
    val_mask_dir = os.path.join(data_root, masks_subdir, "val")

    train_ds = DACL10KMultiLabelDataset(
        img_dir=train_img_dir,
        mask_dir=train_mask_dir,
        num_labels=num_labels,
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
        num_labels=num_labels,
        transform=False,
        img_size=img_size,
        defect_crop_prob=0.0,
        crop_ratio=crop_ratio,
        crop_tries=crop_tries,
        min_defect_ratio=min_defect_ratio,
    )

    # pin_memory: DirectML/CPU doesn't benefit -> keep False
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )

    print(f"✅ Train: {len(train_ds)} images, {len(train_loader)} batches")
    print(f"✅ Val: {len(val_ds)} images, {len(val_loader)} batches")

    return train_loader, val_loader, num_labels