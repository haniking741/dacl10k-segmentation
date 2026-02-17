import os
from pathlib import Path
from PIL import Image
import numpy as np
import random

ROOT = Path("dataset")

def check_split(split_name):
    print("\n==============================")
    print("Checking split:", split_name)
    print("==============================")

    img_dir = ROOT / "images" / split_name
    mask_dir = ROOT / "masks" / split_name

    imgs = sorted([p for p in img_dir.glob("*") if p.suffix.lower() in [".jpg", ".png", ".jpeg"]])
    masks = sorted(mask_dir.glob("*.png"))

    img_names = {p.stem for p in imgs}
    mask_names = {p.stem for p in masks}

    print("Images:", len(imgs))
    print("Masks :", len(masks))

    missing_masks = img_names - mask_names
    missing_imgs  = mask_names - img_names

    print("Missing masks:", len(missing_masks))
    print("Missing images:", len(missing_imgs))

    if missing_masks:
        print("Example missing mask:", list(missing_masks)[:5])

    if missing_imgs:
        print("Example missing image:", list(missing_imgs)[:5])

    # Random mask inspection
    if masks:
        sample = random.sample(masks, min(5, len(masks)))
        print("\nInspecting random masks:")
        for mpath in sample:
            m = Image.open(mpath)
            arr = np.array(m)

            print("File:", mpath.name)
            print("  mode:", m.mode)
            print("  shape:", arr.shape)
            print("  min:", arr.min(), "max:", arr.max())
            print("  unique values sample:", np.unique(arr)[:10])
            print("")

            if m.mode != "L":
                print("❌ ERROR: mask is not single channel (L)")
            
            if len(arr.shape) != 2:
                print("❌ ERROR: mask has multiple channels")

    print("Split check done ✔")

if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        check_split(split)

    print("\n✅ FULL DATASET CHECK COMPLETE")
