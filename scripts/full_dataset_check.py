from pathlib import Path
from PIL import Image
import numpy as np
import random

ROOT = Path(r"C:\Users\Ismail Triki\Desktop\hani_dataset_memoire\dacl10k-segmentation\dataset")
NUM_CLASSES = 20  # change if needed


def list_images(split_name: str):
    img_dir = ROOT / "images" / split_name
    return sorted([p for p in img_dir.glob("*") if p.suffix.lower() in [".jpg", ".png", ".jpeg"]])


def list_masks(split_name: str):
    mask_dir = ROOT / "masks" / split_name
    return sorted(mask_dir.glob("*.png"))


def check_split(split_name: str, sample_masks: int = 5):
    print("\n==============================")
    print("Checking split:", split_name)
    print("==============================")

    imgs = list_images(split_name)
    masks = list_masks(split_name)

    img_names = {p.stem for p in imgs}
    mask_names = {p.stem for p in masks}

    print("Images:", len(imgs))
    print("Masks :", len(masks))

    missing_masks = img_names - mask_names
    missing_imgs = mask_names - img_names

    print("Missing masks:", len(missing_masks))
    print("Missing images:", len(missing_imgs))

    if missing_masks:
        print("Example missing mask:", list(missing_masks)[:10])

    if missing_imgs:
        print("Example missing image:", list(missing_imgs)[:10])

    # Random mask inspection
    if masks:
        sample = random.sample(masks, min(sample_masks, len(masks)))
        print("\nInspecting random masks:")
        for mpath in sample:
            m = Image.open(mpath)
            arr = np.array(m)

            print("File:", mpath.name)
            print("  mode:", m.mode)
            print("  shape:", arr.shape)
            print("  dtype:", arr.dtype)
            print("  min:", int(arr.min()), "max:", int(arr.max()))
            print("  unique sample:", np.unique(arr)[:10])

            # Format checks
            if m.mode != "L":
                print("  ❌ ERROR: mask is not single channel (mode 'L')")
            if arr.ndim != 2:
                print("  ❌ ERROR: mask is not 2D (has multiple channels)")

            # Value checks
            if arr.max() >= NUM_CLASSES:
                print(f"  ❌ ERROR: mask has value {int(arr.max())} >= NUM_CLASSES ({NUM_CLASSES})")
            if 255 in np.unique(arr):
                print("  ⚠️ WARNING: mask contains 255 (often used as ignore_index)")

            print("")

    print("Split check done ✔")


def get_names(split_name: str):
    return {p.stem for p in list_images(split_name)}


def check_overlaps():
    train_names = get_names("train")
    val_names = get_names("val")
    test_names = get_names("test")

    print("\n==============================")
    print("Checking split overlaps (data leakage)")
    print("==============================")
    tv = train_names & val_names
    tt = train_names & test_names
    vt = val_names & test_names

    print("train ∩ val :", len(tv))
    print("train ∩ test:", len(tt))
    print("val   ∩ test:", len(vt))

    if tv:
        print("Example overlaps train/val :", list(tv)[:10])
    if tt:
        print("Example overlaps train/test:", list(tt)[:10])
    if vt:
        print("Example overlaps val/test  :", list(vt)[:10])

    if not tv and not tt and not vt:
        print("✅ No overlaps found ✔")


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        check_split(split)

    check_overlaps()

    print("\n✅ FULL DATASET CHECK COMPLETE")