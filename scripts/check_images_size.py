"""
Check ON-DISK image & mask sizes for the dataset used by your training project.
This does NOT use the dataloader. It reads files from dataset/images/* and dataset/masks/*

Edit DATA_ROOT to your project dataset folder.
"""

import os
from pathlib import Path
from PIL import Image
from collections import Counter
import numpy as np

DATA_ROOT = Path(r"C:\Users\Informatics\Desktop\dataset_mémoire\segmentation_project\dataset")  # <-- EDIT

def analyze_split(split_name: str):
    img_dir = DATA_ROOT / "images" / split_name
    mask_dir = DATA_ROOT / "masks" / split_name

    print(f"\n{'='*80}")
    print(f"📊 SPLIT: {split_name.upper()}")
    print(f"Images: {img_dir}")
    print(f"Masks : {mask_dir}")
    print(f"{'='*80}")

    if not img_dir.exists() or not mask_dir.exists():
        print("❌ Split folder missing.")
        return

    img_files = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in [".jpg", ".png"]])
    if not img_files:
        print("❌ No images found.")
        return

    img_sizes = []
    mask_sizes = []
    mismatch = 0
    missing_mask = 0

    for p in img_files:
        mask_path = (mask_dir / (p.stem + ".png"))
        if not mask_path.exists():
            missing_mask += 1
            continue

        try:
            with Image.open(p) as im:
                iw, ih = im.size
            with Image.open(mask_path) as mk:
                mw, mh = mk.size

            img_sizes.append((iw, ih))
            mask_sizes.append((mw, mh))

            if (iw, ih) != (mw, mh):
                mismatch += 1

        except Exception as e:
            print(f"⚠️ Error reading {p.name}: {e}")

    print(f"✅ Images found: {len(img_files)}")
    print(f"✅ Paired (image+mask) read: {len(img_sizes)}")
    print(f"⚠️ Missing masks: {missing_mask}")
    print(f"⚠️ Size mismatches (image != mask): {mismatch}")

    def stats(name, sizes):
        ws = [s[0] for s in sizes]
        hs = [s[1] for s in sizes]
        print(f"\n{name} size stats:")
        print(f"  Width : min={min(ws)} max={max(ws)} avg={int(np.mean(ws))} median={int(np.median(ws))}")
        print(f"  Height: min={min(hs)} max={max(hs)} avg={int(np.mean(hs))} median={int(np.median(hs))}")

        cnt = Counter(sizes)
        print(f"  Top-5 most common sizes:")
        for i, ((w, h), c) in enumerate(cnt.most_common(5), 1):
            print(f"    {i}. {w}×{h}: {c} ({100*c/len(sizes):.1f}%)")

    if img_sizes:
        stats("IMAGES", img_sizes)
    if mask_sizes:
        stats("MASKS ", mask_sizes)

if __name__ == "__main__":
    print("🔍 Checking ON-DISK sizes (no dataloader, no resize in-memory)")
    for split in ["train", "val"]:
        analyze_split(split)