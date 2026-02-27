from pathlib import Path
from PIL import Image

DATA_ROOT = Path(r"C:\Users\Informatics\Desktop\dataset_mémoire\segmentation_project\dataset")

def find_mismatches(split):
    img_dir = DATA_ROOT / "images" / split
    mask_dir = DATA_ROOT / "masks" / split

    print(f"\nChecking {split}...")

    for img_path in img_dir.iterdir():
        if img_path.suffix.lower() not in [".jpg", ".png"]:
            continue

        mask_path = mask_dir / (img_path.stem + ".png")
        if not mask_path.exists():
            continue

        with Image.open(img_path) as im:
            iw, ih = im.size
        with Image.open(mask_path) as mk:
            mw, mh = mk.size

        if (iw, ih) != (mw, mh):
            print(f"❌ {img_path.name}")
            print(f"   Image: {iw}×{ih}")
            print(f"   Mask : {mw}×{mh}")

find_mismatches("train")
find_mismatches("val")