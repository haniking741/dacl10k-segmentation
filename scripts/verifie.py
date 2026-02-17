from pathlib import Path

# Paths
img_dir = Path("dataset/images/train")
mask_dir = Path("dataset/masks/train")

# Get all files
images = {f.stem for f in img_dir.glob("*.jpg")}
masks = {f.stem for f in mask_dir.glob("*.png")}

print(f"Images: {len(images)}")
print(f"Masks: {len(masks)}")

# Find mismatches
missing_masks = images - masks
missing_images = masks - images

if missing_masks:
    print(f"\n⚠ Images without masks: {len(missing_masks)}")
    for m in list(missing_masks)[:10]:
        print(f"  - {m}")
else:
    print("\n✅ All images have masks!")

if missing_images:
    print(f"\n⚠ Masks without images: {len(missing_images)}")
    for m in list(missing_images)[:10]:
        print(f"  - {m}")
else:
    print("✅ All masks have images!")