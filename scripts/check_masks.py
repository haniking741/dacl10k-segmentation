"""
Verify mask values are correct (0-19 range)
"""
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

# Paths
MASK_DIR = Path(r"C:\Users\Ismail Triki\Desktop\hani_dataset_memoire\dacl10k-segmentation\dataset\masks\train")
IMG_DIR = Path(r"C:\Users\Ismail Triki\Desktop\hani_dataset_memoire\dacl10k-segmentation\dataset\images\train")

# Check 10 random masks
masks = list(MASK_DIR.glob("*.png"))
print(f"📂 Found {len(masks)} masks\n")

print("="*70)
print("MASK VALUE VERIFICATION")
print("="*70)

for i, mask_path in enumerate(masks[:10]):
    mask = np.array(Image.open(mask_path))
    
    unique_vals = np.unique(mask)
    min_val = mask.min()
    max_val = mask.max()
    
    # Count pixels per class
    class_counts = {}
    for val in unique_vals:
        count = np.sum(mask == val)
        pct = 100 * count / mask.size
        class_counts[val] = (count, pct)
    
    print(f"\n{i+1}. {mask_path.name}")
    print(f"   Shape: {mask.shape}")
    print(f"   Min: {min_val}, Max: {max_val}")
    print(f"   Unique values: {unique_vals}")
    print(f"   Value distribution:")
    for val in sorted(class_counts.keys()):
        count, pct = class_counts[val]
        print(f"      Class {val:2d}: {count:8,} pixels ({pct:5.2f}%)")
    
    # Check if values are valid (0-19)
    if max_val > 19 or min_val < 0:
        print(f"   ⚠️  WARNING: Invalid values! (should be 0-19)")
    else:
        print(f"   ✅ Values in valid range (0-19)")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

# Check all masks quickly
all_values = set()
background_dominant = 0
total_bg_pixels = 0
total_pixels = 0

for mask_path in masks:
    mask = np.array(Image.open(mask_path))
    all_values.update(np.unique(mask).tolist())
    
    bg_pct = 100 * np.sum(mask == 0) / mask.size
    if bg_pct > 70:
        background_dominant += 1
    
    total_bg_pixels += np.sum(mask == 0)
    total_pixels += mask.size

print(f"Total masks checked: {len(masks)}")
print(f"All unique values across dataset: {sorted(all_values)}")
print(f"Min value: {min(all_values)}, Max value: {max(all_values)}")
print(f"Images with >70% background: {background_dominant}/{len(masks)} ({100*background_dominant/len(masks):.1f}%)")
print(f"Overall background %: {100*total_bg_pixels/total_pixels:.2f}%")

if max(all_values) > 19 or min(all_values) < 0:
    print("\n❌ ERROR: Masks have invalid values!")
    print("   Expected: 0-19")
    print(f"   Found: {min(all_values)}-{max(all_values)}")
else:
    print("\n✅ All masks have valid values (0-19)")

# Visualize one mask
print("\n📊 Visualizing first mask...")
mask = np.array(Image.open(masks[0]))
img_path = IMG_DIR / (masks[0].stem + ".jpg")
if not img_path.exists():
    img_path = IMG_DIR / (masks[0].stem + ".png")

if img_path.exists():
    img = Image.open(img_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap='tab20', vmin=0, vmax=19)
    axes[1].set_title(f"Mask (values: {np.unique(mask)})")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig("mask_verification.png", dpi=150, bbox_inches='tight')
    print("✅ Saved visualization to mask_verification.png")
    plt.show()