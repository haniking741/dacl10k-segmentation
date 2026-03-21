"""
compute_class_weights.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Compute class weights for balanced multi-label training
For classes: Crack, Spalling, Rust

Usage:
    python scripts/compute_class_weights.py
"""

import os
import sys
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm


def compute_bce_pos_weight(
    masks_dir="dataset2/masks_multilabel/train",
    classes_to_load=[1, 7, 11],  # crack, spalling, rust
):
    """
    Compute BCE pos_weight for multi-label classification
    
    pos_weight[i] = negative_pixels[i] / positive_pixels[i]
    
    This tells BCE loss how much to penalize missing a positive pixel
    """
    
    masks_dir = Path(masks_dir)
    class_names = ["crack", "spalling", "rust"]
    
    print("=" * 70)
    print("COMPUTING BCE POS WEIGHTS (3 CLASSES)")
    print("=" * 70)
    print(f"Masks directory: {masks_dir}")
    print(f"Classes to load: {classes_to_load}")
    print(f"Class names: {class_names}")
    print()
    
    # Check if directory exists
    if not masks_dir.exists():
        print(f"❌ ERROR: Directory not found!")
        print(f"   Path: {masks_dir.absolute()}")
        print()
        print("Troubleshooting:")
        print("  1. Make sure you're in the project root:")
        print(f"     cd C:\\Users\\Informatics\\Desktop\\dataset_mémoire\\segmentation_project")
        print("  2. Check if dataset2 folder exists")
        print("  3. Check if masks_multilabel/train exists")
        return None
    
    # Get all base names from class 01 files
    all_files = list(masks_dir.glob("*_class01.png"))
    
    if len(all_files) == 0:
        print("❌ ERROR: No mask files found!")
        print(f"   Searched in: {masks_dir.absolute()}")
        print(f"   Pattern: *_class01.png")
        print()
        print("Debugging:")
        print("  Run this command to check:")
        print(f"  dir {masks_dir}\\*_class01.png")
        return None
    
    base_names = [f.stem.replace("_class01", "") for f in all_files]
    
    print(f"✅ Found {len(base_names)} training images")
    print(f"   This will take ~3-5 minutes...")
    print()
    
    # Initialize counters
    total_pixels = 0
    positive_pixels = np.zeros(len(classes_to_load), dtype=np.int64)
    
    print("Scanning masks...")
    for base_name in tqdm(base_names, desc="Progress"):
        # Get image size from first available mask
        first_mask_path = None
        for cid in classes_to_load:
            p = masks_dir / f"{base_name}_class{cid:02d}.png"
            if p.exists():
                first_mask_path = p
                break
        
        if first_mask_path is None:
            continue
        
        # Get image dimensions
        img = Image.open(first_mask_path)
        h, w = img.size[1], img.size[0]
        total_pixels += h * w
        
        # Count positive pixels per class
        for i, class_id in enumerate(classes_to_load):
            mask_path = masks_dir / f"{base_name}_class{class_id:02d}.png"
            
            if mask_path.exists():
                mask = np.array(Image.open(mask_path))
                positive_pixels[i] += (mask > 0).sum()
    
    # Compute negative pixels
    negative_pixels = total_pixels - positive_pixels
    
    # Compute pos_weight = neg / pos
    pos_weight = negative_pixels / (positive_pixels + 1e-6)
    
    # Compute sqrt-scaled version (more stable)
    pos_weight_sqrt = np.sqrt(pos_weight)
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS:")
    print("=" * 70)
    print(f"Total pixels: {total_pixels:,}")
    print()
    
    print(f"{'Class':<15} {'Positive':>15} {'Negative':>15} {'PosWeight':>12} {'Sqrt':>12}")
    print("-" * 70)
    
    for i, name in enumerate(class_names):
        print(f"{name:<15} {positive_pixels[i]:>15,} {negative_pixels[i]:>15,} "
              f"{pos_weight[i]:>12.2f} {pos_weight_sqrt[i]:>12.2f}")
    
    print()
    print("=" * 70)
    print("COPY THIS TO config.py:")
    print("=" * 70)
    print()
    print("# Original (mathematically correct):")
    print(f"BCE_POS_WEIGHT = {pos_weight.tolist()}")
    print()
    print("# Sqrt-scaled (MORE STABLE, RECOMMENDED): ✅")
    print(f"BCE_POS_WEIGHT = {pos_weight_sqrt.tolist()}")
    print()
    print("=" * 70)
    
    print()
    print("📊 Interpretation:")
    print("-" * 70)
    for i, name in enumerate(class_names):
        freq = (positive_pixels[i] / total_pixels) * 100
        print(f"{name:12s}: {freq:.2f}% of pixels, weight = {pos_weight_sqrt[i]:.2f}×")
    
    print()
    print("✅ Recommendation: Use sqrt-scaled version (more stable)")
    print()
    
    return pos_weight, pos_weight_sqrt


def main():
    """Main function"""
    
    print()
    print("🔍" * 35)
    print("DACL10K CLASS WEIGHT COMPUTATION")
    print("🔍" * 35)
    print()
    
    # Compute weights
    result = compute_bce_pos_weight(
        masks_dir="dataset2/masks_multilabel/train",
        classes_to_load=[1, 7, 11],  # crack, spalling, rust
    )
    
    if result is not None:
        pos_weight, pos_weight_sqrt = result
        
        print()
        print("=" * 70)
        print("✅ SUCCESS!")
        print("=" * 70)
        print()
        print("Next steps:")
        print("━" * 70)
        print("1. Open config.py")
        print("2. Update these lines:")
        print()
        print("   CLASSES_TO_LOAD = [1, 7, 11]  # crack, spalling, rust")
        print("   CLASS_NAMES = ['crack', 'spalling', 'rust']")
        print(f"   BCE_POS_WEIGHT = {pos_weight_sqrt.tolist()}")
        print()
        print("3. Delete old checkpoint:")
        print("   rm checkpoints\\checkpoint_best_multilabel.pth")
        print()
        print("4. Start training:")
        print("   python train_multilabel.py")
        print("━" * 70)
        print()
        print(f"Expected mIoU: 0.38-0.43 ✅✅✅")
        print()
    else:
        print()
        print("=" * 70)
        print("❌ FAILED!")
        print("=" * 70)
        print()
        print("Please check the error messages above and:")
        print("  1. Make sure you're in the correct directory")
        print("  2. Verify dataset2/masks_multilabel/train exists")
        print("  3. Check if mask files exist")
        print()


if __name__ == "__main__":
    main()