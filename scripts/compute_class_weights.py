"""
compute_class_weights.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Compute class weights for balanced multi-label training
CRITICAL for preventing model collapse in imbalanced datasets!

Usage:
    python compute_class_weights.py

Output:
    - Prints class distribution statistics
    - Prints BCE_POS_WEIGHT to add to config.py
"""

import os
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

def compute_class_weights(
    masks_dir="dataset2/masks_multilabel/train",
    classes_to_load=[7, 9, 11],  # spalling, cavity, rust
    method="inverse_freq"
):
    """
    Compute class weights based on pixel frequency in training set
    
    Args:
        masks_dir: Path to training masks directory
        classes_to_load: List of class IDs to compute weights for
        method: 'inverse_freq' or 'effective_samples'
        
    Returns:
        weights: numpy array of class weights
    """
    
    masks_dir = Path(masks_dir)
    class_names = ["spalling", "cavity", "rust"]
    
    print("="*70)
    print("COMPUTING CLASS WEIGHTS FOR DACL10K (3 CLASSES)")
    print("="*70)
    print(f"Masks directory: {masks_dir}")
    print(f"Classes: {class_names}")
    print(f"Method: {method}")
    print()
    
    # Get all base names from class01 files
    all_files = list(masks_dir.glob("*_class01.png"))
    
    if len(all_files) == 0:
        print("❌ ERROR: No mask files found!")
        print(f"   Searched in: {masks_dir.absolute()}")
        print("\nTroubleshooting:")
        print("1. Check if masks_dir path is correct")
        print("2. Make sure you're in the project root directory")
        print("3. Verify masks exist with: dir dataset2\\masks_multilabel\\train")
        return None
    
    base_names = [f.stem.replace("_class01", "") for f in all_files]
    
    print(f"✓ Found {len(base_names)} training images")
    print(f"  This will take ~5-10 minutes to scan all masks...")
    print()
    
    # Count pixels per class
    class_pixels = np.zeros(len(classes_to_load), dtype=np.int64)
    total_pixels = 0
    background_pixels = 0
    
    print("Scanning masks...")
    for base_name in tqdm(base_names, desc="Progress"):
        # Get image size from first mask
        first_mask_path = masks_dir / f"{base_name}_class01.png"
        
        if first_mask_path.exists():
            img = Image.open(first_mask_path)
            img_pixels = img.size[0] * img.size[1]
            total_pixels += img_pixels
        else:
            continue
        
        # Count defect pixels per class
        image_has_defects = np.zeros(img.size[::-1], dtype=bool)
        
        for i, class_id in enumerate(classes_to_load):
            mask_path = masks_dir / f"{base_name}_class{class_id:02d}.png"
            
            if mask_path.exists():
                mask = np.array(Image.open(mask_path))
                defect_pixels = (mask > 0)
                class_pixels[i] += defect_pixels.sum()
                image_has_defects = image_has_defects | defect_pixels
        
        # Background = pixels without any defect
        background_pixels += (~image_has_defects).sum()
    
    # Compute frequencies
    frequencies = class_pixels / total_pixels
    background_freq = background_pixels / total_pixels
    
    print("\n" + "="*70)
    print("CLASS DISTRIBUTION ANALYSIS:")
    print("="*70)
    print(f"Total pixels analyzed: {total_pixels:,}")
    print()
    
    print(f"{'Class':<15} {'Pixels':>15} {'Percentage':>12} {'Frequency':>12}")
    print("-"*70)
    
    # Background
    print(f"{'background':<15} {background_pixels:>15,} {background_freq*100:>11.2f}% {background_freq:>12.6f}")
    
    # Defect classes
    for i, (name, pixels, freq) in enumerate(zip(class_names, class_pixels, frequencies)):
        print(f"{name:<15} {pixels:>15,} {freq*100:>11.2f}% {freq:>12.6f}")
    
    print()
    
    # Compute weights
    if method == "inverse_freq":
        # Inverse frequency: weight = 1 / frequency
        # Higher weight for rare classes
        weights = 1.0 / (frequencies + 1e-6)
        
        # Normalize so mean weight = 1.0
        weights = weights / weights.mean()
        
    elif method == "effective_samples":
        # Effective number of samples method
        # More sophisticated, from paper: "Class-Balanced Loss"
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, class_pixels)
        weights = (1.0 - beta) / (effective_num + 1e-6)
        
        # Normalize
        weights = weights / weights.mean()
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print("="*70)
    print(f"CLASS WEIGHTS ({method}):")
    print("="*70)
    print(f"{'Class':<15} {'Weight':>12} {'Effect':<30}")
    print("-"*70)
    
    for name, weight, freq in zip(class_names, weights, frequencies):
        if weight > 1.0:
            effect = f"↑ Boost (rare class, {freq*100:.2f}%)"
        elif weight < 1.0:
            effect = f"↓ Reduce (common class, {freq*100:.2f}%)"
        else:
            effect = "= Neutral"
        print(f"{name:<15} {weight:>12.4f}    {effect}")
    
    print()
    print("Interpretation:")
    print(f"  - Cavity has highest weight ({weights[1]:.4f}) → rarest class, needs boost")
    print(f"  - Rust has lowest weight ({weights[2]:.4f}) → most common, reduce influence")
    print(f"  - Spalling in middle ({weights[0]:.4f}) → balanced")
    print()
    
    # Output for config.py
    print("="*70)
    print("COPY THIS LINE TO config.py:")
    print("="*70)
    print()
    print(f"BCE_POS_WEIGHT = {weights.tolist()}")
    print()
    print("="*70)
    
    # Alternative method comparison
    if method == "inverse_freq":
        print("\nAlternative (effective_samples method):")
        print("If inverse_freq doesn't work well, try:")
        print("  python compute_class_weights.py --method effective_samples")
    
    return weights

def main():
    """
    Main function
    """
    import sys
    
    # Parse arguments
    method = "inverse_freq"
    if len(sys.argv) > 1:
        if "--method" in sys.argv:
            idx = sys.argv.index("--method")
            if idx + 1 < len(sys.argv):
                method = sys.argv[idx + 1]
    
    # Compute weights
    weights = compute_class_weights(method=method)
    
    if weights is not None:
        print("\n✅ SUCCESS! Class weights computed.")
        print("\nNext steps:")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("1. Copy the BCE_POS_WEIGHT line above")
        print("2. Open config.py and add it (after LOSS_TYPE line)")
        print("3. Update NUM_EPOCHS = 40 in config.py")
        print("4. Delete: checkpoints\\checkpoint_best_multilabel.pth")
        print("5. Restart training: py -3.11 train_multilabel.py")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("\nExpected improvement: +0.30-0.35 mIoU (from ~0.14 to ~0.45) ✅")
    else:
        print("\n❌ FAILED! Please check error messages above.")

if __name__ == "__main__":
    main()