"""
Compute class weights for imbalanced DACL10K dataset
Run this once to generate weights for config.py
"""
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

NUM_CLASSES = 20
MASK_DIR = Path(r"C:\Users\Ismail Triki\Desktop\hani_dataset_memoire\dacl10k-segmentation\dataset\masks\train")

print("📊 Computing class weights from training masks...")
print(f"📂 Mask directory: {MASK_DIR}")

counts = np.zeros(NUM_CLASSES, dtype=np.float64)

masks = list(MASK_DIR.glob("*.png"))
print(f"📂 Found {len(masks)} training masks\n")

# Count pixels per class
for mask_path in tqdm(masks, desc="Analyzing masks"):
    try:
        arr = np.array(Image.open(mask_path))
        
        # Count pixels for each class
        for c in range(NUM_CLASSES):
            counts[c] += np.sum(arr == c)
    except Exception as e:
        print(f"⚠️ Error reading {mask_path}: {e}")
        continue

# Avoid division by zero
counts = np.maximum(counts, 1.0)

# Calculate weights (inverse frequency, log smoothed)
total_pixels = counts.sum()
freq = counts / total_pixels

# Inverse frequency with log smoothing (stable)
weights = 1.0 / np.log(1.02 + freq)
weights = weights / weights.mean()  # Normalize around 1.0

# Display results
print("\n" + "="*70)
print("CLASS DISTRIBUTION & WEIGHTS")
print("="*70)
print(f"{'Class':<5} {'Pixels':>12} {'Frequency':>12} {'Weight':>10}")
print("-"*70)

for i in range(NUM_CLASSES):
    print(f"{i:<5} {int(counts[i]):>12,} {freq[i]:>12.6f} {weights[i]:>10.4f}")

print("="*70)

# Print formatted for config.py
print("\n📋 Copy this to config.py:")
print("-"*70)
print("CLASS_WEIGHTS = [")
for w in weights:
    print(f"    {w:.6f},")
print("]")
print("-"*70)

print(f"\n✅ Done! Total pixels analyzed: {int(total_pixels):,}")
print(f"✅ Most common class: {np.argmax(counts)} (freq: {freq[np.argmax(counts)]:.2%})")
print(f"✅ Least common class: {np.argmin(counts)} (freq: {freq[np.argmin(counts)]:.2%})")