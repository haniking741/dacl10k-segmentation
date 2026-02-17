# Save as scripts/visualize_samples.py
import random
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def visualize_samples(split="train", num_samples=6):
    img_dir = Path(f"dataset/images/{split}")
    mask_dir = Path(f"dataset/masks/{split}")
    
    images = list(img_dir.glob("*.jpg"))
    samples = random.sample(images, min(num_samples, len(images)))
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4*num_samples))
    
    for idx, img_path in enumerate(samples):
        mask_path = mask_dir / img_path.with_suffix(".png").name
        
        img = Image.open(img_path)
        mask = Image.open(mask_path)
        
        axes[idx, 0].imshow(img)
        axes[idx, 0].set_title(f"Image: {img_path.name}")
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(mask, cmap='tab20')
        axes[idx, 1].set_title(f"Mask: {mask_path.name}")
        axes[idx, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"visualization_{split}.png", dpi=150, bbox_inches='tight')
    print(f"✅ Saved visualization_{split}.png")
    plt.show()

if __name__ == "__main__":
    visualize_samples("train", num_samples=6)