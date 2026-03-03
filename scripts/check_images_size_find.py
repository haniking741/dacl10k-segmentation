import os
import sys

# Add project root to Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
import numpy as np
from PIL import Image
from pathlib import Path

root = Path(r"...\dataset2\masks_multilabel\train")
stem = "dacl10k_v2_train_0000_class01"  # بدّل باسم موجود

stack = []
for k in range(1,20):
    p = root / f"{stem}_class{k:02d}.png"
    m = np.array(Image.open(p))
    stack.append((m>0).astype(np.uint8))
stack = np.stack(stack, axis=0)  # [19,H,W]
overlap = (stack.sum(axis=0) > 1).mean()
print("overlap ratio:", overlap)