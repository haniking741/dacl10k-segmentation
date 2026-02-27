from pathlib import Path
import numpy as np
from PIL import Image

root = Path(r"C:\Users\Informatics\Desktop\dataset_mémoire\segmentation_project\dataset2\masks_multilabel\train")

bad = 0
for p in root.glob("*_class*.png"):
    u = np.unique(np.array(Image.open(p)))
    if not set(u.tolist()).issubset({0,255}):
        bad += 1
        print("NOT binary:", p.name, "unique:", u[:10])
        if bad > 10:
            break

print("bad files:", bad)