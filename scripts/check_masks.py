import os
import sys

# Add project root to Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)


import torch
import config
from data.dataset_multilabel import get_dataloaders_multilabel

train_loader, _, num_labels = get_dataloaders_multilabel(
    data_root=config.DATA_ROOT,
    batch_size=2,
    num_workers=0,
    img_size=config.IMG_SIZE,
    images_subdir=config.IMAGES_SUBDIR,
    masks_subdir=config.MASKS_SUBDIR,
    cpu_mode=True,
)

class_pixel_count = torch.zeros(num_labels)

for imgs, masks in train_loader:
    class_pixel_count += masks.sum(dim=(0, 2, 3))
    
print("Pixels per class:")
for i, count in enumerate(class_pixel_count):
    print(f"Class {i}: {count.item()}")
