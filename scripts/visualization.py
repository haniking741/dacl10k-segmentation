import os
import sys

# Add project root to Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
import matplotlib.pyplot as plt
import config
from data.dataset_multilabel import get_dataloaders_multilabel

train_loader, _, _ = get_dataloaders_multilabel(
    data_root=config.DATA_ROOT,
    batch_size=1,
    num_workers=0,
    img_size=config.IMG_SIZE,
    images_subdir=config.IMAGES_SUBDIR,
    masks_subdir=config.MASKS_SUBDIR,
    cpu_mode=True,
)

imgs, masks = next(iter(train_loader))

img = imgs[0].permute(1,2,0).numpy()
mask = masks[0][0].numpy() # class 0

plt.subplot(1,2,1)
plt.imshow(img)
plt.title("Image")

plt.subplot(1,2,2)
plt.imshow(mask)
plt.title("Mask (class 0)")

plt.show()