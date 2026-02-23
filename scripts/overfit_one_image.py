"""
Overfit test on ONE image.
If the model cannot learn one image -> pipeline/loss issue.
"""

import os
import sys

# 🔥 IMPORTANT: add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

# -----------------------
# CONFIG (عدل المسارات لو لزم)
# -----------------------
IMG_PATH = r"C:\Users\Ismail Triki\Desktop\hani_dataset_memoire\dacl10k-segmentation\dataset\images\train\dacl10k_v2_train_0976.jpg"
MASK_PATH = r"C:\Users\Ismail Triki\Desktop\hani_dataset_memoire\dacl10k-segmentation\dataset\masks\train\dacl10k_v2_train_0976.png"

IMG_SIZE = (384, 384)
NUM_CLASSES = 20
STEPS = 300
LR = 0.001

# -----------------------
# Device (DirectML / CUDA / CPU)
# -----------------------
def get_device():
    if torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")
    try:
        import torch_directml
        if torch_directml.is_available():
            print("Using DirectML")
            return torch_directml.device(0)
    except:
        pass
    print("Using CPU")
    return torch.device("cpu")

device = torch.device("cpu")
# -----------------------
# Load Model
# -----------------------
from models.unet import get_model

model = get_model(
    model_type="unet",
    n_classes=NUM_CLASSES,
    device=device
)

# 🔥 IMPORTANT: use simple CE only
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -----------------------
# Load Image & Mask
# -----------------------
image = Image.open(IMG_PATH).convert("RGB")
mask = Image.open(MASK_PATH).convert("L")

# Resize
image = TF.resize(image, IMG_SIZE)
mask = TF.resize(mask, IMG_SIZE, interpolation=Image.NEAREST)

# To tensor
image = TF.to_tensor(image)
image = TF.normalize(image, mean=[0.485,0.456,0.406],
                               std=[0.229,0.224,0.225])
mask = torch.from_numpy(np.array(mask)).long()

# Add batch dim
image = image.unsqueeze(0).to(device)
mask = mask.unsqueeze(0).to(device)

print("Unique mask labels:", torch.unique(mask))

# -----------------------
# TRAIN LOOP (Overfit)
# -----------------------
model.train()

for step in range(STEPS):

    optimizer.zero_grad()
    outputs = model(image)
    loss = criterion(outputs, mask)
    loss.backward()
    optimizer.step()

    if step % 20 == 0:

        preds = torch.argmax(outputs, dim=1)

        bg_pred = (preds == 0).float().mean().item()
        unique_preds = torch.unique(preds).detach().cpu().numpy()

        print(f"\nStep {step}")
        print(f"Loss: {loss.item():.4f}")
        print(f"% predicted background: {bg_pred*100:.2f}%")
        print("Unique predicted labels:", unique_preds)

print("\nDONE.")