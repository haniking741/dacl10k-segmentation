# ============================================================================
# TEST EVALUATION – LOAD BEST CHECKPOINT AND COMPUTE METRICS ON TEST SET
# ============================================================================

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# ------------------------------------------------------------
# Step 0: Change to the repository directory and add to path
# ------------------------------------------------------------
repo_path = "/kaggle/working/dacl10k-segmentation"
if os.path.exists(repo_path):
    os.chdir(repo_path)
    sys.path.insert(0, repo_path)   # ensure we can import config and other modules
    print(f"Changed working directory to: {os.getcwd()}")
else:
    raise FileNotFoundError(f"Repository not found at {repo_path}")

# Now import config and other modules from the repo
import config
from models.deeplabv3 import get_model
from data.dataset_multilabel import DACL10KMultiLabelDataset
from utils.metrics_multilabel import MultiLabelSegmentationMetrics

# ------------------------------------------------------------
# 1. Device
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------------------------------------
# 2. Test dataset loader – check if test folder exists, else use val
# ------------------------------------------------------------
test_img_dir  = os.path.join(config.DATA_ROOT, config.IMAGES_SUBDIR, "test")
test_mask_dir = os.path.join(config.DATA_ROOT, config.MASKS_SUBDIR, "test")

# If test folder does not exist, use validation set instead
if not os.path.exists(test_img_dir):
    print(f"⚠️ Test folder not found: {test_img_dir}")
    print("Using validation set for evaluation instead.")
    test_img_dir  = os.path.join(config.DATA_ROOT, config.IMAGES_SUBDIR, "val")
    test_mask_dir = os.path.join(config.DATA_ROOT, config.MASKS_SUBDIR, "val")

# Verify that the folder now exists
if not os.path.exists(test_img_dir):
    raise FileNotFoundError(f"Neither test nor val folder found at {test_img_dir}")

# Create dataset with no augmentation
test_dataset = DACL10KMultiLabelDataset(
    img_dir=test_img_dir,
    mask_dir=test_mask_dir,
    classes_to_load=config.CLASSES_TO_LOAD,
    transform=False,               # no augmentation for testing
    img_size=config.IMG_SIZE,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
    pin_memory=True,
)
print(f"Test/Val images : {len(test_dataset)}")
print(f"Test/Val batches: {len(test_loader)}")

# ------------------------------------------------------------
# 3. Load best checkpoint
# ------------------------------------------------------------
checkpoint_path = os.path.join(config.SAVE_DIR, "checkpoint_best_multilabel.pth")
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

# Create model with same configuration
model = get_model(config.MODEL_TYPE, test_dataset.num_labels, device)

checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')} (best mIoU: {checkpoint.get('best_miou', 0):.4f})")

# ------------------------------------------------------------
# 4. Evaluate on test/val set
# ------------------------------------------------------------
metrics = MultiLabelSegmentationMetrics(
    num_classes=test_dataset.num_labels,
    class_names=config.CLASS_NAMES,
    threshold=getattr(config, "THRESHOLD", 0.5),
    ignore_empty=True,
)
metrics.reset()

with torch.no_grad():
    for images, masks in test_loader:
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        metrics.update(logits, masks)

test_metrics = metrics.get_metrics()
print("\n" + "=" * 60)
print("EVALUATION RESULTS (on " + ("test" if "test" in test_img_dir else "validation") + " set)")
print("=" * 60)
print(f"Mean IoU       : {test_metrics['mean_IoU']:.4f}")
print(f"Mean F1        : {test_metrics['mean_F1']:.4f}")
print(f"Mean Precision : {test_metrics['mean_Precision']:.4f}")
print(f"Mean Recall    : {test_metrics['mean_Recall']:.4f}")
print("\nPer‑class IoU:")
for cls in test_metrics['per_class']:
    if cls['valid']:
        print(f"  {cls['name']:12s}: {cls['IoU']:.4f}")

# ------------------------------------------------------------
# 5. (Optional) Visualize a few predictions
# ------------------------------------------------------------
def unnormalize(tensor):
    """Reverse ImageNet normalization for display."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    return tensor * std + mean

model.eval()
with torch.no_grad():
    images, masks = next(iter(test_loader))
    images = images.to(device)
    masks = masks.to(device)
    logits = model(images)
    probs = torch.sigmoid(logits)
    preds = (probs >= config.THRESHOLD).float()

num_samples = min(3, len(images))
fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
if num_samples == 1:
    axes = axes.reshape(1, -1)  # make indexing consistent

for i in range(num_samples):
    # original image
    img = unnormalize(images[i].cpu()).clamp(0,1).permute(1,2,0).numpy()
    axes[i,0].imshow(img)
    axes[i,0].set_title("Original")
    axes[i,0].axis('off')

    # ground truth (RGB overlay for 3 classes)
    gt = masks[i].cpu().numpy().transpose(1,2,0)
    gt_rgb = np.zeros((gt.shape[0], gt.shape[1], 3))
    if gt.shape[2] >= 3:
        gt_rgb[...,0] = gt[...,0]   # class 0
        gt_rgb[...,1] = gt[...,1]   # class 1
        gt_rgb[...,2] = gt[...,2]   # class 2
    axes[i,1].imshow(gt_rgb)
    axes[i,1].set_title("Ground truth")
    axes[i,1].axis('off')

    # prediction
    pred = preds[i].cpu().numpy().transpose(1,2,0)
    pred_rgb = np.zeros((pred.shape[0], pred.shape[1], 3))
    if pred.shape[2] >= 3:
        pred_rgb[...,0] = pred[...,0]
        pred_rgb[...,1] = pred[...,1]
        pred_rgb[...,2] = pred[...,2]
    axes[i,2].imshow(pred_rgb)
    axes[i,2].set_title("Prediction")
    axes[i,2].axis('off')

plt.tight_layout()
plt.show()
