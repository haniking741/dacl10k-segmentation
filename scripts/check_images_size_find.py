# debug_data.py
import torch
from data.dataset_multilabel import get_dataloaders_multilabel
import config

print(“Loading data…”)
train_loader, val_loader, num_labels = get_dataloaders_multilabel(
    Data_root=config.DATA_ROOT,
    Batch_size=2,
    Num_workers=0,
    Img_size=(512, 512),
    Images_subdir=”images”,
    Masks_subdir=”masks_multilabel”,
    Cpu_mode=False,
)

print(f"\n✅ Loaded {len(train_loader)} batches")
print(f"✅ Num labels: {num_labels}")

# Check first batch
Images, masks = next(iter(val_loader))

print(f"\n📊 Batch shapes:")
print(f" Images: {images.shape}")
print(f"  Masks: {masks.shape}")
print(f"  Masks dtype: {masks.dtype}")
print(f" Masks min/max: {masks.min()}/{masks.max()}")

print(f"\n🔍 Per-class mask stats:")
for c in range(min(19, masks.shape[1])):
    count = masks[0, c].sum().item()
    print(f"  Class {c:02d}: {count:.0f} positive pixels")

total_pos = masks[0].sum().item()
total_pixels = masks[0].numel()
Print(f"\n📊 Total positive pixels: {total_pos:.0f} / {total_pixels}")
Print(f"   Ratio: {100*total_pos/total_pixels:.2f}%")

if total_pos == 0:
    Print("\n🚨 ERROR: ALL MASKS ARE ZERO!")
    Print("Your masks are empty or not loading!")
elif total_pos > total_pixels:
    Print("\n✅ GOOD: Multi-label overlap detected!")
Else:
    Print(f"\n⚠️ Suspicious: No overlap? Check if truly multi-label")

