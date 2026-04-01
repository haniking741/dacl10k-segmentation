import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import random

DATA_ROOT = Path(r"C:\Users\Informatics\Desktop\dataset_mémoire\segmentation_project\dataset2")
SPLIT = "train"
IMG_DIR = DATA_ROOT / "images" / SPLIT
MSK_DIR = DATA_ROOT / "masks_multilabel" / SPLIT

CLASS_NAMES = [
    "graffiti", "drainage", "wetspot", "weathering", "crack",
    "rockpocket", "spalling", "washouts/concrete corrosion",
    "cavity", "efflorescence", "rust", "protective equipment",
    "exposed rebars", "bearing", "hollowareas", "joint tape",
    "restformwork", "alligator crack", "expansion joint",
]
def load_multilabel_masks(stem: str, H: int, W: int):
    masks = np.zeros((19, H, W), dtype=np.uint8)
    active = []
    for c in range(1, 20):
        p = MSK_DIR / f"{stem}_class{c:02d}.png"
        if not p.exists():
            continue
        m = np.array(Image.open(p).convert("L"))
        m = (m > 0).astype(np.uint8)
        masks[c-1] = m
        if m.any():
            active.append(c)  # 1..19
    return masks, active

def overlay_per_class(img_rgb, masks_19, alpha=0.45):
    """
    يرسم كل كلاس بلون مختلف.
    ملاحظة مهمة: لو عندك تداخل (نفس البكسل = 1 في أكثر من كلاس)
    فسنختار آخر كلاس في الترتيب ليظهر فوق (يمكن تغيير المنطق بسهولة).
    """
    img = img_rgb.astype(np.float32).copy()
    H, W = img.shape[:2]

    # ألوان ثابتة (19 لون) من colormap
    cmap = plt.get_cmap("tab20")
    colors = np.array([
    [255, 0, 0],      # 1 graffiti - red
    [0, 255, 0],      # 2 drainage - green
    [0, 0, 255],      # 3 wetspot - blue
    [255, 255, 0],    # 4 weathering - yellow
    [255, 0, 255],    # 5 crack - magenta
    [0, 255, 255],    # 6 rockpocket - cyan
    [128, 0, 0],      # 7 spalling - dark red
    [0, 128, 0],      # 8 corrosion - dark green
    [0, 0, 128],      # 9 cavity - dark blue
    [255, 128, 0],    # 10 efflorescence - orange
    [128, 0, 255],    # 11 rust - purple
    [0, 128, 255],    # 12 protective equipment
    [128, 128, 0],    # 13 exposed rebars
    [128, 0, 128],    # 14 bearing
    [0, 128, 128],    # 15 hollowareas
    [64, 0, 0],       # 16 joint tape
    [0, 64, 0],       # 17 restformwork
    [0, 0, 64],       # 18 alligator crack
    [192, 192, 192],  # 19 expansion joint
], dtype=np.float32)

    # خريطة من 0..19: 0 = لا شيء, k = كلاس k
    label_map = np.zeros((H, W), dtype=np.int16)

    # لو يوجد تداخل: هذا يجعل الكلاس الذي يأتي "أخيرا" يطغى بصريًا
    for k in range(19):
        m = masks_19[k].astype(bool)
        label_map[m] = (k + 1)

    out = img.copy()
    for k in range(19):
        m = (label_map == (k + 1))
        if not m.any():
            continue
        color = colors[k]
        out[m] = out[m] * (1 - alpha) + color * alpha

    return out.astype(np.uint8), label_map

# -------- Random image every run ----------
img_paths = sorted(list(IMG_DIR.glob("*.jpg")) + list(IMG_DIR.glob("*.png")))
img_path = random.choice(img_paths)  # ✅ random every run
stem = img_path.stem

img = np.array(Image.open(img_path).convert("RGB"))
H, W = img.shape[:2]
masks, active = load_multilabel_masks(stem, H, W)

overlay, label_map = overlay_per_class(img, masks, alpha=0.45)

active_names = [CLASS_NAMES[c-1] for c in active]
print("Image:", img_path.name, "size:", (W, H))
print("Active classes:", active, active_names)

plt.figure(figsize=(16, 7))
plt.subplot(1, 2, 1)
plt.title(f"Original (full-res) {W}x{H}")
plt.imshow(img)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Overlay (different color per class)")
plt.imshow(overlay)
plt.axis("off")

# Legend مختصر للكلاسات الموجودة فقط
if len(active) > 0:
    # عرض أسماء الكلاسات النشطة في أسفل الشكل
    txt = " | ".join([f"{c:02d}:{CLASS_NAMES[c-1]}" for c in active])
    plt.gcf().text(0.5, 0.02, txt, ha="center", fontsize=10)

plt.tight_layout()
plt.show()