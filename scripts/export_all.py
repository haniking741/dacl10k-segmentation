import os, json, shutil
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

# ==========================
# CONFIG
# ==========================
ROOT = Path(r"C:\Users\Informatics\Desktop\dataset_mémoire\dacl10k-DatasetNinja")

SPLITS = {
    "train": {"img": ROOT / "train" / "img", "ann": ROOT / "train" / "ann"},
    "val": {"img": ROOT / "val" / "img", "ann": ROOT / "val" / "ann"},
    "test": {"img": ROOT / "test" / "img", "ann": ROOT / "test" / "ann"},
}

OUT = Path(r"C:\Users\Informatics\Desktop\dataset_mémoire\segmentation_project\dataset2")
OUT_IMAGES = OUT / "images" # images/{split}/xxx.jpg
OUT_SEM = OUT / "masks_semantic" # masks_semantic/{split}/xxx.png (0..19)
OUT_ML = OUT / "masks_multilabel" # masks_multilabel/{split}/xxx_classYY.png (0/255)

# classes (0 background + 19 defects)
CLASS_NAMES = [
    "background",
    "graffiti",
    "drainage",
    "wetspot",
    "weathering",
    "crack",
    "rockpocket",
    "spalling",
    "washouts/concrete corrosion",
    "cavity",
    "efflorescence",
    "rust",
    "protective equipment",
    "exposed rebars",
    "bearing",
    "hollowareas",
    "joint tape",
    "restformwork",
    "alligator crack",
    "expansion joint",
]
NAME2ID = {name: i for i, name in enumerate(CLASS_NAMES)}

# Baseline paper uses "separate binary mask per defect/object"
# Usually exclude background => channels = 1..19 (19 channels)
ML_CLASS_IDS = list(range(1, 20)) # set to 18 if you want by removing one class

# ==========================
# Helpers
# ==========================
def ensure_dirs():
    for split in SPLITS.keys():
        (OUT_IMAGES / split).mkdir(parents=True, exist_ok=True)
        (OUT_SEM / split).mkdir(parents=True, exist_ok=True)
        (OUT_ML / split).mkdir(parents=True, exist_ok=True)

def clamp_points(points, w, h):
    out = []
    for x, y in points:
        x = max(0, min(int(round(x)), w - 1))
        y = max(0, min(int(round(y)), h - 1))
        out.append((x, y))
    return out

def json_path_for_image(ann_dir: Path, image_name: str) -> Path:
    return ann_dir / f"{image_name}.json" # datasetninja: <file>.jpg.json

def polygons_from_ann(ann_data):
    objects = ann_data.get("objects", [])
    for obj in objects:
        if obj.get("geometryType") != "polygon":
            continue
        cls = obj.get("classTitle", "").strip()
        pts = obj.get("points", {}).get("exterior", [])
        if cls and pts and len(pts) >= 3:
            yield cls, pts

def build_semantic_mask(w, h, ann_data):
    """
    single-channel semantic mask 0..19
    """
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    for cls_name, pts in polygons_from_ann(ann_data):
        cls_id = NAME2ID.get(cls_name, 0)
        pts = clamp_points(pts, w, h)
        draw.polygon(pts, fill=int(cls_id))
    return mask

def build_multilabel_masks(w, h, ann_data):
    """
    returns dict {class_id: PIL mask (0/255)}
    """
    masks = {cid: Image.new("L", (w, h), 0) for cid in ML_CLASS_IDS}
    draws = {cid: ImageDraw.Draw(masks[cid]) for cid in ML_CLASS_IDS}

    for cls_name, pts in polygons_from_ann(ann_data):
        cls_id = NAME2ID.get(cls_name, 0)
        if cls_id not in masks:
            continue
        pts = clamp_points(pts, w, h)
        draws[cls_id].polygon(pts, fill=255)
    return masks

def coco_init(categories):
    return {"images": [], "annotations": [], "categories": categories}

def coco_categories():
    # COCO categories: usually exclude background
    cats = []
    cid = 1
    for k in range(1, 20):
        cats.append({"id": cid, "name": CLASS_NAMES[k]})
        cid += 1
    return cats

def polygon_area(poly):
    # poly: [x1,y1,x2,y2,...]
    x = poly[0::2]
    y = poly[1::2]
    area = 0.0
    n = len(x)
    for i in range(n):
        j = (i + 1) % n
        area += x[i] * y[j] - x[j] * y[i]
    return abs(area) / 2.0

def bbox_from_poly(poly):
    xs = poly[0::2]
    ys = poly[1::2]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]

def build_coco_for_split(split_name, img_files):
    categories = coco_categories()
    coco = coco_init(categories)

    cat_name_to_coco_id = {c["name"]: c["id"] for c in categories}

    ann_id = 1
    img_id = 1

    for img_path in img_files:
        image_name = img_path.name
        ann_path = json_path_for_image(SPLITS[split_name]["ann"], image_name)

        if not ann_path.exists():
            # test may have labels; if missing, skip annotations but still include image
            with Image.open(img_path) as im:
                w, h = im.size
            coco["images"].append({"id": img_id, "file_name": image_name, "width": w, "height": h})
            img_id += 1
            continue

        with Image.open(img_path) as im:
            w, h = im.size

        with open(ann_path, "r", encoding="utf-8") as f:
            ann_data = json.load(f)

        coco["images"].append({"id": img_id, "file_name": image_name, "width": w, "height": h})

        for cls_name, pts in polygons_from_ann(ann_data):
            if cls_name == "background":
                continue
            if cls_name not in cat_name_to_coco_id:
                continue

            pts = clamp_points(pts, w, h)
            seg = []
            for x, y in pts:
                seg.extend([float(x), float(y)])

            if len(seg) < 6:
                continue

            area = polygon_area(seg)
            bbox = bbox_from_poly(seg)

            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_name_to_coco_id[cls_name],
                "segmentation": [seg],
                "area": float(area),
                "bbox": bbox,
                "iscrowd": 0
            })
            ann_id += 1

        img_id += 1

    return coco

# ==========================
# Main
# ==========================
def main():
    ensure_dirs()

    for split_name, paths in SPLITS.items():
        img_dir = paths["img"]
        ann_dir = paths["ann"]

        print("\n" + "=" * 100)
        print(f"Processing split: {split_name}")
        print(f"Images: {img_dir}")
        print(f"Ann : {ann_dir}")

        img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
        print(f"Found images: {len(img_files)}")

        created_sem = 0
        created_ml = 0
        missing_json = 0
        json_size_mismatch = 0

        for img_path in img_files:
            image_name = img_path.name
            ann_path = json_path_for_image(ann_dir, image_name)

            # copy image
            dst_img = OUT_IMAGES / split_name / image_name
            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)

            with Image.open(img_path) as im:
                w, h = im.size

            if not ann_path.exists():
                missing_json += 1
                continue

            with open(ann_path, "r", encoding="utf-8") as f:
                ann_data = json.load(f)

            jsz = ann_data.get("size", {})
            jw, jh = jsz.get("width", None), jsz.get("height", None)
            if (jw is not None and jh is not None) and (int(jw) != w or int(jh) != h):
                json_size_mismatch += 1

            # semantic mask
            sem = build_semantic_mask(w, h, ann_data)
            sem.save(OUT_SEM / split_name / (Path(image_name).stem + ".png"))
            created_sem += 1

            # multilabel masks
            ml_masks = build_multilabel_masks(w, h, ann_data)
            for cid, m in ml_masks.items():
                m.save(OUT_ML / split_name / f"{Path(image_name).stem}_class{cid:02d}.png")
            created_ml += 1

        print(f"✅ semantic masks created: {created_sem}")
        print(f"✅ multilabel sets created: {created_ml}")
        print(f"⚠️ missing json: {missing_json}")
        print(f"⚠️ json size mismatch count (report): {json_size_mismatch}")

        # COCO export
        coco = build_coco_for_split(split_name, img_files)
        coco_path = OUT / f"coco_instances_{split_name}.json"
        with open(coco_path, "w", encoding="utf-8") as f:
            json.dump(coco, f)
        print(f"✅ COCO saved: {coco_path}")

    print("\nDONE ✅")
    print("Outputs:")
    print(" - images:", OUT_IMAGES)
    print(" - semantic masks:", OUT_SEM)
    print(" - multilabel masks:", OUT_ML)
    print(" - coco jsons:", OUT)

if __name__ == "__main__":
    main()