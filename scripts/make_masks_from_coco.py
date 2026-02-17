#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, os, shutil, argparse
from pathlib import Path
from PIL import Image, ImageDraw

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def build_cat_mapping(categories, mode="contiguous"):
    """
    mode:
    - contiguous: 0=background, then 1..N by category order in COCO file
    - category_id: use raw COCO category_id values (may be non-contiguous)
    """
    if mode == "category_id":
        mapping = {c["id"]: c["id"] for c in categories}
        inv = {c["id"]: c["name"] for c in categories}
        return mapping, inv
    
    mapping = {}
    inv = {0: "background"}
    for idx, c in enumerate(categories, start=1):
        mapping[c["id"]] = idx
        inv[idx] = c["name"]
    return mapping, inv

def rasterize_polygons_to_mask(width, height, anns_for_img, cat_map):
    """
    Output: single-channel mask (L) where pixel value = class index.
    If polygons overlap, later ones overwrite earlier ones.
    """
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    for ann in anns_for_img:
        cat_id = ann.get("category_id")
        cls_val = cat_map.get(cat_id, 0)
        segs = ann.get("segmentation", [])
        
        if not isinstance(segs, list) or len(segs) == 0:
            continue
        
        for poly in segs:
            if not isinstance(poly, list) or len(poly) < 6:
                continue
            pts = [(poly[i], poly[i+1]) for i in range(0, len(poly) - 1, 2)]
            draw.polygon(pts, fill=int(cls_val))
    
    return mask

def index_images_recursively(root: Path):
    """
    يبني فهرس: اسم_الصورة.jpg -> مسارها الحقيقي داخل root (حتى لو داخل subfolders)
    إذا تكرر نفس الاسم في أكثر من مكان، يأخذ أول واحد ويطبع تحذير.
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    idx = {}
    
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            name = p.name
            if name in idx:
                # نادر جداً، لكن نبه فقط
                # (نترك الأول كما هو)
                continue
            idx[name] = p
    
    return idx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True, help="Path to instances_*.json")
    ap.add_argument("--src_img_dir", required=True, 
                    help="Root folder that contains images (can include subfolders)")
    ap.add_argument("--out_img_dir", required=True, 
                    help="Output folder dataset/images/train or val")
    ap.add_argument("--out_mask_dir", required=True, 
                    help="Output folder dataset/masks/train or val")
    ap.add_argument("--label_mode", default="contiguous", 
                    choices=["contiguous", "category_id"])
    ap.add_argument("--copy_images", action="store_true", 
                    help="Copy images into out_img_dir")
    ap.add_argument("--overwrite", action="store_true", 
                    help="Overwrite existing files")
    args = ap.parse_args()
    
    coco_path = Path(args.coco)
    src_img_dir = Path(args.src_img_dir)
    out_img_dir = Path(args.out_img_dir)
    out_mask_dir = Path(args.out_mask_dir)
    
    ensure_dir(out_img_dir)
    ensure_dir(out_mask_dir)
    
    data = json.loads(coco_path.read_text(encoding="utf-8"))
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])
    
    cat_map, inv_map = build_cat_mapping(categories, mode=args.label_mode)
    
    mapping_path = coco_path.parent / f"class_mapping_{coco_path.stem}_{args.label_mode}.json"
    mapping_path.write_text(json.dumps(inv_map, ensure_ascii=False, indent=2), encoding="utf-8")
    print("✅ Saved class mapping:", mapping_path)
    
    # group annotations by image_id
    ann_by_img = {}
    for ann in annotations:
        img_id = ann.get("image_id")
        if img_id is None:
            continue
        ann_by_img.setdefault(img_id, []).append(ann)
    
    # ✅ NEW: build recursive index for images
    if not src_img_dir.exists():
        print(f"❌ src_img_dir does not exist: {src_img_dir}")
        return
    
    print("🔎 Indexing images under:", src_img_dir)
    img_index = index_images_recursively(src_img_dir)
    print(f"✅ Indexed {len(img_index)} image files (recursive).")
    
    total = len(images)
    done = 0
    missing = 0
    
    for imginfo in images:
        img_id = imginfo["id"]
        file_name = imginfo["file_name"]
        w = int(imginfo.get("width", 0))
        h = int(imginfo.get("height", 0))
        
        # ✅ FIX: Extract just the base filename for lookup
        lookup_name = Path(file_name).name
        src_img_path = img_index.get(lookup_name)
        
        if not src_img_path or not src_img_path.exists():
            print(f"⚠ Missing image: {lookup_name}")
            missing += 1
            continue
        
        # copy image
        dst_img_path = out_img_dir / lookup_name
        if args.copy_images:
            if dst_img_path.exists() and not args.overwrite:
                pass
            else:
                shutil.copy2(src_img_path, dst_img_path)
        
        # create mask
        mask_name = Path(lookup_name).with_suffix(".png").name
        dst_mask_path = out_mask_dir / mask_name
        
        if dst_mask_path.exists() and not args.overwrite:
            done += 1
            continue
        
        anns_for_img = ann_by_img.get(img_id, [])
        mask = rasterize_polygons_to_mask(w, h, anns_for_img, cat_map)
        mask.save(dst_mask_path)
        
        done += 1
        if done % 200 == 0:
            print(f"Progress: {done}/{total}")
    
    print(f"✅ DONE: masks created for {done}/{total} images.")
    print(f"⚠ Missing images: {missing}")
    print("Images out:", out_img_dir)
    print("Masks out :", out_mask_dir)

if __name__ == "__main__":
    main()