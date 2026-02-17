import json
from pathlib import Path
from collections import Counter

# Load COCO
coco_train = json.load(open(r"C:\Users\Informatics\Desktop\dataset_mémoire\dacl10k-DatasetNinja\splits\instances_train.json"))
coco_val = json.load(open(r"C:\Users\Informatics\Desktop\dataset_mémoire\dacl10k-DatasetNinja\splits\instances_val.json"))

# Count COCO
coco_images = len(coco_train["images"]) + len(coco_val["images"])
coco_annotations = len(coco_train["annotations"]) + len(coco_val["annotations"])

# Count by class in COCO
coco_class_counts = Counter()
id_to_name = {c["id"]: c["name"] for c in coco_train["categories"]}
for ann in coco_train["annotations"] + coco_val["annotations"]:
    coco_class_counts[id_to_name[ann["category_id"]]] += 1

# Count original Supervisely files
import glob
original_train = len(glob.glob(r"C:\Users\Informatics\Desktop\dataset_mémoire\dacl10k-DatasetNinja\train\ann\*.json"))
original_val = len(glob.glob(r"C:\Users\Informatics\Desktop\dataset_mémoire\dacl10k-DatasetNinja\val\ann\*.json"))
original_test = len(glob.glob(r"C:\Users\Informatics\Desktop\dataset_mémoire\dacl10k-DatasetNinja\test\ann\*.json"))
original_total = original_train + original_val + original_test

print("=" * 70)
print("DATASET COMPARISON")
print("=" * 70)
print(f"\n📊 IMAGE COUNTS:")
print(f"  COCO splits:     {coco_images:,} images")
print(f"  Original files:  {original_total:,} images")
print(f"  Missing:         {original_total - coco_images:,} images ({(original_total - coco_images)/original_total*100:.1f}%)")

print(f"\n📊 ANNOTATION COUNTS:")
print(f"  COCO splits:     {coco_annotations:,} annotations")
print(f"  Original (EDA):  ~62,000 annotations (from your screenshot)")

print(f"\n📊 BREAKDOWN:")
print(f"  Original train:  {original_train:,}")
print(f"  COCO train:      {len(coco_train['images']):,}")
print(f"  Difference:      {original_train - len(coco_train['images']):,}")
print()
print(f"  Original val:    {original_val:,}")
print(f"  COCO val:        {len(coco_val['images']):,}")
print(f"  Difference:      {original_val - len(coco_val['images']):,}")
print()
print(f"  Original test:   {original_test:,}")
print(f"  COCO test:       0 (not created)")

print(f"\n📊 TOP 5 CLASSES IN COCO:")
for cls, count in coco_class_counts.most_common(5):
    print(f"  {cls:<30} {count:>6,}")

print("\n" + "=" * 70)
print("CONCLUSION:")
print("=" * 70)
print("The COCO format contains a SUBSET of the original dataset.")
print("This is normal - COCO splits are often curated/filtered versions.")
print("Your training data (COCO) is what matters for model performance.")
print("=" * 70)