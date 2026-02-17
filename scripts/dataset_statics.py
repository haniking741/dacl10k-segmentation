# Save as scripts/dataset_stats.py
import json
from pathlib import Path
from collections import Counter

def analyze_split(split_name):
    coco_path = Path(f"C:/Users/Informatics/Desktop/dataset_mémoire/dacl10k-DatasetNinja/splits/instances_{split_name}.json")
    
    if not coco_path.exists():
        print(f"⚠ File not found: {coco_path}")
        return None, None, None
    
    data = json.load(open(coco_path, encoding="utf-8"))
    
    # Get categories mapping
    categories = {c["id"]: c["name"] for c in data["categories"]}
    
    # Count annotations per class
    cat_counts = Counter([ann["category_id"] for ann in data["annotations"]])
    
    # Count images
    num_images = len(data["images"])
    num_annotations = len(data["annotations"])
    
    return cat_counts, categories, num_images, num_annotations

def main():
    splits = ["train", "val", "test"]
    
    all_counts = Counter()
    total_images = 0
    total_annotations = 0
    categories = None
    
    print("=" * 80)
    print("📊 DACL10K Dataset Statistics")
    print("=" * 80)
    
    for split in splits:
        cat_counts, cats, num_imgs, num_anns = analyze_split(split)
        
        if cat_counts is None:
            continue
        
        if categories is None:
            categories = cats
        
        all_counts += cat_counts
        total_images += num_imgs
        total_annotations += num_anns
        
        print(f"\n{'─' * 80}")
        print(f"📁 {split.upper()} SET")
        print(f"{'─' * 80}")
        print(f"  Images: {num_imgs}")
        print(f"  Annotations: {num_anns}")
        print(f"  Avg annotations per image: {num_anns/num_imgs:.2f}")
        print(f"\n  Class Distribution:")
        
        for cat_id, count in cat_counts.most_common():
            percentage = (count / num_anns) * 100
            print(f"    {categories[cat_id]:<30} {count:>6} ({percentage:>5.2f}%)")
    
    # Overall statistics
    print(f"\n{'=' * 80}")
    print(f"📊 OVERALL DATASET STATISTICS")
    print(f"{'=' * 80}")
    print(f"  Total Images: {total_images}")
    print(f"  Total Annotations: {total_annotations}")
    print(f"  Avg annotations per image: {total_annotations/total_images:.2f}")
    print(f"  Number of Classes: {len(categories)}")
    print(f"\n  Combined Class Distribution:")
    
    for cat_id, count in all_counts.most_common():
        percentage = (count / total_annotations) * 100
        print(f"    {categories[cat_id]:<30} {count:>6} ({percentage:>5.2f}%)")
    
    print(f"\n{'=' * 80}")
    
    # Class imbalance analysis
    max_count = max(all_counts.values())
    min_count = min(all_counts.values())
    imbalance_ratio = max_count / min_count
    
    print(f"\n⚠ Class Imbalance Analysis:")
    print(f"  Most frequent: {categories[all_counts.most_common(1)[0][0]]} ({max_count} instances)")
    print(f"  Least frequent: {categories[all_counts.most_common()[-1][0]]} ({min_count} instances)")
    print(f"  Imbalance ratio: {imbalance_ratio:.2f}x")
    
    if imbalance_ratio > 10:
        print(f"  ⚠ WARNING: High class imbalance detected!")
        print(f"     Consider using weighted loss or data augmentation")

if __name__ == "__main__":
    main()