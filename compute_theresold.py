# threshold_tuning.py
"""
Find optimal threshold by testing different values on validation set
"""

import torch
import numpy as np
from tqdm import tqdm
import config
from data.dataset_multilabel import get_dataloaders_multilabel
from models.deeplabv3 import get_deeplabv3
from utils.metrics_multilabel import MultiLabelSegmentationMetrics

def find_optimal_threshold(
    checkpoint_path="checkpoints/checkpoint_best_multilabel.pth",
    threshold_range=np.arange(0.10, 0.60, 0.05),  # Test 0.10, 0.15, ..., 0.55
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Test different thresholds and find the one with best mIoU
    """
    
    print("="*70)
    print("THRESHOLD OPTIMIZATION")
    print("="*70)
    print(f"Testing thresholds: {threshold_range}")
    print(f"Checkpoint: {checkpoint_path}")
    print()
    
    # Load model
    print("Loading model...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    num_labels = checkpoint.get('num_labels', 3)
    
    model = get_deeplabv3(config.MODEL_TYPE, num_labels, device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print(f"✓ Model loaded (num_labels={num_labels})")
    
    # Load validation data
    print("Loading validation data...")
    _, val_loader = get_dataloaders_multilabel(
        data_root=config.DATA_ROOT,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        num_labels=num_labels
    )
    print(f"✓ Validation set: {len(val_loader)} batches")
    print()
    
    # Collect all predictions and ground truth
    print("Running inference on validation set...")
    all_preds = []  # Will store sigmoid outputs [0, 1]
    all_masks = []  # Ground truth binary masks
    
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Inference"):
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)
            
            # Store
            all_preds.append(probs.cpu())
            all_masks.append(masks.cpu())
    
    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)  # [N, num_labels, H, W]
    all_masks = torch.cat(all_masks, dim=0)  # [N, num_labels, H, W]
    
    print(f"✓ Collected predictions: {all_preds.shape}")
    print()
    
    # Test different thresholds
    print("="*70)
    print("TESTING DIFFERENT THRESHOLDS:")
    print("="*70)
    print(f"{'Threshold':<12} {'mIoU':<12} {'mean_F1':<12} {'Precision':<12} {'Recall':<12}")
    print("-"*70)
    
    results = []
    best_threshold = None
    best_miou = 0.0
    
    for threshold in threshold_range:
        # Apply threshold to get binary predictions
        binary_preds = (all_preds > threshold).float()
        
        # Compute metrics
        metrics_calc = MultiLabelSegmentationMetrics(num_labels)
        
        for pred, mask in zip(binary_preds, all_masks):
            metrics_calc.update(pred.unsqueeze(0), mask.unsqueeze(0))
        
        metrics = metrics_calc.compute()
        
        # Store results
        results.append({
            'threshold': threshold,
            'miou': metrics['miou'],
            'mean_f1': metrics['mean_f1'],
            'mean_precision': metrics['mean_precision'],
            'mean_recall': metrics['mean_recall']
        })
        
        # Print
        print(f"{threshold:<12.2f} {metrics['miou']:<12.4f} {metrics['mean_f1']:<12.4f} "
              f"{metrics['mean_precision']:<12.4f} {metrics['mean_recall']:<12.4f}")
        
        # Track best
        if metrics['miou'] > best_miou:
            best_miou = metrics['miou']
            best_threshold = threshold
    
    print("="*70)
    print()
    
    # Summary
    print("="*70)
    print("RESULTS SUMMARY:")
    print("="*70)
    print(f"Best Threshold: {best_threshold:.2f}")
    print(f"Best mIoU: {best_miou:.4f}")
    print()
    
    # Show improvement
    current_threshold = getattr(config, 'THRESHOLD', 0.25)
    current_result = [r for r in results if abs(r['threshold'] - current_threshold) < 0.01]
    
    if current_result:
        current_miou = current_result[0]['miou']
        improvement = best_miou - current_miou
        improvement_pct = (improvement / current_miou) * 100
        
        print(f"Current threshold ({current_threshold:.2f}): mIoU = {current_miou:.4f}")
        print(f"Optimal threshold ({best_threshold:.2f}): mIoU = {best_miou:.4f}")
        print(f"Improvement: +{improvement:.4f} ({improvement_pct:+.1f}%)")
    
    print()
    print("="*70)
    print("UPDATE config.py:")
    print("="*70)
    print(f"THRESHOLD = {best_threshold:.2f}")
    print("="*70)
    
    return best_threshold, results

if __name__ == "__main__":
    best_threshold, results = find_optimal_threshold()
    
    # Optional: plot results
    try:
        import matplotlib.pyplot as plt
        
        thresholds = [r['threshold'] for r in results]
        mious = [r['miou'] for r in results]
        precisions = [r['mean_precision'] for r in results]
        recalls = [r['mean_recall'] for r in results]
        
        plt.figure(figsize=(12, 5))
        
        # Plot 1: mIoU vs Threshold
        plt.subplot(1, 2, 1)
        plt.plot(thresholds, mious, 'b-o', linewidth=2, markersize=6)
        plt.axvline(best_threshold, color='r', linestyle='--', label=f'Best: {best_threshold:.2f}')
        plt.xlabel('Threshold', fontsize=12)
        plt.ylabel('mIoU', fontsize=12)
        plt.title('mIoU vs Threshold', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 2: Precision/Recall vs Threshold
        plt.subplot(1, 2, 2)
        plt.plot(thresholds, precisions, 'g-o', label='Precision', linewidth=2, markersize=6)
        plt.plot(thresholds, recalls, 'r-s', label='Recall', linewidth=2, markersize=6)
        plt.axvline(best_threshold, color='k', linestyle='--', alpha=0.5, label=f'Best: {best_threshold:.2f}')
        plt.xlabel('Threshold', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Precision/Recall vs Threshold', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('threshold_analysis.png', dpi=150, bbox_inches='tight')
        print("\n✓ Plot saved as: threshold_analysis.png")
        
    except ImportError:
        print("\n(matplotlib not installed - skipping plot)")