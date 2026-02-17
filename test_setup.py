# Replace your test_setup.py with this:
"""
Quick Test Script - Run this FIRST to verify everything works
Tests dataset loading, model creation, and one training step
"""
import torch
import sys
import os

def run_tests():
    # Add parent directory to path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    print("="*70)
    print("🧪 QUICK TEST - Verifying Setup")
    print("="*70 + "\n")

    # Test 1: Import modules
    print("1️⃣ Testing imports...")
    try:
        import config
        from models.unet import get_model
        from data.dataset import get_dataloaders
        from utils.metrics import SegmentationMetrics
        print("   ✅ All imports successful\n")
    except Exception as e:
        print(f"   ❌ Import failed: {e}\n")
        sys.exit(1)

    # Test 2: Check dataset
    print("2️⃣ Checking dataset...")
    try:
        if not os.path.exists(config.DATA_ROOT):
            print(f"   ❌ Dataset not found at: {config.DATA_ROOT}")
            print(f"   Please update DATA_ROOT in config.py")
            sys.exit(1)
        
        train_img_dir = os.path.join(config.DATA_ROOT, "images", "train")
        train_mask_dir = os.path.join(config.DATA_ROOT, "masks", "train")
        
        if not os.path.exists(train_img_dir):
            print(f"   ❌ Train images not found: {train_img_dir}")
            sys.exit(1)
        
        n_train = len([f for f in os.listdir(train_img_dir) if f.endswith(('.jpg', '.png'))])
        n_masks = len([f for f in os.listdir(train_mask_dir) if f.endswith('.png')])
        
        print(f"   ✅ Found {n_train} training images")
        print(f"   ✅ Found {n_masks} training masks\n")
    except Exception as e:
        print(f"   ❌ Dataset check failed: {e}\n")
        sys.exit(1)

    # Test 3: Load one batch
    print("3️⃣ Loading one batch...")
    try:
        train_loader, val_loader, num_classes = get_dataloaders(
            data_root=config.DATA_ROOT,
            batch_size=2,
            num_workers=0,  # Must be 0 for Windows
            img_size=(256, 256),
            cpu_mode=True
        )
        
        images, masks = next(iter(train_loader))
        print(f"   ✅ Batch shape: images={images.shape}, masks={masks.shape}\n")
    except Exception as e:
        print(f"   ❌ Data loading failed: {e}\n")
        sys.exit(1)

    # Test 4: Create model
    print("4️⃣ Creating model...")
    try:
        device = torch.device('cpu')
        model = get_model('unet_lite', n_classes=20, device=device)
        print(f"   ✅ Model created successfully\n")
    except Exception as e:
        print(f"   ❌ Model creation failed: {e}\n")
        sys.exit(1)

    # Test 5: Forward pass
    print("5️⃣ Testing forward pass...")
    try:
        model.eval()
        with torch.no_grad():
            outputs_test = model(images)
        print(f"   ✅ Forward pass successful: output shape={outputs_test.shape}\n")
    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}\n")
        sys.exit(1)

    # Test 6: Loss computation
    print("6️⃣ Testing loss computation...")
    try:
        criterion = torch.nn.CrossEntropyLoss()
        model.train()  # Enable training mode
        outputs = model(images)  # New forward pass with gradients
        loss = criterion(outputs, masks)
        print(f"   ✅ Loss computed: {loss.item():.4f}\n")
    except Exception as e:
        print(f"   ❌ Loss computation failed: {e}\n")
        sys.exit(1)

    # Test 7: Backward pass
    print("7️⃣ Testing backward pass...")
    try:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"   ✅ Backward pass successful\n")
    except Exception as e:
        print(f"   ❌ Backward pass failed: {e}\n")
        sys.exit(1)
    # Test 8: Metrics
    print("8️⃣ Testing metrics...")
    try:
        metrics = SegmentationMetrics(num_classes=20)
        preds = torch.argmax(outputs, dim=1)
        metrics.update(preds, masks)
        results = metrics.get_metrics()
        print(f"   ✅ mIoU: {results['mIoU']:.4f}\n")
    except Exception as e:
        print(f"   ❌ Metrics computation failed: {e}\n")
        sys.exit(1)

    print("="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
    print("\n📋 Next Steps:")
    print("1. For LOCAL CPU testing (slow): python train.py")
    print("2. For GPU training (fast): Transfer to powerful computer")
    print("\n⚠️  CPU training will be VERY slow. Recommend using GPU!")
    print("="*70 + "\n")

if __name__ == "__main__":
    run_tests()