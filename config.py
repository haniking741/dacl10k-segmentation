"""
Training Configuration (MULTI-LABEL) for DACL10K DeepLabV3+
"""

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
DATA_ROOT = r"C:\Users\Ismail Triki\Desktop\hani_dataset_memoire\dacl10k-segmentation\dataset2"

NUM_LABELS = 19

CLASS_NAMES = [
    "graffiti", "drainage", "wetspot", "weathering", "crack",
    "rockpocket", "spalling", "washouts/concrete corrosion", 
    "cavity", "efflorescence", "rust", "protective equipment",
    "exposed rebars", "bearing", "hollowareas", "joint tape",
    "restformwork", "alligator crack", "expansion joint",
]

MASKS_SUBDIR = "masks_multilabel"
IMAGES_SUBDIR = "images"

# ============================================================================
# MODEL CONFIGURATION - CRITICAL!
# ============================================================================
# Choose one:
# - 'unet' (standard U-Net)
# - 'unet_lite' (lightweight U-Net)
# - 'deeplabv3_resnet50' (DeepLabV3+ with ResNet50 - FASTER)
# - 'deeplabv3_resnet101' (DeepLabV3+ with ResNet101 - BETTER)

MODEL_TYPE = "deeplabv3_resnet50"  # ← CHANGED!

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
CPU_MODE = False

IMG_SIZE = (512, 512)  # Keep same for fair comparison
BATCH_SIZE = 2
NUM_WORKERS = 0
NUM_EPOCHS = 50
PRINT_FREQ = 20

# ============================================================================
# AMP (Mixed Precision)
# ============================================================================
USE_AMP = False  # Keep False for DirectML

# ============================================================================
# OPTIMIZER
# ============================================================================
OPTIMIZER = "adam"
LEARNING_RATE = 1e-4  # ← Slightly lower for DeepLabV3+
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9

# ============================================================================
# LR SCHEDULER
# ============================================================================
USE_SCHEDULER = True
SCHEDULER_TYPE = "cosine"
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 5
SCHEDULER_STEP_SIZE = 10

# ============================================================================
# LOSS (MULTI-LABEL)
# ============================================================================
LOSS_TYPE = "bce_dice"

BCE_POS_WEIGHT = None
DICE_SMOOTH = 1.0

# ============================================================================
# CROP / AUGMENTATION
# ============================================================================
DEFECT_CROP_PROB = 0.7
CROP_RATIO = 0.60
CROP_TRIES = 10
MIN_DEFECT_RATIO = 0.01

# ============================================================================
# CHECKPOINTS
# ============================================================================
SAVE_DIR = "checkpoints"
LOG_DIR = "logs"
SAVE_BEST_ONLY = True
EARLY_STOPPING_PATIENCE = 15
RESUME_CHECKPOINT = None
VAL_FREQUENCY = 1

# ============================================================================
# MISC
# ============================================================================
RANDOM_SEED = 42
GPU_ID = 1  # Your DirectML device
USE_MULTI_GPU = False


def get_config_summary():
    print("\n" + "=" * 70)
    print("TRAINING CONFIGURATION (MULTI-LABEL)")
    print("=" * 70)
    print(f"Mode: {'CPU' if CPU_MODE else 'GPU'}")
    print(f"Model: {MODEL_TYPE}")  # ← Will show 'deeplabv3_resnet50'
    print(f"Image Size: {IMG_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Num Epochs: {NUM_EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Optimizer: {OPTIMIZER}")
    print(f"Loss: {LOSS_TYPE}")
    print(f"AMP: {USE_AMP}")
    print(f"NUM_LABELS: {NUM_LABELS}")
    print(f"Num Workers: {NUM_WORKERS}")
    print(f"DATA_ROOT: {DATA_ROOT}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    get_config_summary()