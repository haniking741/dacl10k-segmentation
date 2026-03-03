"""
Training Configuration (MULTI-LABEL) for DACL10K DeepLabV3+
"""

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
DATA_ROOT = r"C:\Users\Informatics\Desktop\dataset_mémoire\segmentation_project\dataset2"

# Paper uses 18 (excludes background), but 19 also works
NUM_LABELS = 19  # Use 18 to match paper exactly

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
# METRICS CONFIGURATION
# ============================================================================
THRESHOLD = 0.25  # ← FIX: Multi-label threshold (was missing!)

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# Options:
# - 'deeplabv3_resnet50' (basic, no aux loss)
# - 'deeplabv3_resnet50_aux' (with aux loss - matches paper!)
# - 'deeplabv3_resnet101_aux' (larger, matches paper best model)

MODEL_TYPE = "deeplabv3_resnet50"  # ← Use this to match paper!

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
CPU_MODE = False

IMG_SIZE = (512, 512)  # Paper uses 512×512 ✓
BATCH_SIZE = 4
NUM_WORKERS = 4
NUM_EPOCHS = 50  # Paper uses 30, but more is fine
PRINT_FREQ = 20

# ============================================================================
# AMP (Mixed Precision)
# ============================================================================
USE_AMP = True  # DirectML doesn't support

# ============================================================================
# OPTIMIZER
# ============================================================================
OPTIMIZER = "adam"
LEARNING_RATE = 1e-4  # Paper uses multiple LRs, this is good
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9

# ============================================================================
# LR SCHEDULER
# ============================================================================
USE_SCHEDULER = True
SCHEDULER_TYPE = "cosine"  # Paper uses cosine ✓
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 5
SCHEDULER_STEP_SIZE = 10

# ============================================================================
# LOSS (MULTI-LABEL)
# ============================================================================
LOSS_TYPE = "bce_dice"  # Paper uses Dice, but BCE+Dice is better

BCE_POS_WEIGHT = None  # Can add class weights here
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
GPU_ID = 1
USE_MULTI_GPU = False


def get_config_summary():
    print("\n" + "=" * 70)
    print("TRAINING CONFIGURATION (MULTI-LABEL)")
    print("=" * 70)
    print(f"Mode: {'CPU' if CPU_MODE else 'GPU'}")
    print(f"Model: {MODEL_TYPE}")
    print(f"Image Size: {IMG_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Num Epochs: {NUM_EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Optimizer: {OPTIMIZER}")
    print(f"Loss: {LOSS_TYPE}")
    print(f"Threshold: {THRESHOLD}")  # ← Show threshold
    print(f"AMP: {USE_AMP}")
    print(f"NUM_LABELS: {NUM_LABELS}")
    print(f"Num Workers: {NUM_WORKERS}")
    print(f"DATA_ROOT: {DATA_ROOT}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    get_config_summary()
