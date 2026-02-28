"""
Training Configuration (MULTI-LABEL) for DACL10K U-Net
- Multi-label segmentation: 19 binary masks (class01..class19)
- Output channels = 19
"""

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
DATA_ROOT = r"C:\Users\Ismail Triki\Desktop\hani_dataset_memoire\dacl10k-segmentation\dataset2"

# Multi-label = 19 classes (بدون background كقناة)
NUM_LABELS = 19

CLASS_NAMES = [
    "graffiti", # 01
    "drainage", # 02
    "wetspot", # 03
    "weathering", # 04
    "crack", # 05
    "rockpocket", # 06
    "spalling", # 07
    "washouts/concrete corrosion", # 08
    "cavity", # 09
    "efflorescence", # 10
    "rust", # 11
    "protective equipment", # 12
    "exposed rebars", # 13
    "bearing", # 14
    "hollowareas", # 15
    "joint tape", # 16
    "restformwork", # 17
    "alligator crack", # 18
    "expansion joint", # 19
]

# أين توجد masks multi-label
MASKS_SUBDIR = "masks_multilabel" # <-- مهم
IMAGES_SUBDIR = "images"

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
CPU_MODE = False # على RTX 4070 خليها False

MODEL_TYPE = "unet"
IMG_SIZE = (384, 384)
BATCH_SIZE = 2
NUM_WORKERS = 0
NUM_EPOCHS = 50
PRINT_FREQ = 20

# ============================================================================
# AMP (Mixed Precision)
# ============================================================================
USE_AMP = False # ✅ CUDA only

# ============================================================================
# OPTIMIZER
# ============================================================================
OPTIMIZER = "adam"
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9

# ============================================================================
# LR SCHEDULER
# ============================================================================
USE_SCHEDULER = True
SCHEDULER_TYPE = "cosine" # cosine | step | plateau
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 5
SCHEDULER_STEP_SIZE = 10

# ============================================================================
# LOSS (MULTI-LABEL)
# ============================================================================
LOSS_TYPE = "bce_dice" # bce | dice | bce_dice

BCE_POS_WEIGHT = None
# إذا تحب weights لكل class، هنا تحط list طولها 19
# مثال: BCE_POS_WEIGHT = [1.0, 2.0, ...] (اختياري)

DICE_SMOOTH = 1.0

# ============================================================================
# CROP / AUGMENTATION
# ============================================================================
DEFECT_CROP_PROB = 0.2
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
    print(f"AMP: {USE_AMP}")
    print(f"NUM_LABELS: {NUM_LABELS}")
    print(f"Num Workers: {NUM_WORKERS}")
    print(f"DATA_ROOT: {DATA_ROOT}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    get_config_summary()
