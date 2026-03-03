"""
Training Configuration (SINGLE-LABEL MULTI-CLASS)
DACL10K - DeepLabV3+
"""

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
DATA_ROOT = r"C:\Users\Informatics\Desktop\dataset_mémoire\segmentation_project\dataset"

# 19 defects + background = 20 classes
NUM_CLASSES = 20   # 0 = background, 1..19 = defects

CLASS_NAMES = [
    "background",
    "graffiti", "drainage", "wetspot", "weathering", "crack",
    "rockpocket", "spalling", "washouts/concrete corrosion",
    "cavity", "efflorescence", "rust", "protective equipment",
    "exposed rebars", "bearing", "hollowareas", "joint tape",
    "restformwork", "alligator crack", "expansion joint",
]

# IMPORTANT: use original single masks folder
MASKS_SUBDIR = "masks"
IMAGES_SUBDIR = "images"

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
MODEL_TYPE = "resnet50"

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
CPU_MODE = False

IMG_SIZE = (512, 512)
BATCH_SIZE = 4
NUM_WORKERS = 4
NUM_EPOCHS = 50
PRINT_FREQ = 20

# ============================================================================
# AMP (Mixed Precision)
# ============================================================================
USE_AMP = True  # Only works on CUDA

# ============================================================================
# OPTIMIZER
# ============================================================================
OPTIMIZER = "adam"
LEARNING_RATE = 1e-4
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
# LOSS (SINGLE-LABEL)
# ============================================================================
LOSS_TYPE = "ce"
IGNORE_INDEX = 255  # If you have unlabeled pixels, else set to None

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
    print("TRAINING CONFIGURATION (SINGLE-LABEL MULTI-CLASS)")
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
    print(f"NUM_CLASSES: {NUM_CLASSES}")
    print(f"Num Workers: {NUM_WORKERS}")
    print(f"DATA_ROOT: {DATA_ROOT}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    get_config_summary()