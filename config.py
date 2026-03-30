"""
Training Configuration for DACL10K Multi-Label (3 CLASSES)
Optimized for Kaggle GPU (Tesla T4 or P100)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Classes: crack (5), spalling (7), rust (11)
"""

import os

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PATHS – Kaggle dataset (read‑only) + working directories
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATA_ROOT = "/kaggle/input/hani_dataset/dataset2"          # <-- ADJUST IF DIFFERENT
MASKS_SUBDIR = "masks_multilabel"
IMAGES_SUBDIR = "images"
SAVE_DIR = "/kaggle/working/checkpoints2"
LOG_DIR = "/kaggle/working/logs"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATASET – 3 CLASSES (corrected IDs)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NUM_LABELS = 3
CLASSES_TO_LOAD = [5, 7, 11]          # crack=5, spalling=7, rust=11
CLASS_NAMES = ["crack", "spalling", "rust"]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODEL_TYPE = "deeplabv3_resnet50"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HARDWARE – Kaggle GPU
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CPU_MODE = False
GPU_ID = 0

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRAINING – increased epochs for better convergence
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMG_SIZE = (512, 512)
BATCH_SIZE = 8
NUM_WORKERS = 4
NUM_EPOCHS = 50                     # You can increase to 100 if you have time
EARLY_STOPPING_PATIENCE = 12

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OPTIMIZATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OPTIMIZER = "adamw"
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9

USE_SCHEDULER = True
SCHEDULER_TYPE = "cosine"
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 5
SCHEDULER_STEP_SIZE = 10

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LOSS – class weights (sqrt‑scaled from your computed values)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LOSS_TYPE = "bce_dice"
BCE_POS_WEIGHT = [18.09, 6.16, 6.95]   # order: crack, spalling, rust
DICE_SMOOTH = 1.0

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AUGMENTATION (keep as before)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEFECT_CROP_PROB = 0.7
CROP_RATIO = 0.60
CROP_TRIES = 10
MIN_DEFECT_RATIO = 0.01

USE_COLOR_JITTER = True
COLOR_JITTER_BRIGHTNESS = 0.3
COLOR_JITTER_CONTRAST = 0.3
COLOR_JITTER_SATURATION = 0.3
COLOR_JITTER_HUE = 0.1

USE_RANDOM_BLUR = True
BLUR_PROB = 0.3
BLUR_KERNEL_SIZES = [3, 5, 7]

USE_RANDOM_NOISE = True
NOISE_PROB = 0.2
NOISE_STD = 0.02

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MIXED PRECISION (AMP) – supported on Kaggle GPUs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USE_AMP = True

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# INFERENCE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THRESHOLD = 0.5

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MISC
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RANDOM_SEED = 42
PRINT_FREQ = 20

# Create writable directories
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
