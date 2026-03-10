"""
Training Configuration for DACL10K Multi-Label (3 CLASSES)
Optimized for RTX 4070 12GB + Ryzen 9 7900X
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Classes: spalling (7), cavity (9), rust (11)
"""

import os

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PATHS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATA_ROOT = r"C:\Users\PC\Desktop\hani_seg\dacl10k-segmentation\dataset2"
MASKS_SUBDIR = "masks_multilabel"
IMAGES_SUBDIR = "images"
SAVE_DIR = "checkpoints2"
LOG_DIR = "logs"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATASET - 3 CLASSES ONLY! ✅
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NUM_LABELS = 3

# Class IDs from DACL10K (original dataset numbering)
CLASSES_TO_LOAD = [7, 9, 11]  # spalling=7, cavity=9, rust=11

# Names (order matches CLASSES_TO_LOAD)
CLASS_NAMES = ["spalling", "cavity", "rust"]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODEL_TYPE = "deeplabv3_resnet50"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HARDWARE - RTX 4070 CUDA ✅
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CPU_MODE = False
GPU_ID = 1  # CUDA (not DirectML)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRAINING - OPTIMIZED FOR 12GB VRAM ✅
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMG_SIZE = (512, 512)  # Higher resolution
BATCH_SIZE = 10  # Larger batch
NUM_WORKERS = 8  # Multi-threaded loading
NUM_EPOCHS = 20  # More epochs
EARLY_STOPPING_PATIENCE = 12

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OPTIMIZATION ✅
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OPTIMIZER = "adamw"
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9

# Learning Rate Scheduler
USE_SCHEDULER = True
SCHEDULER_TYPE = "cosine"
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 5
SCHEDULER_STEP_SIZE = 10

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LOSS - WITH CLASS WEIGHTS ✅
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LOSS_TYPE = "bce_dice"

# From compute_class_weights.py
# Order: [spalling, cavity, rust]
BCE_POS_WEIGHT = [3.0, 6.0, 2.5]

DICE_SMOOTH = 1.0

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AUGMENTATION - ENHANCED! ✅
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEFECT_CROP_PROB = 0.7
CROP_RATIO = 0.60
CROP_TRIES = 10
MIN_DEFECT_RATIO = 0.01

# Color augmentation
USE_COLOR_JITTER = True
COLOR_JITTER_BRIGHTNESS = 0.3
COLOR_JITTER_CONTRAST = 0.3
COLOR_JITTER_SATURATION = 0.3
COLOR_JITTER_HUE = 0.1

# Blur augmentation
USE_RANDOM_BLUR = True
BLUR_PROB = 0.3
BLUR_KERNEL_SIZES = [3, 5, 7]

# Noise augmentation
USE_RANDOM_NOISE = True
NOISE_PROB = 0.2
NOISE_STD = 0.02

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MIXED PRECISION (AMP) ✅
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USE_AMP = True  # RTX 4070 has Tensor Cores

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# INFERENCE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THRESHOLD = 0.5

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MISC
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RANDOM_SEED = 42
PRINT_FREQ = 20

# Create directories
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)