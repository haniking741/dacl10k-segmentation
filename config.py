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
DATA_ROOT = r"C:\Users\Informatics\Desktop\dataset_mémoire\segmentation_project\dataset2"
MASKS_SUBDIR = "masks_multilabel"
IMAGES_SUBDIR = "images"
SAVE_DIR = "checkpoints"
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
GPU_ID = 0  # CUDA (not DirectML)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRAINING - OPTIMIZED FOR 12GB VRAM ✅
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMG_SIZE = (512, 512)  # Higher resolution
BATCH_SIZE = 8  # Larger batch
NUM_WORKERS = 8  # Multi-threaded loading
NUM_EPOCHS = 50  # More epochs
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
BCE_POS_WEIGHT = [0.3260640869076145, 2.261665349