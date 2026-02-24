"""
Training Configuration for DACL10K U-Net
Optimized for: RX 6600 8GB + Windows + torch-directml
"""

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
DATA_ROOT = r"C:\Users\Informatics\Desktop\dataset_mémoire\segmentation_project\dataset"
NUM_CLASSES = 20  # 19 defect classes + 1 background

CLASS_NAMES = [
    "background",
    "graffiti",
    "drainage",
    "wetspot",
    "weathering",
    "crack",
    "rockpocket",
    "spalling",
    "washouts/concrete corrosion",
    "cavity",
    "efflorescence",
    "rust",
    "protective equipment",
    "exposed rebars",
    "bearing",
    "hollowareas",
    "joint tape",
    "restformwork",
    "alligator crack",
    "expansion joint",
]

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
CPU_MODE = False  # GPU mode

if CPU_MODE:
    MODEL_TYPE = "unet_lite"
    IMG_SIZE = (256, 256)
    BATCH_SIZE = 2
    NUM_WORKERS = 0
    NUM_EPOCHS = 2
    PRINT_FREQ = 50
else:
    # ✅ RTX 4070 OPTIMIZED SETTINGS
    MODEL_TYPE = "unet"
    IMG_SIZE = (1024, 1024)    # ✅ CHANGED: Better quality than 256
    BATCH_SIZE = 2           # ✅ OK for RX 6600 8GB
    NUM_WORKERS = 6          # ✅ Required for Windows
    NUM_EPOCHS = 50
    PRINT_FREQ = 20

# ============================================================================
# MIXED PRECISION (AMP)
# ============================================================================
USE_AMP = True  # ✅ Keep OFF for DirectML

# ============================================================================
# OPTIMIZER CONFIGURATION
# ============================================================================
# ✅ CHANGED: Adam is easier and works better
OPTIMIZER = "adam"          # ✅ CHANGED from 'sgd' to 'adam'
LEARNING_RATE = 0.001        # ✅ CHANGED from 0.01 to 0.0001
WEIGHT_DECAY = 1e-4

# SGD specific (not used with Adam)
MOMENTUM = 0.9

# ============================================================================
# LEARNING RATE SCHEDULER
# ============================================================================
USE_SCHEDULER = True
SCHEDULER_TYPE = "cosine"

# ReduceLROnPlateau params
SCHEDULER_PATIENCE = 5
SCHEDULER_FACTOR = 0.5

# StepLR params
SCHEDULER_STEP_SIZE = 10

# ============================================================================
# LOSS FUNCTION CONFIGURATION
# ============================================================================
LOSS_TYPE = "focal_dice"

# Focal Loss params
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

# Class weights
CLASS_WEIGHTS = [
    0.053914,  # 0:  background
    1.096377,  # 1:  graffiti
    0.968970,  # 2:  drainage
    0.956229,  # 3:  wetspot
    0.365729,  # 4:  weathering
    1.368079,  # 5:  crack
    1.386403,  # 6:  rockpocket
    0.823869,  # 7:  spalling
    1.204228,  # 8:  washouts/concrete corrosion
    1.273950,  # 9:  cavity
    0.789514,  # 10: efflorescence
    0.942899,  # 11: rust
    0.565548,  # 12: protective equipment
    1.479404,  # 13: exposed rebars
    0.880581,  # 14: bearing
    0.887381,  # 15: hollowareas
    1.452674,  # 16: joint tape
    1.464628,  # 17: restformwork
    1.025850,  # 18: alligator crack
    1.013773,  # 19: expansion joint
]
# ============================================================================
# TRAINING SETTINGS
# ============================================================================
SAVE_DIR = "checkpoints"
LOG_DIR = "logs"

SAVE_BEST_ONLY = True
EARLY_STOPPING_PATIENCE = 15

# ✅ CHANGED: None for fresh start
RESUME_CHECKPOINT = None # Validation
VAL_FREQUENCY = 1

# ============================================================================
# VISUALIZATION
# ============================================================================
VIS_SAMPLES = 4
VIS_FREQUENCY = 5

# ============================================================================
# RANDOM SEED
# ============================================================================
RANDOM_SEED = 42

# ============================================================================
# GPU SETTINGS
# ============================================================================
GPU_ID = 0
USE_MULTI_GPU = False

# ============================================================================
# WANDB LOGGING (optional)
# ============================================================================
USE_WANDB = False
WANDB_PROJECT = "dacl10k-segmentation"
WANDB_ENTITY = None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_config_summary():
    """Print configuration summary"""
    print("\n" + "=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"Mode:              {'CPU (TESTING)' if CPU_MODE else 'GPU (FULL TRAINING)'}")
    print(f"Model:             {MODEL_TYPE}")
    print(f"Image Size:        {IMG_SIZE}")
    print(f"Batch Size:        {BATCH_SIZE}")
    print(f"Num Epochs:        {NUM_EPOCHS}")
    print(f"Learning Rate:     {LEARNING_RATE}")
    print(f"Optimizer:         {OPTIMIZER}")
    print(f"Loss Function:     {LOSS_TYPE}")
    print(f"Use Scheduler:     {USE_SCHEDULER}")
    print(f"Use AMP:           {USE_AMP}")
    print(f"Num Classes:       {NUM_CLASSES}")
    print(f"Num Workers:       {NUM_WORKERS}")
    print(f"Save Directory:    {SAVE_DIR}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    get_config_summary()