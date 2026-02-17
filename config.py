"""
Training Configuration for DACL10K U-Net
Modify these parameters based on your hardware
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
 "expansion joint"
]

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Choose your training mode
# Option 1: CPU mode (local testing only, very slow)
# Option 2: GPU mode (for Google Colab or if you have GPU)
CPU_MODE = True  # Set to False if using GPU

# Model configuration
if CPU_MODE:
    MODEL_TYPE = 'unet_lite'  # Lightweight model for CPU
    IMG_SIZE = (256, 256)     # Smaller images
    BATCH_SIZE = 2            # Small batch size
    NUM_WORKERS = 0           # Fewer workers
    NUM_EPOCHS = 5            # Just for testing
    PRINT_FREQ = 50           # Print every 50 batches
else:
    MODEL_TYPE = 'unet'       # Standard U-Net for GPU
    IMG_SIZE = (512, 512)     # Full resolution
    BATCH_SIZE = 8            # Larger batch size
    NUM_WORKERS = 0           # More workers
    NUM_EPOCHS = 50           # Full training
    PRINT_FREQ = 20           # Print every 20 batches

# ============================================================================
# OPTIMIZER CONFIGURATION
# ============================================================================
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
OPTIMIZER = 'adam'  # 'adam' or 'sgd'

# SGD specific (if using SGD)
MOMENTUM = 0.9

# Learning rate scheduler
USE_SCHEDULER = True
SCHEDULER_TYPE = 'cosine'  # 'cosine', 'step', or 'plateau'
SCHEDULER_PATIENCE = 5     # For ReduceLROnPlateau
SCHEDULER_FACTOR = 0.5     # LR reduction factor
SCHEDULER_STEP_SIZE = 10   # For StepLR

# ============================================================================
# LOSS FUNCTION CONFIGURATION
# ============================================================================
LOSS_TYPE = 'ce'  # 'ce' (CrossEntropy), 'focal', or 'dice'

# Focal Loss parameters (if using focal loss)
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

# Class weights (optional, for imbalanced classes)
# Set to None to not use class weights
# Or provide list of 20 weights (one per class)
CLASS_WEIGHTS = None  # Will be computed from dataset if None

# ============================================================================
# TRAINING SETTINGS
# ============================================================================
SAVE_DIR = "checkpoints"
LOG_DIR = "logs"
SAVE_BEST_ONLY = True
EARLY_STOPPING_PATIENCE = 15  # Stop if no improvement for N epochs

# Resume training
RESUME_CHECKPOINT = None  # Path to checkpoint to resume from, or None

# Validation
VAL_FREQUENCY = 1  # Validate every N epochs

# Mixed precision training (only for GPU)
USE_AMP = False if CPU_MODE else True  # Automatic Mixed Precision

# ============================================================================
# VISUALIZATION
# ============================================================================
VIS_SAMPLES = 4  # Number of samples to visualize per validation
VIS_FREQUENCY = 5  # Visualize every N epochs

# ============================================================================
# RANDOM SEED
# ============================================================================
RANDOM_SEED = 42

# ============================================================================
# GPU SETTINGS (if available)
# ============================================================================
GPU_ID = 0  # GPU device ID
USE_MULTI_GPU = False  # Use DataParallel for multiple GPUs

# ============================================================================
# WANDB LOGGING (optional)
# ============================================================================
USE_WANDB = False  # Set to True to use Weights & Biases logging
WANDB_PROJECT = "dacl10k-segmentation"
WANDB_ENTITY = None  # Your wandb username, or None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_config_summary():
    """Print configuration summary"""
    print("\n" + "="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
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
    print("="*70 + "\n")


if __name__ == "__main__":
    get_config_summary()