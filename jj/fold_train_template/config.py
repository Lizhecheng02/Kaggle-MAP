# Model configuration
VER = 1
MODEL_NAME = "microsoft/phi-4"
EPOCHS = 3
MAX_LEN = 512
FOLDS = 5

# Directory settings
OUTPUT_DIR = f"map_{MODEL_NAME.replace('/', '_')}_ver_{VER}"

# Training parameters
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 16
LEARNING_RATE = 2e-4
LOGGING_STEPS = 10
SAVE_STEPS = 90
EVAL_STEPS = 90


# Data paths
TRAIN_DATA_PATH = "../outputs/train_fold.parquet"
TEST_DATA_PATH = "../../input/map-charting-student-math-misunderstandings/test.csv"
INFERENCE_DATA_PATH = "../outputs/train_fold.parquet"

# Model save paths
BEST_MODEL_PATH = f"{OUTPUT_DIR}/best"
LABEL_ENCODER_PATH = f"{OUTPUT_DIR}/label_encoder.joblib"

# Other settings
RANDOM_SEED = 42
VALIDATION_SPLIT = 0.2

LABEL_SMOOTHING_FACTOR = 0.0

TRAIN_FULL_DATA = False

# GPU settings
CUDA_VISIBLE_DEVICES = None

# Submission settings
SUBMISSION_OUTPUT_PATH = "submission.csv"

# WandB settings
USE_WANDB = True
WANDB_PROJECT = "map"
WANDB_RUN_NAME = OUTPUT_DIR
WANDB_ENTITY = None

# Early stopping settings
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_THRESHOLD = 0.001

WARM_UP = 0.0

# LoRA configurations
LORA_RANK = 64
LORA_ALPHA = 128
LORA_TARGET_MODULES = [
    "q_proj",
    "v_proj",
    "k_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
LORA_DROPOUT = 0.1
LORA_BIAS = "none"  # "none", "all", "lora_only"
USE_DORA = False

# Memory optimization settings
USE_GRADIENT_CHECKPOINTING = True  # Enable gradient checkpointing
USE_8BIT_ADAM = False  # Use 8-bit Adam optimizer for memory efficiency
MAX_GRAD_NORM = 1.0  # Gradient clipping value
