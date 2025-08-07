"""
設定ファイル - MathBERT モデルのトレーニングと推論用設定
"""

# Model configuration
VER = 1
MODEL_NAME = "/kaggle/input/MathBERT"
EPOCHS = 5
MAX_LEN = 256

# Directory settings
OUTPUT_DIR = f"ver_{VER}"

# Training parameters
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
LEARNING_RATE = 2e-4
LOGGING_STEPS = 50
SAVE_STEPS = 200
EVAL_STEPS = 200

# Data paths
TRAIN_DATA_PATH = '/kaggle/input/map-charting-student-math-misunderstandings/train.csv'
TEST_DATA_PATH = '/kaggle/input/map-charting-student-math-misunderstandings/test.csv'

# Model save paths
BEST_MODEL_PATH = f"{OUTPUT_DIR}/best"
LABEL_ENCODER_PATH = f"{OUTPUT_DIR}/label_encoder.joblib"

# Other settings
RANDOM_SEED = 42
VALIDATION_SPLIT = 0.2

# Submission output path
SUBMISSION_OUTPUT_PATH = f"{OUTPUT_DIR}/submission.csv"

# WandB configuration
WANDB_PROJECT = "mathbert-misconception"
WANDB_RUN_NAME = f"mathbert_v{VER}"
USE_WANDB = True

# Learning rate scheduler configuration
LR_SCHEDULER_TYPE = "cosine"
WARMUP_RATIO = 0.0
WARMUP_STEPS = 500
