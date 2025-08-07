"""
設定ファイル - ModernBERT モデルのトレーニングと推論用設定
"""

# Model configuration
VER = 3  # バージョンアップ（Multi-sample dropout版）
MODEL_NAME = "/kaggle/input/ModernBERT-large"
EPOCHS = 5
MAX_LEN = 256  # プロンプトが長くなったため増加

# Directory settings
OUTPUT_DIR = f"ver_{VER}"

# Training parameters
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
LEARNING_RATE = 2e-5
LOGGING_STEPS = 50
SAVE_STEPS = 200
EVAL_STEPS = 200
GRADIENT_ACCUMULATION_STEPS = 1  # グラディエント累積ステップ数

# Data paths
TRAIN_DATA_PATH = '/kaggle/input/map-charting-student-math-misunderstandings/train.csv'
TEST_DATA_PATH = '/kaggle/input/map-charting-student-math-misunderstandings/test.csv'

# Model save paths
BEST_MODEL_PATH = f"{OUTPUT_DIR}/best"
BEST_MAP3_MODEL_PATH = f"{OUTPUT_DIR}/best_map3"
LABEL_ENCODER_PATH = f"{OUTPUT_DIR}/label_encoder.joblib"

# Other settings
RANDOM_SEED = 42
VALIDATION_SPLIT = 0.2

# Submission settings
SUBMISSION_OUTPUT_PATH = 'submission.csv'

# Wandb settings
USE_WANDB = True
WANDB_PROJECT = "modernbert-math-misconceptions"
WANDB_RUN_NAME = f"modernbert-large-v{VER}"
WANDB_ENTITY = None  # Set your wandb entity if needed

# Learning rate scheduler settings
USE_COSINE_SCHEDULER = True
WARMUP_STEPS = 500

# Multi-sample dropout settings
USE_MULTI_DROPOUT = True
NUM_DROPOUT_HEADS = 5
DROPOUT_P = 0.5
