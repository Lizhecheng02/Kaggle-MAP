"""
設定ファイル - Deberta モデルのトレーニングと推論用設定
"""

# Model configuration
VER = 1
MODEL_NAME = "/root/kaggle/map-charting-student-math-misunderstandings/models/deberta-v3-large"
EPOCHS = 3
MAX_LEN = 256

# Directory settings
OUTPUT_DIR = f"ver_{VER}"

# Training parameters
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
GRADIENT_ACCUMULATION_STEPS = 2  # Increase this when reducing batch size to simulate larger batches
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

# Submission settings
SUBMISSION_OUTPUT_PATH = 'submission.csv'

# WANDB settings
WANDB_PROJECT = 'deberta_large-math-misconceptions'
WANDB_ENTITY = None  # Set your WANDB username/team if needed
WANDB_RUN_NAME = f'ver{VER}'

# Scheduler settings
WARMUP_STEPS = 500
SCHEDULER_TYPE = 'cosine'

# Training settings
REPORT_TO = "wandb"
BF16 = True  # TRAIN WITH BF16 IF LOCAL GPU IS NEWER GPU
FP16 = False  # INFER WITH FP16 BECAUSE KAGGLE IS T4 GPU
REFERENCE_COMPILE = False
