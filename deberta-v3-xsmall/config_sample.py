"""
設定ファイル - Deberta モデルのトレーニングと推論用設定
"""

# Model configuration
VER = 1
MODEL_NAME = "/root/kaggle/map-charting-student-math-misunderstandings/models/deberta-v3-xsmall"
EPOCHS = 20
MAX_LEN = 256

# Directory settings
OUTPUT_DIR = f"ver_{VER}"

# Training parameters
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
LEARNING_RATE = 5e-5
LOGGING_STEPS = 50
SAVE_STEPS = 200
EVAL_STEPS = 200

# Scheduler settings
USE_COSINE_SCHEDULER = True
WARMUP_RATIO = 0.1
NUM_CYCLES = 0.5

# Data augmentation settings
USE_DATA_AUGMENTATION = True
AUGMENTATION_CONFIG = {
    'synonym_replacement_prob': 0.1,
    'random_insertion_prob': 0.1,
    'random_swap_prob': 0.1,
    'random_deletion_prob': 0.05,
    'back_translation': False,
    'paraphrase': False
}

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
