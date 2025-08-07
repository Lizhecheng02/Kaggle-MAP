"""
設定ファイル - ELECTRA-large モデルのトレーニングと推論用設定
"""

# Model configuration
VER = 1
MODEL_NAME = "google/electra-large-discriminator"
EPOCHS = 25  # ELECTRAは収束が速いため少し増やす
MAX_LEN = 320  # ELECTRAは効率的なため中間的な長さ

# Directory settings
OUTPUT_DIR = f"electra_ver_{VER}"

# Training parameters - ELECTRA用に最適化
TRAIN_BATCH_SIZE = 24  # ModernBERTとDeBERTaの中間
EVAL_BATCH_SIZE = 48
LEARNING_RATE = 1.5e-5  # ELECTRAには中程度の学習率
LOGGING_STEPS = 50
SAVE_STEPS = 200
EVAL_STEPS = 200

# Data paths
TRAIN_DATA_PATH = '/kaggle/input/map-charting-student-math-misunderstandings/train.csv'
TEST_DATA_PATH = '/kaggle/input/map-charting-student-math-misunderstandings/test.csv'

# Model save paths
BEST_MODEL_PATH = f"{OUTPUT_DIR}/best"
LABEL_ENCODER_PATH = f"{OUTPUT_DIR}/label_encoder.joblib"
BEST_MAP3_MODEL_PATH = f"{OUTPUT_DIR}/best_map3"

# Other settings
RANDOM_SEED = 42
VALIDATION_SPLIT = 0.2

# Submission settings
SUBMISSION_OUTPUT_PATH = 'submission_electra.csv'

# Wandb settings
USE_WANDB = True
WANDB_PROJECT = "electra-math-misconceptions"
WANDB_RUN_NAME = f"electra-large-v{VER}"
WANDB_ENTITY = None  # Set your wandb entity if needed

# Learning rate scheduler settings
USE_COSINE_SCHEDULER = True
WARMUP_STEPS = 600  # ELECTRAは少し長めのウォームアップが有効

# ELECTRA specific settings
GRADIENT_CHECKPOINTING = True  # メモリ効率化
WEIGHT_DECAY = 0.01  # 正則化
ADAM_EPSILON = 1e-6