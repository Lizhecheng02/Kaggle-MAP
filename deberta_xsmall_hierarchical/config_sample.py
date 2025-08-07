"""
階層的分類用の設定ファイル
"""

# Model configuration
VER = 2
MODEL_NAME = "/root/kaggle/map-charting-student-math-misunderstandings/models/deberta-v3-xsmall"
EPOCHS_CATEGORY = 20  # Categoryモデル用
EPOCHS_MISCONCEPTION = 30  # Misconceptionモデル用
MAX_LEN = 256

# Directory settings
OUTPUT_DIR = f"ver_hierarchical_{VER}"

# Training parameters
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
LEARNING_RATE_CATEGORY = 3e-5  # Categoryモデル用（少し低め）
LEARNING_RATE_MISCONCEPTION = 5e-5  # Misconceptionモデル用
LOGGING_STEPS = 50
SAVE_STEPS = 200
EVAL_STEPS = 200

# Data paths
TRAIN_DATA_PATH = '/kaggle/input/map-charting-student-math-misunderstandings/train.csv'
TEST_DATA_PATH = '/kaggle/input/map-charting-student-math-misunderstandings/test.csv'

# Model save paths
CATEGORY_MODEL_PATH = f"{OUTPUT_DIR}/category_model"
TRUE_MISCONCEPTION_MODEL_PATH = f"{OUTPUT_DIR}/true_misconception_model"
FALSE_MISCONCEPTION_MODEL_PATH = f"{OUTPUT_DIR}/false_misconception_model"

# Label encoder paths
CATEGORY_ENCODER_PATH = f"{OUTPUT_DIR}/category_encoder.joblib"
TRUE_MISCONCEPTION_ENCODER_PATH = f"{OUTPUT_DIR}/true_misconception_encoder.joblib"
FALSE_MISCONCEPTION_ENCODER_PATH = f"{OUTPUT_DIR}/false_misconception_encoder.joblib"

# Other settings
RANDOM_SEED = 42
VALIDATION_SPLIT = 0.2

# Category labels
CATEGORIES = [
    'True_Correct',
    'True_Neither',
    'True_Misconception',
    'False_Correct',
    'False_Neither',
    'False_Misconception'
]

# Misconceptionを持つカテゴリ
MISCONCEPTION_CATEGORIES = ['True_Misconception', 'False_Misconception']
