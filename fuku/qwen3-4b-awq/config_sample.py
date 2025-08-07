"""
設定ファイル - Deberta モデルのトレーニングと推論用設定
"""

# Model configuration
VER = 2
MODEL_NAME = "/kaggle/input/qwen-3-4b-awq"
MODEL_TYPE = "qwen2"  # Add model type for proper handling
EPOCHS = 10  # Reduce epochs for initial testing
MAX_LEN = 512  # Increase max length for better context

# Directory settings
OUTPUT_DIR = f"ver_{VER}"

# Training parameters
TRAIN_BATCH_SIZE = 128  # バッチサイズを128に変更
EVAL_BATCH_SIZE = 16   # 検証用バッチサイズをメモリ節約のため16に設定
LEARNING_RATE = 1e-4    # 線形スケーリングルールに従って学習率を調整
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

# ----------------------------------------------------------------------------
# wandb設定
# ----------------------------------------------------------------------------
USE_WANDB = True  # W&Bで実験をログ管理するかどうかを切り替え
WANDB_PROJECT = 'qwen3-4b-awq'  # W&Bプロジェクト名
WANDB_ENTITY = None  # W&Bエンティティ名（ユーザーまたはチーム）
WANDB_RUN_NAME = f"{MODEL_TYPE}_ver{VER}"  # 実験ランの名前
WANDB_LOG_MODEL = True  # トレーニング後にモデルチェックポイントをW&Bにログ
