"""
設定ファイル - Deberta モデルのトレーニングと推論用設定
"""

# Model configuration
VER = 2
MODEL_NAME = "/kaggle/input/qwen-3-1.7b"
MODEL_TYPE = "qwen2"  # Add model type for proper handling
EPOCHS = 5  # Reduce epochs for initial testing
MAX_LEN = 512  # Increase max length for better context

# Directory settings
OUTPUT_DIR = f"ver_{VER}"

# Training parameters
TRAIN_BATCH_SIZE = 16  # Further reduced to avoid CUDA errors
EVAL_BATCH_SIZE = 32  # Further reduced to avoid CUDA errors
GRADIENT_ACCUMULATION_STEPS = 4  # メモリ効率向上のため
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

# GPU settings
CUDA_VISIBLE_DEVICES = "0"  # GPU device to use. Set to None to use all available GPUs

# Submission settings
SUBMISSION_OUTPUT_PATH = 'submission.csv'

# WandB settings
USE_WANDB = True  # Set to False to disable WandB
WANDB_PROJECT = "qwen3-1.7b-math-misconceptions"
WANDB_RUN_NAME = f"qwen3-1.7b-ver{VER}"
WANDB_ENTITY = None  # Set your WandB entity (username or team name) if needed

# Early stopping settings
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 10  # 改善が見られない評価回数の上限（評価はEVAL_STEPSごとに実行される）
EARLY_STOPPING_THRESHOLD = 0.001  # 改善とみなす最小変化量

# LoRA configuration
LORA_RANK = 16  # LoRAのランク
LORA_ALPHA = 32  # LoRAのスケーリングパラメータ
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]  # 対象モジュール
LORA_DROPOUT = 0.1  # LoRAのドロップアウト率
LORA_BIAS = "none"  # biasの扱い: "none", "all", "lora_only"
