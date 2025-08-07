"""
設定ファイル - Deberta モデルのトレーニングと推論用設定
"""

# Model configuration
VER = 2
MODEL_NAME = "/kaggle/input/qwen2.5-VL-7b-instruct"
MODEL_TYPE = "qwen2_vl"  # Add model type for proper handling
EPOCHS = 3  # Reduce epochs for initial testing
MAX_LEN = 300  # Increase max length for better context

# Directory settings
OUTPUT_DIR = f"ver_{VER}"

# Training parameters
TRAIN_BATCH_SIZE = 2  # Reduced batch size for memory efficiency
EVAL_BATCH_SIZE = 4  # Reduced batch size for memory efficiency
GRADIENT_ACCUMULATION_STEPS = 16  # Increased to maintain effective batch size of 8
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
WANDB_PROJECT = "qwen2.5-VL-7b-math-misconceptions"
WANDB_RUN_NAME = f"qwen2.5-VL-7b-ver{VER}"
WANDB_ENTITY = None  # Set your WandB entity (username or team name) if needed

# Early stopping settings
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 10  # 改善が見られない評価回数の上限（評価はEVAL_STEPSごとに実行される）
EARLY_STOPPING_THRESHOLD = 0.001  # 改善とみなす最小変化量

# LoRA settings
USE_LORA = True  # LoRAを使用するかどうか（Falseの場合はフルファインチューニング）
LORA_R = 64  # LoRAのランク（小さいほどパラメータ数が少ない）
LORA_ALPHA = 128  # LoRAのスケーリングパラメータ（通常はrの2倍程度が推奨）
LORA_DROPOUT = 0.1  # LoRAのドロップアウト率
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]  # 対象モジュール（Qwenモデルのアテンション層）
LORA_BIAS = "none"  # bias設定: "none", "all", "lora_only"
LORA_MODULES_TO_SAVE = ["classifier"]  # 追加で保存するモジュール（分類層など）
