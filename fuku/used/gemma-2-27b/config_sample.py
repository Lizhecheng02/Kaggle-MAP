"""
設定ファイル - Gemma-2-9b モデルのトレーニングと推論用設定
"""

# Model configuration
VER = 2
MODEL_NAME = "/kaggle/input/gemma-2-27b-it"
MODEL_TYPE = "gemma2"  # Add model type for proper handling
EPOCHS = 3  # Reduce epochs for initial testing
MAX_LEN = 512  # Increase max length for better context

# Directory settings
OUTPUT_DIR = f"ver_{VER}"

# Training parameters
TRAIN_BATCH_SIZE = 4  # Further reduced to avoid CUDA errors
EVAL_BATCH_SIZE = 4  # Further reduced to avoid CUDA errors
GRADIENT_ACCUMULATION_STEPS=16
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

# Quantization (bitsandbytes) settings
# 4bit量子化を有効/無効化
USE_4BIT = True
# 量子化方式: "nf4" または "fp4"
BNB_4BIT_QUANT_TYPE = "nf4"
# ダブル量子化を使用するか
BNB_4BIT_USE_DOUBLE_QUANT = True
# 計算dtypeの指定: "bf16" | "fp16" | "fp32" | None(自動)
BNB_COMPUTE_DTYPE = "bf16"

# GPU settings
CUDA_VISIBLE_DEVICES = "0"  # GPU device to use. Set to None to use all available GPUs

# Submission settings
SUBMISSION_OUTPUT_PATH = 'submission.csv'

# WandB settings
USE_WANDB = True  # Set to False to disable WandB
WANDB_PROJECT = "gemma-2-27b-it-math-misconceptions"
WANDB_RUN_NAME = f"gemma-2-27b-it-ver{VER}"
WANDB_ENTITY = None  # Set your WandB entity (username or team name) if needed

# Early stopping settings
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 10  # Number of evaluations with no improvement after which training will be stopped
EARLY_STOPPING_THRESHOLD = 0.001  # Minimum change in the monitored metric to qualify as an improvement

# LoRA configuration
LORA_R = 16  # LoRAのランク
LORA_ALPHA = 32  # LoRAのスケーリングパラメータ
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # Gemma用の対象モジュール
LORA_DROPOUT = 0.1  # LoRAのドロップアウト率
LORA_BIAS = "none"  # バイアスの設定
