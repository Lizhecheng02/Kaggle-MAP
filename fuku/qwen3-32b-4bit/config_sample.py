"""
設定ファイル - Deberta モデルのトレーニングと推論用設定
"""

# Model configuration
VER = "4bit"
MODEL_NAME = "/hdd/models/Qwen3-32B"
MODEL_TYPE = "qwen2"  # Add model type for proper handling
EPOCHS = 3  # Reduce epochs for initial testing
MAX_LEN = 300  # Increase max length for better context

# Directory settings
OUTPUT_DIR = f"ver_{VER}"

# Training parameters
TRAIN_BATCH_SIZE = 8  # Batch size 2 for RTX 5090 with 31GB VRAM
EVAL_BATCH_SIZE = 8  # Eval can use larger batch size
GRADIENT_ACCUMULATION_STEPS = 8  # Reduced to 32 for faster training
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
WANDB_PROJECT = "qwen3-32b-math-misconceptions"
WANDB_RUN_NAME = f"qwen3-32b-ver{VER}"
WANDB_ENTITY = None  # Set your WandB entity (username or team name) if needed

# Early stopping settings
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 10  # 改善が見られない評価回数の上限（評価はEVAL_STEPSごとに実行される）
EARLY_STOPPING_THRESHOLD = 0.001  # 改善とみなす最小変化量

# LoRA configuration
LORA_RANK = 64  # LoRAのランク - reduced for memory efficiency
LORA_ALPHA = 128  # LoRAのスケーリングパラメータ - reduced proportionally
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]  # 対象モジュール
LORA_DROPOUT = 0.1  # LoRAのドロップアウト率
LORA_BIAS = "none"  # biasの扱い: "none", "all", "lora_only"

# Memory optimization settings
USE_GRADIENT_CHECKPOINTING = True  # Enable gradient checkpointing
USE_8BIT_ADAM = False  # Use 8-bit Adam optimizer for memory efficiency
MAX_GRAD_NORM = 1.0  # Gradient clipping value

# 4-bit quantization settings
USE_4BIT_QUANTIZATION = True  # Enable 4-bit quantization
BNB_4BIT_COMPUTE_DTYPE = "bfloat16"  # Compute dtype for 4-bit base models
BNB_4BIT_QUANT_TYPE = "nf4"  # Quantization type (fp4 or nf4)
BNB_4BIT_USE_DOUBLE_QUANT = True  # Use double quantization
BNB_4BIT_QUANT_STORAGE_DTYPE = "uint8"  # Storage dtype for quantized weights
