"""
設定ファイル - Deberta モデルのトレーニングと推論用設定
"""

# Model configuration
VER = "awq"
MODEL_NAME = "/hdd/models/Qwen3-32B-AWQ"
MODEL_TYPE = "qwen3"  # Qwen3 model type for AWQ
EPOCHS = 3  # Reduce epochs for initial testing
MAX_LEN = 300  # Increase max length for better context

# Directory settings
OUTPUT_DIR = f"ver_{VER}"

# Training parameters
TRAIN_BATCH_SIZE = 2  # AWQモデルはより大きなバッチサイズが可能
EVAL_BATCH_SIZE = 4   # 推論時のバッチサイズ
GRADIENT_ACCUMULATION_STEPS = 16  # AWQで効率的な勾配蓄積
LEARNING_RATE = 2e-4
LOGGING_STEPS = 50
SAVE_STEPS = 229
EVAL_STEPS = 229


# Data paths
TRAIN_DATA_PATH = '/kaggle/input/map-charting-student-math-misunderstandings/train.csv'
TEST_DATA_PATH = '/kaggle/input/map-charting-student-math-misunderstandings/test.csv'

# Model save paths
BEST_MODEL_PATH = f"{OUTPUT_DIR}/best_map3"
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

# AWQ quantization settings
USE_AWQ_QUANTIZATION = True  # Enable AWQ quantization (pre-quantized model)
AWQ_FUSE_LAYERS = True  # Enable layer fusion for better performance
AWQ_FUSE_MAX_SEQ_LEN = 4096  # Maximum sequence length for fused layers
