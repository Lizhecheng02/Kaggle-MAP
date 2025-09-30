"""
設定ファイル - Phi-4 モデルのトレーニングと推論用設定
"""

# Model configuration
VER = 2
MODEL_NAME = "/kaggle/input/Phi-4-mini-instruct"
MODEL_TYPE = "phi"  # Phi-4-mini-instruct model type
EPOCHS = 3  # Reduce epochs for initial testing
MAX_LEN = 250  # Phi-4-mini-instruct supports up to 128K context

# Directory settings
OUTPUT_DIR = f"ver_{VER}"

# Training parameters
TRAIN_BATCH_SIZE = 8  # Phi-4-mini-instruct is more memory efficient
EVAL_BATCH_SIZE = 8  # Eval batch size
GRADIENT_ACCUMULATION_STEPS = 8  # Reduced for smaller model
LEARNING_RATE = 3e-4
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
WANDB_PROJECT = "phi-4-mini-instruct-math-misconceptions"
WANDB_RUN_NAME = f"phi-4-mini-intruct-ver{VER}"
WANDB_ENTITY = None  # Set your WandB entity (username or team name) if needed

# Early stopping settings
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 10  # 改善が見られない評価回数の上限（評価はEVAL_STEPSごとに実行される）
EARLY_STOPPING_THRESHOLD = 0.001  # 改善とみなす最小変化量

# LoRA configuration for Phi-4-mini-instruct
LORA_RANK = 32  # LoRAのランク - optimized for smaller model
LORA_ALPHA = 64  # LoRAのスケーリングパラメータ - 2:1 ratio with rank
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # Phi-4-mini-instruct uses GQA
LORA_DROPOUT = 0.1  # LoRAのドロップアウト率
LORA_BIAS = "none"  # biasの扱い: "none", "all", "lora_only"

# Memory optimization settings
USE_GRADIENT_CHECKPOINTING = True  # Enable gradient checkpointing
USE_8BIT_ADAM = False  # Use 8-bit Adam optimizer for memory efficiency
MAX_GRAD_NORM = 1.0  # Gradient clipping value
