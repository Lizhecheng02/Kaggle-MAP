"""
設定ファイル - Phi-4 モデルのトレーニングと推論用設定
"""

# Model configuration
VER = 2
MODEL_NAME = "/kaggle/input/phi-4"
MODEL_TYPE = "phi"  # Phi-4 model type
EPOCHS = 3  # Reduce epochs for initial testing
MAX_LEN = 300  # Phi-4 supports longer context

# Directory settings
OUTPUT_DIR = f"ver_{VER}"

# Training parameters
TRAIN_BATCH_SIZE = 8  # Smaller batch size for Phi-4
EVAL_BATCH_SIZE = 8  # Eval batch size
GRADIENT_ACCUMULATION_STEPS = 8  # Increased for effective batch size
LEARNING_RATE = 2e-4
LOGGING_STEPS = 50
SAVE_STEPS = 229
EVAL_STEPS = 229


# Data paths
TRAIN_DATA_PATH = '/kaggle/input/map-charting-student-math-misunderstandings/train.csv'
# TRAIN_DATA_PATH = "/kaggle/input/map-charting-student-math-misunderstandings/train_ocr_corrected_openai.csv"
TEST_DATA_PATH = '/kaggle/input/map-charting-student-math-misunderstandings/test.csv'

# Model save paths
BEST_MODEL_PATH = f"{OUTPUT_DIR}/best"
LABEL_ENCODER_PATH = f"{OUTPUT_DIR}/label_encoder.joblib"

# Other settings
RANDOM_SEED = 42
VALIDATION_SPLIT = 0.000000000000001

# GPU settings
CUDA_VISIBLE_DEVICES = "1"  # GPU device to use. Set to None to use all available GPUs

# Submission settings
SUBMISSION_OUTPUT_PATH = 'submission.csv'

# WandB settings
USE_WANDB = True  # Set to False to disable WandB
WANDB_PROJECT = "phi-4-math-misconceptions"
WANDB_RUN_NAME = f"phi-4-ver{VER}"
WANDB_ENTITY = None  # Set your WandB entity (username or team name) if needed

# Early stopping settings
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 10  # 改善が見られない評価回数の上限（評価はEVAL_STEPSごとに実行される）
EARLY_STOPPING_THRESHOLD = 0.001  # 改善とみなす最小変化量

# LoRA configuration for Phi-4
LORA_RANK = 64  # LoRAのランク - optimized for Phi-4
LORA_ALPHA = 128  # LoRAのスケーリングパラメータ - 1:1 ratio with rank
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # Phi-4 target modules
LORA_DROPOUT = 0.1  # LoRAのドロップアウト率 - reduced for Phi-4
LORA_BIAS = "none"  # biasの扱い: "none", "all", "lora_only"
USE_DORA = False  # DoRA (Weight-Decomposed Low-Rank Adaptation) を使用する場合はTrue

# Memory optimization settings
USE_GRADIENT_CHECKPOINTING = True  # Enable gradient checkpointing
USE_8BIT_ADAM = False  # Use 8-bit Adam optimizer for memory efficiency
MAX_GRAD_NORM = 1.0  # Gradient clipping value

# Attention implementation settings
# "eager": 標準のPyTorch実装
# "flash_attention_2": Flash Attention 2実装（高速・省メモリ）
ATTENTION_IMPLEMENTATION = "eager"  # Options: "eager", "flash_attention_2"
