"""
設定ファイル - Deberta モデルのトレーニングと推論用設定
"""

# Model configuration
VER = 2
MODEL_NAME = "Qwen/Qwen3-0.6B"
MODEL_TYPE = "qwen3"  # Add model type for proper handling
EPOCHS = 3  # Reduce epochs for initial testing
MAX_LEN = 300  # Increase max length for better context

openai_api_key = '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'
# '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'
# GPU configuration
CUDA_VISIBLE_DEVICES = "3"  # Use GPU 1 only

# Directory settings
OUTPUT_DIR = f"/data/llm/ver_{VER}"

# Training parameters
TRAIN_BATCH_SIZE = 32  # Reduced for large model memory efficiency
EVAL_BATCH_SIZE = 32  # Further reduced to avoid CUDA errors
GRADIENT_ACCUMULATION_STEPS = 2  # Increased for memory efficiency
LEARNING_RATE = 2e-4
WARMUP_STEPS = 100  # Warmup steps for learning rate scheduler
WEIGHT_DECAY = 0.01  # Weight decay for regularization
LOGGING_STEPS = 50
SAVE_STEPS = 200
EVAL_STEPS = 200
USE_FP16 = False  # Use FP16 training for memory efficiency


# Data paths
TRAIN_DATA_PATH ='/home/sato/project/map/train.csv'
# '/home/sato/project/map/train_32835_updated.csv'
# '/home/sato/project/map/train_final.csv'
# '/home/sato/project/map/train_changetext.csv'
# '/home/sato/project/map/train_ocr_corrected_openai.csv'
# '/home/sato/project/map/train.csv'
# '/home/sato/project/map/train_ocr_corrected_openai.csv'
# '/home/sato/project/map/train.csv'
# '/home/sato/project/map/train_ocr_corrected_openai_checkpoint_30400.csv'
# '/home/sato/project/map/train.csv'
# '/home/sato/project/map/train_ocr_corrected_openai_checkpoint_30400.csv'
# '/home/sato/project/map/train_ocr_corrected.csv'
TEST_DATA_PATH = '/home/sato/project/map/test.csv'

# Model save paths
BEST_MODEL_PATH = f"{OUTPUT_DIR}/best"
LABEL_ENCODER_PATH = f"{OUTPUT_DIR}/label_encoder.joblib" 

# Base model path for pseudo labeling (事前訓練済みモデルのパス)
BASE_MODEL_PATH = f"{OUTPUT_DIR}/best"  # train.pyで作成されたモデルのパス

# Other settings
RANDOM_SEED = 42
VALIDATION_SPLIT = 0.2


# Submission settings
SUBMISSION_OUTPUT_PATH = 'submission.csv'

# WandB settings
USE_WANDB = True  # Set to False to disable WandB
WANDB_PROJECT = "qwen3-0.6b-ocr-correct"
WANDB_RUN_NAME = "qwen3-0.6b-origin-real_choice"
# "qwen3-0.6b-4o-mini"
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

# 3段階学習用の設定
STAGE1_EPOCHS = 3  # 第1段階のエポック数
STAGE3_EPOCHS = 1  # 第3段階のエポック数
STAGE3_LEARNING_RATE_RATIO = 1.0  # 第3段階の学習率比率（元の学習率に対する倍率）
SKIP_STAGE1_TRAINING = False  # 第1段階をスキップするかどうか
STAGE1_PRETRAINED_MODEL_PATH = "./stage1_final"  # 第1段階をスキップする場合の学習済み重みパス
