"""
設定ファイル - Deberta モデルのトレーニングと推論用設定
"""

# Model configuration
VER = 2
MODEL_NAME = "/hdd/models/qwen-3-0.6b"
MODEL_TYPE = "qwen2"  # Add model type for proper handling
EPOCHS = 3  # Reduce epochs for initial testing
MAX_LEN = 300  # Increase max length for better context

# Directory settings
OUTPUT_DIR = f"ver_{VER}"

# Training parameters
TRAIN_BATCH_SIZE = 32  # Further reduced to avoid CUDA errors
EVAL_BATCH_SIZE = 32  # Further reduced to avoid CUDA errors
GRADIENT_ACCUMULATION_STEPS = 2  # メモリ効率向上のため
LEARNING_RATE = 2e-4
LOGGING_STEPS = 50
SAVE_STEPS = 229
EVAL_STEPS = 229


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
WANDB_PROJECT = "qwen3-0.6b-math-misconceptions"
WANDB_RUN_NAME = f"qwen3-0.6b-ver{VER}"
WANDB_ENTITY = None  # Set your WandB entity (username or team name) if needed

# Early stopping settings
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 10  # 改善が見られない評価回数の上限（評価はEVAL_STEPSごとに実行される）
EARLY_STOPPING_THRESHOLD = 0.001  # 改善とみなす最小変化量


# LoRA configuration
LORA_RANK = 64  # LoRAのランク
LORA_ALPHA = 128  # LoRAのスケーリングパラメータ
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]  # 対象モジュール
LORA_DROPOUT = 0.05  # LoRAのドロップアウト率
LORA_BIAS = "none"  # biasの扱い: "none", "all", "lora_only"

# Memory optimization settings
USE_GRADIENT_CHECKPOINTING = True  # Enable gradient checkpointing
USE_8BIT_ADAM = False  # Use 8-bit Adam optimizer for memory efficiency
MAX_GRAD_NORM = 1.0  # Gradient clipping value

# Prompt configuration
PROMPT_VERSION = "create_prompt_v2"  # Options: "create_prompt_v1", "create_prompt_v2", "create_prompt_original"

# QuestionId configuration
# Import QuestionIds from prompt_utils
from prompt_utils import questions
QUESTION_IDS = list(questions.keys())

# QuestionId-specific paths
def get_question_output_dir(question_id):
    """QuestionId用の出力ディレクトリを取得"""
    return f"{OUTPUT_DIR}/question_{question_id}"

def get_question_model_path(question_id):
    """QuestionId用のベストモデルパスを取得"""
    return f"{get_question_output_dir(question_id)}/best"

def get_question_label_encoder_path(question_id):
    """QuestionId用のラベルエンコーダパスを取得"""
    return f"{get_question_output_dir(question_id)}/label_encoder.joblib"

def get_summary_dir():
    """結果サマリー用のディレクトリを取得"""
    return f"{OUTPUT_DIR}/summary"
