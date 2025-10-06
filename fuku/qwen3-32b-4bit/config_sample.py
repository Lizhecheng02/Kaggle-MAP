"""
設定ファイル - Deberta モデルのトレーニングと推論用設定
"""

# Model configuration
VER = "4bit"
MODEL_NAME = "/kaggle/input/models/Qwen3-32B"
MODEL_TYPE = "qwen2"  # Add model type for proper handling
EPOCHS = 3  # Reduce epochs for initial testing
MAX_LEN = 300  # Increase max length for better context

# Directory settings
OUTPUT_DIR = f"ver_{VER}"

# Training parameters
TRAIN_BATCH_SIZE = 4  # Batch size 2 for RTX 5090 with 31GB VRAM
EVAL_BATCH_SIZE = 4  # Eval can use larger batch size
GRADIENT_ACCUMULATION_STEPS = 16  # Reduced to 32 for faster training
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

# 4-bit quantization settings
USE_4BIT_QUANTIZATION = True  # Enable 4-bit quantization
BNB_4BIT_COMPUTE_DTYPE = "bfloat16"  # Compute dtype for 4-bit base models
BNB_4BIT_QUANT_TYPE = "nf4"  # Quantization type (fp4 or nf4)
BNB_4BIT_USE_DOUBLE_QUANT = True  # Use double quantization
BNB_4BIT_QUANT_STORAGE_DTYPE = "uint8"  # Storage dtype for quantized weights

# Target label generation settings
TARGET_QUESTION_ID_COL = "QuestionId"
TARGET_BASECATEGORY_COL = "BaseCategory"
TARGET_MISCONCEPTION_COL = "Misconception"
TARGET_NEITHER_KEYWORD = "Neither"
TARGET_NEITHER_PREFIX = "Q"
TARGET_MISCONCEPTION_FILLNA_VALUE = "NA"
TARGET_BASECATEGORY_FALLBACK_COLS = ["Category"]

# Submission generation settings
SUBMISSION_USE_QUESTION_LABEL_CHOICES = True
# 許容ラベル集合（QuestionIdごとの制約）。LabelEncoder.classes_ と一致する表記で指定すること。
QUESTION_LABEL_CHOICES = {
    '31772': [
        '31772:Correct:NA',
        'Q:Neither:NA',
        '31772:Misconception:Incomplete',
        '31772:Misconception:WNB',
    ],
    '31774': [
        '31774:Correct:NA',
        'Q:Neither:NA',
        '31774:Misconception:SwapDividend',
        '31774:Misconception:Mult',
        '31774:Misconception:FlipChange',
    ],
    '31777': [
        '31777:Correct:NA',
        'Q:Neither:NA',
        '31777:Misconception:Incomplete',
        '31777:Misconception:Irrelevant',
        '31777:Misconception:Wrong_Fraction',
    ],
    '31778': [
        '31778:Correct:NA',
        'Q:Neither:NA',
        '31778:Misconception:Additive',
        '31778:Misconception:Irrelevant',
        '31778:Misconception:WNB',
    ],
    '32829': [
        '32829:Correct:NA',
        'Q:Neither:NA',
        '32829:Misconception:Not_variable',
        '32829:Misconception:Adding_terms',
        '32829:Misconception:Inverse_operation'
    ],
    '32833': [
        '32833:Correct:NA',
        'Q:Neither:NA',
        '32833:Misconception:Inversion',
        '32833:Misconception:Duplication',
        '32833:Misconception:Wrong_Operation'
    ],
    '32835': [
        '32835:Correct:NA',
        'Q:Neither:NA',
        '32835:Misconception:Whole_numbers_larger',
        '32835:Misconception:Longer_is_bigger',
        '32835:Misconception:Ignores_zeroes',
        '32835:Misconception:Shorter_is_bigger',
    ],
    '33471': [
        '33471:Correct:NA',
        'Q:Neither:NA',
        '33471:Misconception:Wrong_fraction',
        '33471:Misconception:Incomplete',
    ],
    '33472': [
        '33472:Correct:NA',
        'Q:Neither:NA',
        '33472:Misconception:Adding_across',
        '33472:Misconception:Denominator-only_change',
        '33472:Misconception:Incorrect_equivalent_fraction_addition',
    ],
    '33474': [
        '33474:Correct:NA',
        'Q:Neither:NA',
        '33474:Misconception:Division',
        '33474:Misconception:Subtraction',
    ],
    '76870': [
        '76870:Correct:NA',
        'Q:Neither:NA',
        '76870:Misconception:Unknowable',
        '76870:Misconception:Definition',
        '76870:Misconception:Interior',
    ],
    '89443': [
        '89443:Correct:NA',
        'Q:Neither:NA',
        '89443:Misconception:Positive',
        '89443:Misconception:Tacking',
    ],
    '91695': [
        '91695:Correct:NA',
        'Q:Neither:NA',
        '91695:Misconception:Wrong_term',
        '91695:Misconception:Firstterm',
    ],
    '104665': [
        '104665:Correct:NA',
        'Q:Neither:NA',
        '104665:Misconception:Base_rate',
        '104665:Misconception:Multiplying_by_4',
    ],
    '109465': [
        '109465:Correct:NA',
        'Q:Neither:NA',
        '109465:Misconception:Certainty',
        '109465:Misconception:Scale',
    ]
}
