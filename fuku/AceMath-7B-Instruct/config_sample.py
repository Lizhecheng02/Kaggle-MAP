"""
設定ファイル - AceMath-7B-Instruct 用設定
"""

"""モデル設定"""
VER = 2
MODEL_NAME = "/hdd/models/AceMath-7B-Instruct"
MODEL_TYPE = "ace-math"  # モデルの種類（情報用途）
EPOCHS = 3
MAX_LEN = 250

"""出力ディレクトリ"""
OUTPUT_DIR = f"ver_{VER}"

"""学習パラメータ"""
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-4
LOGGING_STEPS = 50
SAVE_STEPS = 229
EVAL_STEPS = 229


"""データパス"""
TRAIN_DATA_PATH = '/kaggle/input/map-charting-student-math-misunderstandings/train.csv'
TEST_DATA_PATH = '/kaggle/input/map-charting-student-math-misunderstandings/test.csv'

"""保存パス"""
BEST_MODEL_PATH = f"{OUTPUT_DIR}/best"
LABEL_ENCODER_PATH = f"{OUTPUT_DIR}/label_encoder.joblib"

"""その他"""
RANDOM_SEED = 42
VALIDATION_SPLIT = 0.2

"""GPU設定"""
CUDA_VISIBLE_DEVICES = "0"  # 使用GPU（Noneで全GPU）

"""提出ファイル設定"""
SUBMISSION_OUTPUT_PATH = 'submission.csv'

"""WandB設定"""
USE_WANDB = True
WANDB_PROJECT = "ace-math-7b-instruct"
WANDB_RUN_NAME = f"ace-math-7b-instruct-ver{VER}"
WANDB_ENTITY = None  # Set your WandB entity (username or team name) if needed

"""Early Stopping 設定"""
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_THRESHOLD = 0.001

"""LoRA 設定（AceMath想定の一般的構成）"""
LORA_RANK = 64
LORA_ALPHA = 128
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LORA_DROPOUT = 0.1
LORA_BIAS = "none"
USE_DORA = False

"""メモリ最適化"""
USE_GRADIENT_CHECKPOINTING = True
USE_8BIT_ADAM = False
MAX_GRAD_NORM = 1.0

"""アテンション実装設定"""
# "eager": 標準PyTorch, "flash_attention_2": 高速・省メモリ
ATTENTION_IMPLEMENTATION = "eager"

"""パディング設定（モデル依存の特殊トークンを避ける）"""
# True の場合、pad_token は eos_token を流用（多くのLLaMA系で推奨）
USE_EOS_AS_PAD = True
# モデル付属のpad_tokenを使いたい場合は明示指定（通常はNoneのまま）
PAD_TOKEN_STR = None
PAD_TOKEN_ID = None

# プロンプト設定は utils.py に集約（本ファイルでは定義しません）
