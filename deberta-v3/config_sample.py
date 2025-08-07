"""
設定ファイル - Deberta モデルのトレーニングと推論用設定
"""

# Model configuration
VER = 2
MODEL_NAME = "/root/kaggle/map-charting-student-math-misunderstandings/models/deberta-v3-base"  # より大きなモデル
EPOCHS = 30  # エポック数を増加
MAX_LEN = 384  # 最大長を増加（トークン分析に基づく）

# Directory settings
OUTPUT_DIR = f"ver_{VER}"

# Training parameters (ハイパーパラメータの最適化)
TRAIN_BATCH_SIZE = 64  # バッチサイズを128に設定
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 0.8e-4  # バッチサイズに合わせて学習率を調整
WARMUP_RATIO = 0.1  # ウォームアップ期間を追加
WEIGHT_DECAY = 0.01  # 正則化を追加

# スケジューラー設定
SCHEDULER_TYPE = "cosine"  # コサインスケジューラーを使用
NUM_CYCLES = 0.5

# ロギングとチェックポイント設定
LOGGING_STEPS = 50
SAVE_STEPS = 500
EVAL_STEPS = 500
GRADIENT_ACCUMULATION_STEPS = 2  # 実効バッチサイズを増加

# データ拡張設定
USE_DATA_AUGMENTATION = True
MIN_SAMPLES_PER_CLASS = 500  # クラスごとの最小サンプル数
MIXUP_ALPHA = 0.2  # Mixupの強度

# マルチタスク学習設定
USE_MULTITASK = True
MAIN_TASK_WEIGHT = 0.7
AUX_TASK_WEIGHT = 0.3

# Focal Loss設定（クラス不均衡対策）
USE_FOCAL_LOSS = True
FOCAL_GAMMA = 2.0

# アンサンブル設定
ENSEMBLE_MODELS = 1  # アンサンブルなし（ユーザーの要望に従って）
ENSEMBLE_SEEDS = [42]  # 単一モデルのシード

# ストラテジー設定
USE_STRATIFIED_SPLIT = True  # 層化分割を使用
USE_CLASS_WEIGHTS = True  # クラス重みを使用
LABEL_SMOOTHING = 0.1  # ラベルスムージング

# Data paths
TRAIN_DATA_PATH = '/kaggle/input/map-charting-student-math-misunderstandings/train.csv'
TEST_DATA_PATH = '/kaggle/input/map-charting-student-math-misunderstandings/test.csv'

# Model save paths
BEST_MODEL_PATH = f"{OUTPUT_DIR}/best"
LABEL_ENCODER_PATH = f"{OUTPUT_DIR}/label_encoder.joblib"
AUXILIARY_ENCODERS_PATH = f"{OUTPUT_DIR}/auxiliary_encoders.joblib"

# Other settings
RANDOM_SEED = 42
VALIDATION_SPLIT = 0.15  # より小さな検証セット（データを最大限活用）

# 推論設定
USE_TTA = True  # Test Time Augmentation
TTA_ROUNDS = 3

# Submission settings
SUBMISSION_OUTPUT_PATH = 'submission.csv'
