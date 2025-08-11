# config.py: 全体の設定をまとめたモジュール

# GPU設定
CUDA_VISIBLE_DEVICES = "0"

# バージョンと出力ディレクトリ
VER = 1
DIR = f"ver_{VER}"
MODEL_DIR = f"{DIR}/best"

# データパス
TRAIN_CSV_PATH = '/kaggle/input/map-charting-student-math-misunderstandings/train.csv'
TEST_CSV_PATH = '/kaggle/input/map-charting-student-math-misunderstandings/test.csv'

# モデル名（HuggingFace）
MODEL_NAME = '/root/kaggle/map-charting-student-math-misunderstandings/models/deberta-v3-large'

# 学習ハイパーパラメータ
EPOCHS = 20
PER_DEVICE_TRAIN_BATCH_SIZE = 16
PER_DEVICE_EVAL_BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 16
LEARNING_RATE = 0.4e-3

# ロギング・保存設定
LOGGING_DIR = './logs'
LOGGING_STEPS = 50
SAVE_STEPS = 200
EVAL_STEPS = 200
SAVE_TOTAL_LIMIT = 1
METRIC_FOR_BEST_MODEL = 'map@3'
