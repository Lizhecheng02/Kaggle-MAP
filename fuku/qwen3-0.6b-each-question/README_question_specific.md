# QuestionId別モデル学習・推論システム

このシステムは、各QuestionIdに対して個別のモデルを学習し、推論時も問題ごとにモデルを使い分ける仕組みです。

## システム概要

- **15個のQuestionId**それぞれに対して個別のモデルを学習
- 各モデルはそのQuestionIdのデータのみで学習
- 推論時は自動的に適切なモデルを選択
- 検証も各QuestionId内でtrain/val分割

## ファイル構成

```
qwen3-0.6b-each-question/
├── config.py              # 設定ファイル（QuestionId管理機能追加）
├── train.py               # QuestionId別学習スクリプト
├── submit.py              # QuestionId別推論スクリプト
├── utils.py               # ユーティリティ関数（QuestionId対応）
├── prompts.py             # プロンプト生成関数
├── prompt_utils.py        # 問題データとメタデータ
├── data_collator.py       # データコラレータ
├── tests/                 # テストスクリプト
│   ├── test_question_specific_training.py
│   └── validate_system.py
└── README_question_specific.md
```

## 出力フォルダ構造

```
ver_2/                     # OUTPUT_DIR
├── question_31772/        # QuestionId別ディレクトリ
│   ├── best/             # 最終モデル保存先
│   ├── best_map3/        # 最高MAP@3モデル保存先
│   ├── logs/             # 学習ログ
│   └── label_encoder.joblib
├── question_31774/
│   └── ...
└── summary/              # 全体結果サマリー
    ├── all_questions_summary.json
    └── question_*_results.json
```

## 使用方法

### 1. システムの検証

まず、システムが正常に設定されているかを確認します。

```bash
# 基本的な検証
python tests/validate_system.py

# 詳細なテスト
python tests/test_question_specific_training.py
```

### 2. 学習の実行

全QuestionIdに対してループで学習を実行します。

```bash
python train.py
```

学習プロセス:
- 各QuestionIdのデータをフィルタリング
- QuestionId特有のラベルエンコーダーを作成
- クラス数に応じてモデルを初期化
- LoRAを使用してファインチューニング
- 検証は同じQuestionId内でtrain/val分割
- MAP@3スコアで評価

### 3. 推論の実行

学習済みモデルを使用して推論を実行します。

```bash
python submit.py
```

推論プロセス:
- テストデータをQuestionIdごとにグループ化
- 各QuestionIdに対応するモデルを読み込み
- QuestionIdごとに推論を実行
- 全問題の予測結果を統合して提出ファイルを作成

## 設定変更

### 主要設定（config.py）

```python
# モデル設定
MODEL_NAME = "/hdd/models/qwen-3-0.6b"
EPOCHS = 3
MAX_LEN = 300
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32

# プロンプト設定
PROMPT_VERSION = "create_prompt_v2"

# LoRA設定
LORA_RANK = 64
LORA_ALPHA = 128
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]

# データパス
TRAIN_DATA_PATH = '/kaggle/input/map-charting-student-math-misunderstandings/train.csv'
TEST_DATA_PATH = '/kaggle/input/map-charting-student-math-misunderstandings/test.csv'
```

### QuestionId管理

システムは`prompt_utils.py`の`questions`辞書からQuestionIdのリストを自動取得します。

```python
from config import QUESTION_IDS  # 利用可能なQuestionIdのリスト

# パス管理関数
get_question_output_dir(question_id)      # QuestionId別出力ディレクトリ
get_question_model_path(question_id)      # QuestionId別モデルパス
get_question_label_encoder_path(question_id)  # QuestionId別ラベルエンコーダパス
```

## 期待される効果

1. **問題特化学習**: 各問題の特性に合わせた学習が可能
2. **誤答パターン学習**: 問題ごとの誤答パターンをより正確に学習
3. **性能向上**: 全体的な精度向上が期待できる
4. **並列化**: 将来的にQuestionIdごとの並列学習も可能

## トラブルシューティング

### よくある問題

1. **クラス数不足エラー**
   - 一部のQuestionIdでクラス数が1以下の場合、学習をスキップします
   - ログで確認: "Warning: Question XXX has only N class(es). Skipping..."

2. **メモリ不足**
   - バッチサイズを調整: `TRAIN_BATCH_SIZE`, `EVAL_BATCH_SIZE`
   - グラディエント蓄積: `GRADIENT_ACCUMULATION_STEPS`

3. **モデル読み込みエラー**
   - PEFTライブラリのインストール: `pip install peft`
   - モデルパスの確認

### ログの確認

- 学習ログ: `ver_2/question_*/logs/`
- WandB（有効な場合）: プロジェクト別にQuestionId別のランが作成される
- 結果サマリー: `ver_2/summary/all_questions_summary.json`

### デバッグモード

特定のQuestionIdのみテストしたい場合:

```python
# config.pyで一時的に制限
QUESTION_IDS = ['31772', '31774']  # テスト対象のみ
```

## パフォーマンス監視

システムは各QuestionIdの学習結果を記録し、以下の統計を提供します:

- 成功/失敗したQuestion数
- 各QuestionのMAP@3スコア
- クラス数とサンプル数
- 学習時間とリソース使用量

結果は`ver_2/summary/`に保存されます。