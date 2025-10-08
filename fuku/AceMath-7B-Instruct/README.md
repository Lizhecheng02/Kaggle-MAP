# AceMath-7B-Instruct 学習・推論パイプライン

本リポジトリは、Kaggle「Map Charting Student Math Misunderstandings」データセットを対象に、AceMath-7B-Instruct を用いたLoRA微調整・推論・提出ファイル生成を行うスクリプト群です。Phi-4 向けの実装から AceMath-7B-Instruct 向けに移行済みです。

## 構成
- `config.py`: 変数・ハイパラ・経路など全設定を一元管理
- `train.py`: 学習（LoRA での微調整）
- `valid.py`: 全学習データに対する推論（確率配列の保存）
- `submit.py`: テストデータに対する推論と提出ファイル生成
- `utils.py`: 前処理・プロンプト作成・評価関数など
- `data_collator.py`: 動的パディングのためのデータコラレータ

## 前提
- Python 仮想環境（venv）を使用します。例: `/root/kaggle/myenv`
- 主要ライブラリ: `transformers`, `peft`, `datasets`, `torch`, `wandb`
- モデル配置: 例 `/hdd/models/AceMath-7B-Instruct`
- データ配置: `/kaggle/input/map-charting-student-math-misunderstandings/{train.csv,test.csv}`

## セットアップ
```bash
# venv を有効化（環境に合わせて変更してください）
source /root/kaggle/myenv/bin/activate

# バージョン確認（参考）
python -V
pip list | egrep -i 'transformers|peft|datasets|wandb|torch'
```

必要に応じて `config.py` の各設定値を編集してください（ハードコードは行わず、必ず設定から制御します）。

## 設定 (`config.py`) の主な項目
- モデル/学習
  - `MODEL_NAME`: ベースモデルのパス（例: `/hdd/models/AceMath-7B-Instruct`）
  - `EPOCHS`, `MAX_LEN`, `TRAIN_BATCH_SIZE`, `EVAL_BATCH_SIZE`, `GRADIENT_ACCUMULATION_STEPS`, `LEARNING_RATE`
  - `ATTENTION_IMPLEMENTATION`: `"eager"` または `"flash_attention_2"`
  - `USE_GRADIENT_CHECKPOINTING`: 勾配チェックポイントの有効化
- LoRA
  - `LORA_RANK`, `LORA_ALPHA`, `LORA_TARGET_MODULES`, `LORA_DROPOUT`, `LORA_BIAS`, `USE_DORA`
- データ/出力
  - `TRAIN_DATA_PATH`, `TEST_DATA_PATH`, `OUTPUT_DIR`
  - `BEST_MODEL_PATH`, `LABEL_ENCODER_PATH`, `SUBMISSION_OUTPUT_PATH`
- 早期終了・WandB
  - `USE_EARLY_STOPPING`, `EARLY_STOPPING_PATIENCE`, `EARLY_STOPPING_THRESHOLD`
  - `USE_WANDB`, `WANDB_PROJECT`, `WANDB_RUN_NAME`, `WANDB_ENTITY`
- パディング（重要）
  - `USE_EOS_AS_PAD`: `pad_token` が無いモデルで `eos_token` をパディングとして流用（推奨）
  - `PAD_TOKEN_STR`, `PAD_TOKEN_ID`: 特定の `pad_token` を使いたい場合に明示指定

## プロンプト仕様（utils.format_input）
Phi の特殊タグを廃し、一般的なテキスト分類向けのプレーンプロンプトを使用しています。
```
Task: Classify the student's misconception category based on the question, their chosen answer, correctness, and the explanation.

Question: ...
Chosen Answer: ...
Correct?: Yes/No
Explanation: ...
```

## 学習（LoRA 微調整）
```bash
source /root/kaggle/myenv/bin/activate
python train.py
```
- 初回実行時、`OUTPUT_DIR` 以下に以下が生成されます：
  - ベストアダプタ: `BEST_MODEL_PATH`（LoRA のみ保存）
  - トークナイザ: `BEST_MODEL_PATH`
  - ラベルエンコーダ: `LABEL_ENCODER_PATH`
  - 可視化: `token_length_distribution.png`
- WandB を無効化する場合: `config.py` の `USE_WANDB=False`

## 検証（全学習データ推論）
```bash
source /root/kaggle/myenv/bin/activate
python valid.py
```
- 出力: `OUTPUT_DIR/train_probabilities.npy`（形状: `[n_samples, n_classes]`）

## 提出ファイル作成
```bash
source /root/kaggle/myenv/bin/activate
python submit.py
```
- 出力: `SUBMISSION_OUTPUT_PATH`（既定: `submission.csv`）
- 内部で LoRA アダプタをベースモデルに適用して推論します

## 実装上のポイント（AceMath-7B-Instruct 対応）
- PAD の扱いを `config.py` で統一（Phi 固有の PAD を廃止）
- AutoModelForSequenceClassification が使えない場合に備え、
  `GenericEncoderForSequenceClassification` でフォールバック
- 勾配チェックポイントや最適化設定は設定ファイルから制御

## よくある問題と対処
- CUDA メモリ不足
  - `TRAIN_BATCH_SIZE` を下げる / `GRADIENT_ACCUMULATION_STEPS` を上げる
  - `USE_GRADIENT_CHECKPOINTING=True` を維持
  - 可能なら `ATTENTION_IMPLEMENTATION="flash_attention_2"`
- パディング関連のエラー
  - `USE_EOS_AS_PAD=True` を維持、または `PAD_TOKEN_STR`/`PAD_TOKEN_ID` を明示
- WandB に接続できない
  - `USE_WANDB=False` に設定

## 再現のヒント
- 実行前に venv を必ず有効化してください
- 環境・資源に応じて `config.py` を調整し、コード側にはハードコードしない方針です

---
改善や追加が必要であればお知らせください。短時間のスモークテストやハイパラ調整のテンプレも用意できます。
