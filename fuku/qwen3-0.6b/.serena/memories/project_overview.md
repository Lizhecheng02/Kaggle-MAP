# プロジェクト概要

- 目的: Kaggle「Map Charting Student Math Misunderstandings」で、Qwen3-0.6BをLoRAで微調整し、学生の誤概念カテゴリをMAP@3で最適化して分類する。
- エントリポイント: `train.py`(学習), `valid.py`(全学習データ推論/検証), `submit.py`(テスト推論+`submission.csv`出力)。
- 設定: `config.py` に全変数を集中管理（モデル名/ロラ設定/学習・評価設定/入出力パス/W&Bなど）。変数はコード中にハードコードしない方針。
- データ: `TRAIN_DATA_PATH`, `TEST_DATA_PATH`（Kaggle入力CSV）。
- 出力: `ver_{VER}` 配下に学習成果物（`best/`, `label_encoder.joblib`, `token_length_distribution.png` など）。提出はプロジェクト直下に `submission.csv`。
- ハードウェア: CUDA GPU を推奨（`CUDA_VISIBLE_DEVICES` を `config.py` で指定）。

## 技術スタック
- 言語/ランタイム: Python 3.x, Linux
- 深層学習: PyTorch, Hugging Face Transformers, Datasets, PEFT (LoRA)
- 前処理/ユーティリティ: pandas, numpy, scikit-learn(LabelEncoder), joblib, matplotlib, tqdm
- 実験管理: Weights & Biases（任意: `USE_WANDB`）

## 主なファイル構成（抜粋）
- `train.py`: 前処理→トークナイズ→LoRA 付き学習→評価→保存。`Trainer` とカスタム `DataCollatorWithPadding` を使用。MAP@3改善で保存するコールバック付き。
- `valid.py`: 学習済みモデル+LoRA を読み込み、全学習データで推論し確率行列を保存（n×クラス数）。
- `submit.py`: 学習済みLoRAモデルと LabelEncoder を読み込み、テストを推論し `submission.csv` を出力。
- `utils.py`: 共有関数（プロンプト整形、MAP@3算出、提出整形など）。
- `data_collator.py`: Qwen 系の入力向けに動的パディングするデータコラレータ。
- `config.py`: すべての設定を一元化。

## ランタイム要件と前提
- Qwen3-0.6B のベースモデルは `MODEL_NAME` で指定（ローカルパス/Hub どちらでも可）。
- Qwen 系モデルの `pad_token` を適切に設定（未設定の場合は0やEOSを使用）。
- Kaggle 環境のファイルパス前提（`/kaggle/input/...`）。ローカル実行時は `config.py` を適宜変更。

## 既知の注意点
- `train.py` には `augmentation` モジュールの参照があるが、レポ内にファイルは見当たらない。`USE_DATA_AUGMENTATION=False`（既定）なら影響なし。必要なら別途追加。