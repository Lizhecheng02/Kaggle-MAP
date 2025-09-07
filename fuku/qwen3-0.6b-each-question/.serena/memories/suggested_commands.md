# 推奨コマンド集（Linux）

- 環境準備
  - venv 作成: `python3 -m venv .venv`
  - venv 有効化: `source .venv/bin/activate`（実行前にどの venv を使うか必ず確認・指定してください）
  - 依存関係（例）: `pip install torch transformers datasets peft scikit-learn pandas numpy matplotlib joblib wandb tqdm`

- 学習/検証/提出
  - 学習: `python train.py`
  - 全学習データ推論: `python valid.py`
  - 提出ファイル生成: `python submit.py`（`submission.csv` 出力）

- 設定の切り替え
  - `config.py` を編集（モデル/LoRA/バッチサイズ/保存先など）。変数はコード内に直接書かない。

- フォーマット/静的解析（任意）
  - フォーマット: `black .` / `isort .`
  - Lint: `flake8 .` または `ruff check .`

- Git 基本
  - 変更確認: `git status`
  - 差分: `git diff`
  - コミット: `git add -A && git commit -m "msg"`

- Linux ユーティリティ
  - ファイル一覧: `ls -la`
  - 検索: `grep -R "pattern" -n .`
  - ファイル探索: `find . -name "*.py"`

- 実行時の注意
  - 実行前に使用する venv を必ず指定してください（本プロジェクトでは Python 実行ごとに確認が必要）。
  - Kaggle 環境外で実行する場合は `TRAIN_DATA_PATH`/`TEST_DATA_PATH` のパスを適宜変更。