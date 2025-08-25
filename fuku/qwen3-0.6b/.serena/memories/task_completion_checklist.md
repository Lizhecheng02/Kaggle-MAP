# タスク完了時のチェックリスト

- 変更の整合性
  - 変数やパスをコード内に直書きしていないか（全て `config.py` 参照に統一）。
  - `config.py` のキー変更が他ファイル（`train.py`/`valid.py`/`submit.py`/`utils.py`）に反映されている。

- 品質確認
  - フォーマット: `black .` / `isort .` を実行。
  - Lint: `flake8 .` もしくは `ruff check .` を実行。

- 動作確認（実行前に venv 確認）
  - Python を実行する前に、使用する venv をどれにするか必ず確認・指定してください。
  - 設定のみ変更した場合でも、`python -c "import train, submit, valid; print('ok')"` などで最低限のImport確認。
  - 学習スクリプトが起動し、初期ステップまで進むことを確認（必要なら `EPOCHS` を小さくしてドライラン）。
  - 提出スクリプトが `submission.csv` を生成できることを確認（軽量データ/小設定で可）。

- ドキュメント
  - `README.md`（運用コマンドや注意点）に変更があれば更新。
  - 新規依存を `suggested_commands.md` に追記。

- リスク/既知の注意点
  - `augmentation` モジュールが未同梱。`USE_DATA_AUGMENTATION=True` にする場合は別途実装を追加。
  - Kaggle 以外の環境ではデータパスを調整すること。