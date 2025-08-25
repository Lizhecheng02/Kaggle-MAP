# よく使うコマンド集

注意: Python実行時は必ず使用するvenvを確認・指定してください（例: `source <VENV_PATH>/bin/activate`）。

## 初期設定（任意のモデルディレクトリにて）
- `cd phi-4-reasoning-plus-cot`（例）
- `cp config_sample.py config.py`（設定ファイル作成）
- `vi config.py`（`MODEL_NAME`, `TRAIN_DATA_PATH`, `TEST_DATA_PATH`, 出力先などを編集）

## 学習/検証/推論
- 学習: `python train.py`
- 検証: `python valid.py`
- 提出作成: `python submit.py`（`submission.csv` を出力）
- CoT統合テスト（ある場合）: `python test_cot_integration.py`

## データ前処理（必要に応じて）
- 前処理の関数利用例:
  - `python - <<'PY'\nimport pandas as pd\nfrom preprocess_train import preprocess_data\ndf = pd.read_csv('train.csv')\ndf2 = preprocess_data(df)\ndf2.to_csv('train_preprocessed.csv', index=False)\nPY`

## Linuxユーティリティ
- 一覧/確認: `ls -la`, `pwd`
- 検索: `grep -R "pattern" .`, `find . -name "*.py"`
- Git: `git status`, `git diff`, `git log --oneline`

## Kaggle 実行の注意
- `config.py` の `TRAIN_DATA_PATH`, `TEST_DATA_PATH` を `/kaggle/input/...` に合わせる。
- 大規模モデルはローカルパス（例: `/hdd/models/...`）を適切に設定。