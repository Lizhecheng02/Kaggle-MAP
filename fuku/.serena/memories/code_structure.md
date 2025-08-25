# ディレクトリ構成（概要）

- ルート
  - `preprocess_train.py`: 学習データ前処理（スペル補正、カテゴリ補正等）
  - `README.md`, `competition.md`, `writeup.md`, `.gitignore`
  - ノートブック: `map-rank-ensemble-5.ipynb`
  - モデル別ディレクトリ（例）
    - `phi-4-reasoning-plus-cot/`, `phi-4-reasoning/`, `qwen3-14b/` ほか
      - `train.py`: 学習
      - `valid.py`: 検証
      - `submit.py`: 予測・提出ファイル生成
      - `utils.py`, `data_collator.py`: 補助関数
      - `config_sample.py`: 設定サンプル（→ `config.py` を作成して利用）
      - （一部）`test_cot_integration.py`: CoT検証用スクリプト
- 出力
  - 学習成果物: `ver_{VER}/` 以下にモデル・エンコーダ・ログ等を保存（各 `config.py` の指定に従う）