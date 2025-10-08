代表的なコマンド:
- 依存関係インストール（例）: `pip install torch transformers datasets peft bitsandbytes scikit-learn pandas numpy matplotlib wandb pytest`
- 学習実行: `python fuku/qwen2.5-32b-4bit/train.py`
- 検証: `python fuku/qwen2.5-32b-4bit/valid.py`
- 提出作成: `python fuku/qwen2.5-32b-4bit/submit.py`
- テスト実行（例）: `pytest -q`

補助:
- Lint/Format はプロジェクト既存ツール定義なし。必要に応じて `ruff`/`black` などを任意導入

Linuxユーティリティ:
- 検索: `rg PATTERN`
- 一覧: `ls -la`、`tree`（インストールされていれば）
- Git: `git status`, `git diff`, `git log`, `git blame`