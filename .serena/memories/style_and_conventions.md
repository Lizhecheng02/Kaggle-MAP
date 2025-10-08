コードスタイル/慣習:
- 設定/可変値は config.py に集約（ハードコード禁止）
- 型ヒントは限定的に使用。Docstring は関数ごとに簡潔な日本語説明あり
- 変数/関数名はスネークケース
- 学習関連は Trainer + callback を活用し、LoRA/量子化の切替は config で制御

設計パターン:
- データ前処理・評価は utils に分離
- Collator を独自実装し DataCollatorWithPadding を使用
- 量子化 + LoRA は PEFT と bitsandbytes で実装

テスト方針（推奨）:
- ユニットテストは tests/ 配下に pytest で配置
- 重い学習を伴わない純粋関数（MAP@3, collator, callback）を優先的にテスト