目的: Kaggle『map-charting-student-math-misunderstandings』で、Qwen 系 LLM をLoRA + 4bit量子化で学習し、MAP@3 を最大化する。

技術スタック:
- Python 3.x
- PyTorch, bitsandbytes (4bit), PEFT (LoRA), Transformers, Datasets
- scikit-learn, pandas, numpy, matplotlib
- Weights & Biases (任意)

主要エントリポイント/スクリプト:
- fuku/qwen2.5-32b-4bit/train.py: 学習
- fuku/qwen2.5-32b-4bit/valid.py: 検証
- fuku/qwen2.5-32b-4bit/submit.py: 予測・提出生成
- fuku/qwen2.5-32b-4bit/config.py: 各種設定
- fuku/qwen2.5-32b-4bit/utils.py: 前処理・評価関数
- fuku/qwen2.5-32b-4bit/data_collator.py: カスタムデータコレーター

コード構成の概略:
- config.py でパラメータ集中管理
- train.py が学習パイプライン全体（データ読み込み→トークナイズ→LoRA/量子化設定→Trainer学習→保存）
- utils.py に MAP@3 計算、プロンプト整形など
- data_collator.py に動的パディングの実装

注意点:
- 大規模モデルのためGPU/VRAM要件が高い。4bit + LoRA 前提
- WandB を使う場合は通信が必要
- Kaggle データパスは config.py に設定済み