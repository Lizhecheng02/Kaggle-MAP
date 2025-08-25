# プロジェクト概要

- 目的: Kaggle「MAP: Charting Student Math Misunderstandings」コンペで、学生の自由記述（解答理由）と選択肢から、正誤や誤概念カテゴリを判定し、提出用の予測を生成する。
- アプローチ: LLM（Phi-4, Qwen など）をLoRA等で微調整し、テキスト分類（MAP@3指標）を最適化。CoT（Chain-of-Thought）を活用したプロンプト整形・蒸留も併用。
- 成果物: サブフォルダ（例: `phi-4-reasoning-plus-cot/`）内の `submit.py` により `submission.csv` を生成。