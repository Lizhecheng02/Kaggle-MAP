# 技術スタック

- 言語/環境: Python 3.x, Linux（Kaggle/ローカルGPU環境想定）
- ライブラリ: 
  - モデル/学習: PyTorch, Hugging Face Transformers, PEFT (LoRA), Datasets
  - 前処理/分析: pandas, numpy, scikit-learn, joblib, matplotlib
  - ロギング: Weights & Biases（任意）
- モデル: Phi-4 系, Qwen 系, 他（`gpt-oss-20b`, `exaone4-32b`, `AceMath-7B-Instruct` など）
- データ: Kaggle 提供の `train.csv`, `test.csv`（パスは各 `config.py` で指定）