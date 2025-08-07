"""
Qwen-3-32B AWQ モデル推論スクリプト - 提出用予測ファイルの生成
"""

import pandas as pd
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments
)
from datasets import Dataset
import joblib
import torch
import torch.nn as nn
from awq import AutoAWQForCausalLM
from peft import PeftModel, PeftConfig

# カスタムモジュールのインポート
from config import *
from utils import prepare_correct_answers, format_input, tokenize_dataset, create_submission


class QwenAWQForSequenceClassification(nn.Module):
    """Qwen AWQモデルを分類タスク用にカスタマイズ（推論用）"""
    def __init__(self, model_path, num_labels):
        super().__init__()
        # LoRA設定を読み込む
        peft_config = PeftConfig.from_pretrained(model_path)
        
        # ベースモデルを読み込む
        base_model = AutoAWQForCausalLM.from_quantized(
            peft_config.base_model_name_or_path,
            fuse_layers=True,
            trust_remote_code=True,
            safetensors=True,
            device_map="auto"
        )
        
        # LoRAアダプターを適用
        self.model = PeftModel.from_pretrained(base_model.model, model_path)
        
        # 分類ヘッドを追加
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # 最後のトークンの隠れ状態を使用
        pooled_output = outputs.last_hidden_state[:, -1, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return type('Output', (), {'logits': logits})()


def main():
    """メイン推論関数"""

    print("Loading label encoder...")
    # ラベルエンコーダーの読み込み
    le = joblib.load(LABEL_ENCODER_PATH)
    n_classes = len(le.classes_)

    print("Loading trained model and tokenizer...")
    # モデルとトークナイザーの読み込み
    try:
        # まず標準的な方法を試す
        model = AutoModelForSequenceClassification.from_pretrained(
            BEST_MODEL_PATH,
            trust_remote_code=True
        )
    except:
        # 失敗した場合はカスタムクラスを使用
        print("Loading custom AWQ classification model...")
        model = QwenAWQForSequenceClassification(BEST_MODEL_PATH, n_classes)
        # 保存された重みを読み込む（必要に応じて）
        if torch.cuda.is_available():
            checkpoint = torch.load(f"{BEST_MODEL_PATH}/pytorch_model.bin")
        else:
            checkpoint = torch.load(f"{BEST_MODEL_PATH}/pytorch_model.bin", map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)
    
    tokenizer = AutoTokenizer.from_pretrained(BEST_MODEL_PATH, trust_remote_code=True)
    
    # パディングトークンの設定
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading test data...")
    # テストデータの読み込み
    test = pd.read_csv(TEST_DATA_PATH)

    print("Loading training data for correct answers...")
    # 正解答案データの準備（訓練データから取得）
    train = pd.read_csv(TRAIN_DATA_PATH)
    train.Misconception = train.Misconception.fillna('NA')
    correct = prepare_correct_answers(train)

    print("Preprocessing test data...")
    # テストデータの前処理
    test = test.merge(correct, on=['QuestionId','MC_Answer'], how='left')
    test.is_correct = test.is_correct.fillna(0)
    test['text'] = test.apply(format_input, axis=1)

    print("Tokenizing test data...")
    # テストデータのトークナイズ
    ds_test = Dataset.from_pandas(test[['text']])
    ds_test = tokenize_dataset(ds_test, tokenizer, MAX_LEN)

    print("Running inference...")
    # 推論の実行
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=TrainingArguments(
            output_dir="./tmp",  # 一時ディレクトリ（必須パラメータ）
            report_to="none",    # wandbを無効化
            per_device_eval_batch_size=8,  # AWQモデル用にバッチサイズを調整
            fp16=True,  # メモリ効率のため追加
        )
    )
    predictions = trainer.predict(ds_test)

    print("Creating submission file...")
    # 提出用ファイルの作成
    submission = create_submission(predictions, test, le)

    # ファイルの保存
    submission.to_csv(SUBMISSION_OUTPUT_PATH, index=False)
    print(f"Submission file saved to: {SUBMISSION_OUTPUT_PATH}")
    print("\nSubmission preview:")
    print(submission.head())
    print(f"\nSubmission shape: {submission.shape}")


if __name__ == "__main__":
    main()