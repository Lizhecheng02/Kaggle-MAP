"""
学習データ全体での推論スクリプト - 全確率を(n, 65)形状のnumpy配列として出力
20%検証用に訓練されたモデルで学習データ全体に対して推論を実行
"""

import pandas as pd
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
import torch.nn as nn
from datasets import Dataset
import joblib
import torch
from tqdm import tqdm

# PEFTのインポートをオプショナルにする
try:
    from peft import PeftModel, PeftConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not available, will use base model only")

# カスタムモジュールのインポート
from config import *
from utils import prepare_correct_answers, format_input, tokenize_dataset

# Gemmaモデル用のカスタムクラス
class GemmaForSequenceClassification(nn.Module):
    """Gemmaモデルを分類タスク用にカスタマイズ"""
    def __init__(self, model_name, num_labels):
        super().__init__()
        from transformers import AutoModelForCausalLM
        self.gemma = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        self.config = self.gemma.config
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, labels=None, **kwargs):
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        outputs = self.gemma(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            inputs_embeds=inputs_embeds,
            output_hidden_states=True
        )
        hidden_states = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else outputs.last_hidden_state
        pooled_output = hidden_states[:, -1, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        if loss is not None:
            return {'loss': loss, 'logits': logits}
        else:
            return {'logits': logits}


def main():
    """メイン推論関数 - 学習データ全体で推論を実行"""

    print("Loading label encoder...")
    # ラベルエンコーダーの読み込み
    le = joblib.load(LABEL_ENCODER_PATH)
    n_classes = len(le.classes_)
    print(f"Number of classes: {n_classes}")

    print("Loading trained model and tokenizer...")

    if PEFT_AVAILABLE:
        # LoRAアダプターを使用する場合
        print(f"Loading fine-tuned LoRA model from: {BEST_MODEL_PATH}")
        print(f"Loading base model from: {MODEL_NAME}")

        # Gemma3用のカスタムモデルを使用
        try:
            # 最初にAutoModelForSequenceClassificationを試す
            base_model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME,
                num_labels=n_classes,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
            )
        except ValueError:
            # 失敗した場合はカスタムクラスを使用
            print("Using custom classification head for Gemma...")
            base_model = GemmaForSequenceClassification(MODEL_NAME, n_classes)
        
        # LoRAアダプターを適用
        model = PeftModel.from_pretrained(base_model, BEST_MODEL_PATH)

        # トークナイザーはベースモデルから読み込む
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        print("Successfully loaded LoRA fine-tuned model")
    else:
        # PEFTが利用できない場合はエラー
        raise ImportError("PEFT is required to load the fine-tuned model. Please install peft: pip install peft")

    # パディングトークンの設定
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # モデルの設定を更新（PeftModelのbase_modelにアクセス）
    if hasattr(model, 'base_model'):
        model.base_model.config.pad_token_id = tokenizer.pad_token_id
        # 内部のモデルにも設定
        if hasattr(model.base_model, 'model'):
            model.base_model.model.config.pad_token_id = tokenizer.pad_token_id
    else:
        model.config.pad_token_id = tokenizer.pad_token_id

    print("Loading full training data...")
    # 学習データ全体の読み込み
    train_full = pd.read_csv(TRAIN_DATA_PATH)
    train_full.Misconception = train_full.Misconception.fillna('NA')
    print(f"Full training data shape: {train_full.shape}")

    print("Preparing correct answers...")
    # 正解答案データの準備
    correct = prepare_correct_answers(train_full)

    print("Preprocessing training data...")
    # 学習データの前処理（テストデータと同じ形式に）
    train_full = train_full.merge(correct, on=['QuestionId','MC_Answer'], how='left')
    train_full.is_correct = train_full.is_correct.fillna(0)
    train_full['text'] = train_full.apply(format_input, axis=1)

    print("Tokenizing training data...")
    # 学習データのトークナイズ
    ds_train = Dataset.from_pandas(train_full[['text']])
    ds_train = tokenize_dataset(ds_train, tokenizer, MAX_LEN)

    # パディングのためのデータコラレータの設定
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print("Running inference on full training data...")
    print("This may take approximately 1 hour...")
    
    # 推論の実行
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,  # tokenizer の代替
        data_collator=data_collator,  # バッチ時に自動でパディングを適用
        args=TrainingArguments(
            output_dir="./tmp",  # 一時ディレクトリ（必須パラメータ）
            report_to="none",    # wandbを無効化
            per_device_eval_batch_size=EVAL_BATCH_SIZE,  # 設定ファイルから取得
            bf16=True,  # Gemmaモデルでbf16を使用
        )
    )
    
    # 推論実行
    predictions = trainer.predict(ds_train)
    
    print("Converting predictions to probabilities...")
    # ロジットを確率に変換
    logits = predictions.predictions
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()
    
    # 結果の形状を確認
    print(f"Probability array shape: {probs.shape}")
    print(f"Expected shape: ({len(train_full)}, {n_classes})")
    
    # numpy配列として保存
    output_path = f"{OUTPUT_DIR}/train_probabilities.npy"
    np.save(output_path, probs)
    print(f"Probabilities saved to: {output_path}")
    
    # 検証のため、最初の5行のtop-3予測を表示
    print("\nSample predictions (first 5 rows):")
    top3_indices = np.argsort(-probs, axis=1)[:5, :3]
    for i, indices in enumerate(top3_indices):
        top3_labels = le.inverse_transform(indices)
        top3_probs = probs[i][indices]
        print(f"Row {i}: {top3_labels} (probs: {top3_probs})")
    
    print("\nInference completed successfully!")
    print(f"Output shape: {probs.shape}")
    print(f"Output saved as numpy array to: {output_path}")


if __name__ == "__main__":
    main()