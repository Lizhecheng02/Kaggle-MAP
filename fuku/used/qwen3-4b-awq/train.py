"""
Qwen2.5-Math モデルトレーニングスクリプト
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    AutoConfig,
    EarlyStoppingCallback
)
from datasets import Dataset
import joblib
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import AutoModel

# カスタムモジュールのインポート
from config import *
from utils import prepare_correct_answers, format_input, tokenize_dataset, compute_map3
import wandb  # W&Bによる実験管理用


class Qwen2ForSequenceClassification(nn.Module):
    """Qwen2モデルを分類タスク用にカスタマイズ"""
    def __init__(self, model_name, num_labels):
        super().__init__()
        from transformers import AutoModel
        self.qwen = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.qwen.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.qwen(input_ids=input_ids, attention_mask=attention_mask)
        # 最後のトークンの隠れ状態を使用
        pooled_output = outputs.last_hidden_state[:, -1, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return type('Output', (), {'loss': loss, 'logits': logits})()


def main():
    """メイントレーニング関数"""

    # 出力ディレクトリの作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # --- W&B初期化 ---
    is_wandb_enabled = USE_WANDB
    if is_wandb_enabled:
        try:
            wandb.init(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                name=WANDB_RUN_NAME,
                config={
                    "epochs": EPOCHS,
                    "train_batch_size": TRAIN_BATCH_SIZE,
                    "eval_batch_size": EVAL_BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "max_len": MAX_LEN,
                    "seed": RANDOM_SEED,
                    "validation_split": VALIDATION_SPLIT
                }
            )
        except wandb.errors.errors.CommError as e:
            print(f"W&B初期化エラー: {e}。W&Bログを無効化して続行します")
            is_wandb_enabled = False

    # --- データの読み込みと前処理 ---
    print("Loading and preprocessing training data...")
    le = LabelEncoder()
    train = pd.read_csv(TRAIN_DATA_PATH)
    train.Misconception = train.Misconception.fillna('NA')
    train['target'] = train.Category + ":" + train.Misconception
    train['label'] = le.fit_transform(train['target'])
    n_classes = len(le.classes_)
    print(f"Train shape: {train.shape} with {n_classes} target classes")

    # --- 特徴量エンジニアリング ---
    print("Performing feature engineering...")
    correct = prepare_correct_answers(train)
    train = train.merge(correct, on=['QuestionId','MC_Answer'], how='left')
    train.is_correct = train.is_correct.fillna(0)

    # --- 入力テキストのフォーマット ---
    print("Formatting input text...")
    train['text'] = train.apply(format_input, axis=1)
    print("Example prompt for our LLM:")
    print(train.text.values[0])

    # --- トークナイザーの初期化 ---
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # パディングトークンの設定
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- トークン長の分析 ---
    print("Analyzing token lengths...")
    lengths = [len(tokenizer.encode(t, truncation=False)) for t in train['text']]
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50)
    plt.title("Token Length Distribution")
    plt.xlabel("Number of tokens")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(f'{OUTPUT_DIR}/token_length_distribution.png')
    plt.close()

    over_limit = (np.array(lengths) > MAX_LEN).sum()
    print(f"There are {over_limit} train sample(s) with more than {MAX_LEN} tokens")

    # --- データの分割 ---
    print("Splitting data into train and validation sets...")
    train_df, val_df = train_test_split(train, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED)
    COLS = ['text','label']
    train_ds = Dataset.from_pandas(train_df[COLS])
    val_ds = Dataset.from_pandas(val_df[COLS])

    # --- データセットのトークナイズ ---
    print("Tokenizing datasets...")
    train_ds = tokenize_dataset(train_ds, tokenizer, MAX_LEN)
    val_ds = tokenize_dataset(val_ds, tokenizer, MAX_LEN)

    # --- モデルの初期化 ---
    print("Initializing model...")
    try:
        # 量子化モデルを読み込む
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=n_classes,
            trust_remote_code=True,
            device_map=None  # デバイスマッピングを無効化
        )
        # パディングトークンIDを設定
        model.config.pad_token_id = tokenizer.pad_token_id
    except:
        # 失敗した場合はカスタムクラスを使用
        print("Using custom classification head for Qwen2...")
        # ベースモデルを読み込む
        base_model = AutoModel.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            device_map=None
        )
        # カスタム分類ヘッドを作成
        model = Qwen2ForSequenceClassification(MODEL_NAME, n_classes)
        model.qwen = base_model

    # --- LoRAアダプターの設定 ---
    print("Configuring LoRA adapter...")
    lora_config = LoraConfig(
        r=16,  # LoRAのランク
        lora_alpha=32,  # LoRAのスケーリングパラメータ
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # 対象モジュール
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )

    # PEFTモデルの作成
    model = get_peft_model(model, lora_config)
    print("Number of trainable parameters:")
    model.print_trainable_parameters()

    # --- トレーニング引数の設定 ---
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",  # エポック単位で評価
        save_strategy="epoch",        # エポック単位でモデル保存
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=LOGGING_STEPS,
        # save_steps=SAVE_STEPS,  # エポック戦略に切り替えたため不要
        # eval_steps=EVAL_STEPS,  # エポック戦略に切り替えたため不要
        save_total_limit=1,
        metric_for_best_model="map@3",
        greater_is_better=True,
        report_to=["wandb"] if USE_WANDB else "none",
        run_name=WANDB_RUN_NAME if USE_WANDB else None,
        load_best_model_at_end=True,
        fp16=False,  # RTX 5090の互換性問題のため一時的にFalse
        gradient_checkpointing=True,  # メモリ効率のため追加
        gradient_accumulation_steps=4,  # メモリ効率向上のため追加
        lr_scheduler_type="cosine",  # 学習率スケジューラーの設定
        warmup_steps=500,  # 最初の500ステップで線形に学習率を上げる
    )

    # --- トレーナーのセットアップとトレーニング ---
    print("Setting up trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_map3,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)]
    )

    print("Starting training...")
    trainer.train()

    # --- 最終的なMAP@3スコアを表示 ---
    print("\nEvaluating on validation set...")
    eval_results = trainer.evaluate()
    print(f"\nValidation MAP@3: {eval_results.get('eval_map@3', 'N/A'):.4f}")

    # --- モデルとエンコーダーの保存 ---
    print("\nSaving model and label encoder...")
    # LoRAアダプターのみを保存
    model.save_pretrained(BEST_MODEL_PATH)
    # トークナイザーも保存
    tokenizer.save_pretrained(BEST_MODEL_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)
    # --- W&Bモデルアップロード ---
    if USE_WANDB and WANDB_LOG_MODEL:
        print("Logging model checkpoint to W&B...")
        wandb.save(os.path.join(BEST_MODEL_PATH, "*"))
    if USE_WANDB:
        wandb.finish()

    print("Training completed successfully!")
    print(f"Model saved to: {BEST_MODEL_PATH}")
    print(f"Label encoder saved to: {LABEL_ENCODER_PATH}")


if __name__ == "__main__":
    main()
