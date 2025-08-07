"""
Deberta モデルトレーニングスクリプト
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
import joblib
import wandb

# カスタムモジュールのインポート
from config import *
from utils import prepare_correct_answers, format_input, tokenize_dataset, compute_map3


def main():
    """メイントレーニング関数"""

    # WANDBの初期化
    if REPORT_TO == "wandb":
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=WANDB_RUN_NAME,
            config={
                "model_name": MODEL_NAME,
                "epochs": EPOCHS,
                "max_len": MAX_LEN,
                "train_batch_size": TRAIN_BATCH_SIZE,
                "eval_batch_size": EVAL_BATCH_SIZE,
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
                "learning_rate": LEARNING_RATE,
                "warmup_steps": WARMUP_STEPS,
                "scheduler_type": SCHEDULER_TYPE,
            }
        )

    # 出力ディレクトリの作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- データの読み込みと前処理 ---
    print("Loading and preprocessing training data...")
    le = LabelEncoder()
    train = pd.read_csv(TRAIN_DATA_PATH)
    
    # 合成データの読み込み
    if USE_SYNTHETIC_DATA:
        print("Loading synthetic data...")
        synthetic_data = pd.read_csv(SYNTHETIC_DATA_PATH, encoding='utf-8', encoding_errors='replace')
        print(f"Synthetic data shape: {synthetic_data.shape}")
        
        # 元のデータと合成データを結合
        original_train_shape = train.shape
        train = pd.concat([train, synthetic_data], ignore_index=True)
        print(f"Combined data shape: {train.shape} (original: {original_train_shape[0]}, synthetic: {synthetic_data.shape[0]})")
    
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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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
    # 検証データは元のデータのみから作成するため、先にオリジナルデータをマーク
    if USE_SYNTHETIC_DATA:
        # オリジナルデータのインデックスを記録
        original_indices = list(range(original_train_shape[0]))
        # オリジナルデータのみを使用して検証データを作成
        original_data = train.iloc[original_indices]
        train_original_df, val_df = train_test_split(original_data, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED)
        
        # 合成データをトレーニングデータに追加
        synthetic_indices = list(range(original_train_shape[0], len(train)))
        synthetic_df = train.iloc[synthetic_indices]
        train_df = pd.concat([train_original_df, synthetic_df], ignore_index=True)
        
        print(f"Training set: {len(train_df)} samples (original: {len(train_original_df)}, synthetic: {len(synthetic_df)})")
        print(f"Validation set: {len(val_df)} samples (original data only)")
    else:
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
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=n_classes,
        reference_compile=REFERENCE_COMPILE,
    )

    # --- トレーニング引数の設定 ---
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        save_strategy="steps",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        save_total_limit=2,
        metric_for_best_model="map@3",
        greater_is_better=True,
        load_best_model_at_end=True,
        report_to=REPORT_TO,
        warmup_steps=WARMUP_STEPS,
        lr_scheduler_type=SCHEDULER_TYPE,
        bf16=BF16,
        fp16=FP16,
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
    )

    print("Starting training...")
    trainer.train()

    # --- 最終的なMAP@3スコアを表示 ---
    print("\nEvaluating on validation set...")
    eval_results = trainer.evaluate()
    print(f"\nValidation MAP@3: {eval_results.get('eval_map@3', 'N/A'):.4f}")

    # --- モデルとエンコーダーの保存 ---
    print("\nSaving model and label encoder...")
    trainer.save_model(BEST_MODEL_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)

    print("Training completed successfully!")
    print(f"Model saved to: {BEST_MODEL_PATH}")
    print(f"Label encoder saved to: {LABEL_ENCODER_PATH}")


if __name__ == "__main__":
    main()
