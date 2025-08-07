"""
ELECTRA-large モデルトレーニングスクリプト
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
    TrainerCallback,
    EarlyStoppingCallback
)
from datasets import Dataset
import joblib
import wandb
import torch

# カスタムモジュールのインポート
from config import *
from utils import prepare_correct_answers, format_input, tokenize_dataset, compute_map3


class BestMap3Callback(TrainerCallback):
    """最高MAP@3スコアを追跡し保存するコールバック"""

    def __init__(self, trainer=None):
        self.best_map3 = 0.0
        self.best_step = 0
        self.trainer = trainer

    def set_trainer(self, trainer):
        """トレーナーを後から設定"""
        self.trainer = trainer

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_map@3" in metrics:
            current_map3 = metrics["eval_map@3"]
            if current_map3 > self.best_map3:
                self.best_map3 = current_map3
                self.best_step = state.global_step
                print(f"\n🎉 New best MAP@3: {self.best_map3:.4f} at step {self.best_step}")

                # best_map3をファイルに保存
                best_map3_path = os.path.join(args.output_dir, "best_map3.txt")
                with open(best_map3_path, "w") as f:
                    f.write(f"Best MAP@3: {self.best_map3:.4f}\n")
                    f.write(f"Step: {self.best_step}\n")

                # 最高MAP@3達成時のモデルを保存
                if self.trainer:
                    print(f"Saving best MAP@3 model to {BEST_MAP3_MODEL_PATH}...")
                    self.trainer.save_model(BEST_MAP3_MODEL_PATH)


def main():
    """メイントレーニング関数"""

    # wandbの初期化
    if USE_WANDB:
        wandb.init(
            project=WANDB_PROJECT,
            name=WANDB_RUN_NAME,
            entity=WANDB_ENTITY,
            config={
                "model_name": MODEL_NAME,
                "epochs": EPOCHS,
                "max_length": MAX_LEN,
                "train_batch_size": TRAIN_BATCH_SIZE,
                "eval_batch_size": EVAL_BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "use_cosine_scheduler": USE_COSINE_SCHEDULER,
                "warmup_steps": WARMUP_STEPS,
                "weight_decay": WEIGHT_DECAY if 'WEIGHT_DECAY' in globals() else 0.01,
            }
        )

    # 出力ディレクトリの作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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

    # --- 入力テキストのフォーマット（ELECTRA用に最適化） ---
    print("Formatting input text for ELECTRA...")
    train['text'] = train.apply(format_input, axis=1)
    print("Example prompt:")
    print(train.text.values[0])

    # --- トークナイザーの初期化 ---
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # --- トークン長の分析 ---
    print("Analyzing token lengths...")
    lengths = [len(tokenizer.encode(t, truncation=False)) for t in train['text']]
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50)
    plt.title("Token Length Distribution (ELECTRA)")
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
    print("Initializing ELECTRA model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=n_classes,
        hidden_dropout_prob=0.1,  # ELECTRAはドロップアウトに敏感
    )

    # Gradient checkpointingを有効化（メモリ効率化）
    if 'GRADIENT_CHECKPOINTING' in globals() and GRADIENT_CHECKPOINTING and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

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
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine" if USE_COSINE_SCHEDULER else "linear",
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY if 'WEIGHT_DECAY' in globals() else 0.01,
        adam_epsilon=ADAM_EPSILON if 'ADAM_EPSILON' in globals() else 1e-8,
        logging_dir=f"{OUTPUT_DIR}/logs",
        fp16=torch.cuda.is_available(),  # GPU利用時はFP16を有効化
        dataloader_num_workers=4,  # データローディングの高速化
        group_by_length=True,  # 似た長さのサンプルをバッチ化
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        save_total_limit=3,
        metric_for_best_model="map@3",
        greater_is_better=True,
        load_best_model_at_end=True,
        report_to="wandb" if USE_WANDB else "none",
        run_name=WANDB_RUN_NAME if USE_WANDB else None,
    )

    # --- トレーナーのセットアップとトレーニング ---
    print("Setting up trainer...")
    best_map3_callback = BestMap3Callback()
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.001
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_map3,
        callbacks=[best_map3_callback, early_stopping_callback],
    )

    # コールバックにトレーナーを設定
    best_map3_callback.set_trainer(trainer)

    print("Starting training...")
    print(f"Training with ELECTRA-large on {len(train_ds)} samples")
    print(f"Validation on {len(val_ds)} samples")
    trainer.train()

    # --- 最終的なMAP@3スコアを表示 ---
    print("\nEvaluating on validation set...")
    eval_results = trainer.evaluate()
    print(f"\nValidation MAP@3: {eval_results.get('eval_map@3', 'N/A'):.4f}")

    # --- 最高MAP@3スコアの表示 ---
    print(f"\n🏆 Best MAP@3: {best_map3_callback.best_map3:.4f} (at step {best_map3_callback.best_step})")

    # --- モデルとエンコーダーの保存 ---
    print("\nSaving model and label encoder...")
    trainer.save_model(BEST_MODEL_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)



    # --- 最終的なベストスコアサマリーの保存 ---
    final_summary_path = os.path.join(OUTPUT_DIR, "training_summary.txt")
    with open(final_summary_path, "w") as f:
        f.write(f"Training Summary\n")
        f.write(f"================\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Total Epochs: {EPOCHS}\n")
        f.write(f"Best MAP@3: {best_map3_callback.best_map3:.4f}\n")
        f.write(f"Best Step: {best_map3_callback.best_step}\n")
        f.write(f"Final MAP@3: {eval_results.get('eval_map@3', 'N/A'):.4f}\n")

    print("Training completed successfully!")
    print(f"Model saved to: {BEST_MODEL_PATH}")
    print(f"Label encoder saved to: {LABEL_ENCODER_PATH}")
    print(f"Best MAP@3 saved to: {os.path.join(OUTPUT_DIR, 'best_map3.txt')}")
    print(f"Best MAP@3 model saved to: {BEST_MAP3_MODEL_PATH}")
    print(f"Training summary saved to: {final_summary_path}")

    # wandbの終了
    if USE_WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()
