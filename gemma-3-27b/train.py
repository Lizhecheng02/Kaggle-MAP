"""
Gemma-3-1B モデルトレーニングスクリプト
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModel,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    AutoConfig,
    BitsAndBytesConfig
)
from datasets import Dataset
import joblib
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import wandb
from transformers import EarlyStoppingCallback, TrainerCallback
import bitsandbytes as bnb

# カスタムモジュールのインポート
from config import *
from utils import prepare_correct_answers, format_input, tokenize_dataset, compute_map3
from data_collator import DataCollatorWithPadding


class SaveBestMap3Callback(TrainerCallback):
    """eval_map@3が最高値を更新した際にモデルを保存するコールバック"""
    def __init__(self, save_dir, tokenizer):
        self.save_dir = save_dir
        self.tokenizer = tokenizer
        self.best_map3 = 0.0

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        current_map3 = metrics.get('eval_map@3', 0.0)

        if current_map3 > self.best_map3:
            self.best_map3 = current_map3
            model = kwargs['model']
            tokenizer = self.tokenizer

            # 専用ディレクトリに保存
            best_map3_path = os.path.join(self.save_dir, 'best_map3')
            os.makedirs(best_map3_path, exist_ok=True)

            # LoRAアダプターのみを保存
            model.save_pretrained(best_map3_path)
            tokenizer.save_pretrained(best_map3_path)

            print(f"\n新しいベストMAP@3スコア: {current_map3:.4f} - モデルを {best_map3_path} に保存しました")

        return control


class GemmaForSequenceClassification(nn.Module):
    """Gemmaモデルを分類タスク用にカスタマイズ"""
    def __init__(self, model_name, num_labels, quantization_config=None):
        super().__init__()
        # AutoModelを使用してGemmaモデルを読み込む
        self.gemma = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map={"": 0},  # 全てをGPU 0に配置
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16
        )
        self.config = self.gemma.config
        # Gemma3の場合はtext_configから hidden_sizeを取得
        if hasattr(self.config, 'text_config'):
            hidden_size = self.config.text_config.hidden_size
        else:
            hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, labels=None, **kwargs):
        # input_ids または inputs_embeds のいずれかが必要
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        outputs = self.gemma(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True
        )
        
        # 最後の隠れ状態を取得
        last_hidden_state = outputs.last_hidden_state
        
        # attention_maskを使用して最後の有効なトークンの位置を取得
        if attention_mask is not None:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_state.shape[0]
            pooled_output = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
        else:
            # attention_maskがない場合は最後の位置を使用
            pooled_output = last_hidden_state[:, -1, :]
            
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 辞書形式で返す（Transformersが期待する形式）
        if loss is not None:
            return {'loss': loss, 'logits': logits}
        else:
            return {'logits': logits}


def main():
    """メイントレーニング関数"""

    # WandBの初期化
    if USE_WANDB:
        wandb.init(
            project=WANDB_PROJECT,
            name=WANDB_RUN_NAME,
            entity=WANDB_ENTITY,
            config={
                "model_name": MODEL_NAME,
                "epochs": EPOCHS,
                "max_len": MAX_LEN,
                "train_batch_size": TRAIN_BATCH_SIZE,
                "eval_batch_size": EVAL_BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "early_stopping_patience": EARLY_STOPPING_PATIENCE if USE_EARLY_STOPPING else None,
                "early_stopping_threshold": EARLY_STOPPING_THRESHOLD if USE_EARLY_STOPPING else None,
                "lora_r": LORA_R,
                "lora_alpha": LORA_ALPHA,
                "lora_dropout": LORA_DROPOUT,
                "lora_target_modules": LORA_TARGET_MODULES,
                "use_gradient_checkpointing": USE_GRADIENT_CHECKPOINTING,
                "use_8bit_adam": USE_8BIT_ADAM,
                "max_grad_norm": MAX_GRAD_NORM,
                "quantization": "4bit",
            }
        )

    # GPU設定
    if CUDA_VISIBLE_DEVICES is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
        print(f"Using CUDA device(s): {CUDA_VISIBLE_DEVICES}")

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

    # --- 入力テキストのフォーマット ---
    print("Formatting input text...")
    train['text'] = train.apply(format_input, axis=1)
    print("Example prompt for our LLM:")
    print(train.text.values[0])

    # --- トークナイザーの初期化 ---
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # パディングトークンの設定
    # Gemmaモデルの場合、eos_tokenをpad_tokenとして使用
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

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

    # --- ラベルエンコーダーの保存（学習前に保存） ---
    print("Saving label encoder...")
    joblib.dump(le, LABEL_ENCODER_PATH)
    print(f"Label encoder saved to: {LABEL_ENCODER_PATH}")

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

    # --- 4bit量子化の設定 ---
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    # --- モデルの初期化 ---
    print("Initializing model with 4-bit quantization...")
    # カスタム分類ヘッドを使用
    model = GemmaForSequenceClassification(MODEL_NAME, n_classes, quantization_config=quantization_config)
    # パディングトークンIDを設定
    model.config.pad_token_id = tokenizer.pad_token_id

    # --- LoRAアダプターの設定 ---
    print("Configuring LoRA adapter...")
    lora_config = LoraConfig(
        r=LORA_R,  # LoRAのランク
        lora_alpha=LORA_ALPHA,  # LoRAのスケーリングパラメータ
        target_modules=LORA_TARGET_MODULES,  # Gemma用の対象モジュール
        lora_dropout=LORA_DROPOUT,
        bias=LORA_BIAS,
        task_type=TaskType.SEQ_CLS,
    )

    # PEFTモデルの作成
    model = get_peft_model(model, lora_config)
    print("Number of trainable parameters:")
    model.print_trainable_parameters()

    # Gradient Checkpointingの設定
    if USE_GRADIENT_CHECKPOINTING:
        # PEFTモデルの場合は基になるモデルで設定
        if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
            # カスタムモデルの場合、内部のgemmaモデルで設定
            if hasattr(model.base_model.model.gemma, 'gradient_checkpointing_enable'):
                model.base_model.model.gemma.gradient_checkpointing_enable()
        elif hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    # device_map="auto"を使用しているため、手動でGPUに移動する必要はない
    print(f"Model loaded with automatic device mapping across {torch.cuda.device_count()} GPUs")

    # --- トレーニング引数の設定 ---
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=LOGGING_STEPS,
        save_total_limit=1,
        metric_for_best_model="map@3",
        greater_is_better=True,
        load_best_model_at_end=True,
        report_to="wandb" if USE_WANDB else "none",
        bf16=True,  # RTX 5090の互換性問題のため一時的にFalse
        gradient_checkpointing=False,  # カスタムモデルではFalseに設定
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,  # メモリ効率向上のため追加
        remove_unused_columns=False,  # カラムを削除しない
        lr_scheduler_type="cosine",  # コサインスケジューラーを使用
        warmup_ratio=0.0,  # ウォームアップを無効化
        optim="adamw_bnb_8bit" if USE_8BIT_ADAM else "adamw_torch",  # 8-bit Adam optimizer
        max_grad_norm=MAX_GRAD_NORM,  # Gradient clipping
    )

    # --- トレーナーのセットアップとトレーニング ---
    print("Setting up trainer...")

    # エポックあたりのステップ数を計算
    steps_per_epoch = len(train_ds) // (TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)  # gradient_accumulation_steps=2のため
    total_steps = steps_per_epoch * EPOCHS
    print(f"\nDataset statistics:")
    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    print(f"Batch size: {TRAIN_BATCH_SIZE} (with gradient accumulation: {TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total training steps: {total_steps}")
    print(f"Evaluation interval: every {EVAL_STEPS} steps (~{EVAL_STEPS/steps_per_epoch:.2f} epochs)")

    # カスタムデータコレーターを使用
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=MAX_LEN)

    # コールバックの設定
    callbacks = []

    # MAP@3最高値保存コールバックを追加
    save_best_map3_callback = SaveBestMap3Callback(OUTPUT_DIR, tokenizer)
    callbacks.append(save_best_map3_callback)
    print(f"SaveBestMap3Callback enabled - models will be saved to {OUTPUT_DIR}/best_map3/")

    # アーリーストッピングコールバックの設定
    if USE_EARLY_STOPPING:
        # EARLY_STOPPING_PATIENCEは評価回数として使用
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            early_stopping_threshold=EARLY_STOPPING_THRESHOLD
        )
        callbacks.append(early_stopping_callback)
        print(f"Early stopping enabled:")
        print(f"  - Patience (evaluations without improvement): {EARLY_STOPPING_PATIENCE}")
        print(f"  - Threshold: {EARLY_STOPPING_THRESHOLD}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_map3,
        callbacks=callbacks,
    )

    print("Starting training...")
    trainer.train()

    # --- 最終的なMAP@3スコアを表示 ---
    print("\nEvaluating on validation set...")
    eval_results = trainer.evaluate()
    print(f"\nValidation MAP@3: {eval_results.get('eval_map@3', 'N/A'):.4f}")

    # --- モデルの保存 ---
    print("\nSaving model...")
    # LoRAアダプターのみを保存
    model.save_pretrained(BEST_MODEL_PATH)
    # トークナイザーも保存
    tokenizer.save_pretrained(BEST_MODEL_PATH)

    print("Training completed successfully!")
    print(f"Model saved to: {BEST_MODEL_PATH}")
    print(f"Label encoder saved to: {LABEL_ENCODER_PATH}")

    # WandBの終了
    if USE_WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()
