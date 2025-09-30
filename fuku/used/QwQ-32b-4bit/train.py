"""
QwQ-32B モデルトレーニングスクリプト
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
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
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, prepare_model_for_kbit_training
from transformers import AutoModel
import wandb
from transformers import EarlyStoppingCallback, TrainerCallback
import gc

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

    def on_evaluate(self, args, state, control, metrics, model=None, **kwargs):
        current_map3 = metrics.get('eval_map@3', 0.0)

        if current_map3 > self.best_map3:
            self.best_map3 = current_map3

            # 専用ディレクトリに保存
            best_map3_path = os.path.join(self.save_dir, 'best_map3')
            os.makedirs(best_map3_path, exist_ok=True)

            # LoRAアダプターのみを保存
            model.save_pretrained(best_map3_path)
            self.tokenizer.save_pretrained(best_map3_path)

            print(f"\n新しいベストMAP@3スコア: {current_map3:.4f} - モデルを {best_map3_path} に保存しました")

        return control


class QwQForSequenceClassification(nn.Module):
    """QwQ-32Bモデルを分類タスク用にカスタマイズ"""
    def __init__(self, model_name, num_labels, quantization_config=None):
        super().__init__()
        if quantization_config is not None:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                quantization_config=quantization_config,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
        else:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.base_model.config.hidden_size, num_labels)
        self.config = self.base_model.config

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        inputs_embeds=None,
        **kwargs,
    ):
        # PEFTのk-bit学習や勾配チェックポイント有効時にinputs_embedsが渡される場合がある
        if inputs_embeds is not None:
            outputs = self.base_model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )
        else:
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # 最後のトークンの隠れ状態を使用
        pooled_output = outputs.last_hidden_state[:, -1, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return type("Output", (), {"loss": loss, "logits": logits})()


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
                "lora_rank": LORA_RANK,
                "lora_alpha": LORA_ALPHA,
                "lora_target_modules": LORA_TARGET_MODULES,
                "lora_dropout": LORA_DROPOUT,
                "lora_bias": LORA_BIAS,
            }
        )

    # GPU設定
    if CUDA_VISIBLE_DEVICES is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
        print(f"Using CUDA device(s): {CUDA_VISIBLE_DEVICES}")

    # メモリキャッシュをクリア
    torch.cuda.empty_cache()
    gc.collect()

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
    # Qwen3モデルの場合、特別なトークンIDを使用
    if tokenizer.pad_token is None:
        # 語彙内の安全なトークンIDを使用
        # Qwenモデルでは、0番のトークンがUNKNOWNトークンとして使われることが多い
        tokenizer.pad_token_id = 0
        tokenizer.pad_token = tokenizer.decode([0])

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

    # --- Label Encoderの保存 ---
    print(f"Saving label encoder to: {LABEL_ENCODER_PATH}")
    joblib.dump(le, LABEL_ENCODER_PATH)

    # --- モデルの初期化 ---
    print("Initializing model...")

    # 4bit量子化設定の準備
    quantization_config = None
    if USE_4BIT_QUANTIZATION:
        print("Setting up 4-bit quantization...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, BNB_4BIT_COMPUTE_DTYPE),
            bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
            bnb_4bit_use_double_quant=BNB_4BIT_USE_DOUBLE_QUANT,
            bnb_4bit_quant_storage_dtype=getattr(torch, BNB_4BIT_QUANT_STORAGE_DTYPE)
        )
        print(f"4-bit quantization enabled with {BNB_4BIT_QUANT_TYPE} quantization")

    # QwQ-32Bは分類用のヘッドが標準では提供されていないため、カスタムクラスを使用
    print("Using custom classification head for QwQ-32B...")
    model = QwQForSequenceClassification(MODEL_NAME, n_classes, quantization_config)
    # パディングトークンIDを設定
    model.config.pad_token_id = tokenizer.pad_token_id

    # --- LoRAアダプターの設定 ---
    print("Configuring LoRA adapter...")
    lora_config = LoraConfig(
        r=LORA_RANK,  # LoRAのランク
        lora_alpha=LORA_ALPHA,  # LoRAのスケーリングパラメータ
        target_modules=LORA_TARGET_MODULES,  # 対象モジュール
        lora_dropout=LORA_DROPOUT,
        bias=LORA_BIAS,
        task_type=TaskType.SEQ_CLS,
    )

    # 4bit量子化モデルをトレーニング用に準備
    if USE_4BIT_QUANTIZATION:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    
    # PEFTモデルの作成
    model = get_peft_model(model, lora_config)
    print("Number of trainable parameters:")
    model.print_trainable_parameters()

    # Gradient checkpointingを有効化
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()

    # モデルのgradient checkpointingを有効化（サポートされる場合）
    if hasattr(model.base_model, 'gradient_checkpointing_enable'):
        model.base_model.gradient_checkpointing_enable()
    elif hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()

    # Trainerがコール可能か判定（Peft経由で透過的に解決される想定）
    supports_gradient_checkpointing = (
        hasattr(model, 'gradient_checkpointing_enable') or
        hasattr(getattr(model, 'base_model', None), 'gradient_checkpointing_enable')
    )
    print(f"Gradient checkpointing supported: {supports_gradient_checkpointing}")

    # モデルは既にdevice_map="auto"でGPUに配置されている

    # 追加のメモリ最適化
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

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
        metric_for_best_model="map@3",
        greater_is_better=True,
        load_best_model_at_end=True,
        report_to="wandb" if USE_WANDB else "none",
        bf16=False,  # 4bit量子化時は混合精度を無効化
        fp16=False,  # 4bit量子化時は混合精度を無効化
        gradient_checkpointing=supports_gradient_checkpointing,  # サポートされる場合のみ有効化
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,  # メモリ効率向上のため
        remove_unused_columns=False,  # カラムを削除しない
        lr_scheduler_type="cosine",  # コサインスケジューラーを使用
        warmup_ratio=0.0,  # ウォームアップを無効化
        save_total_limit=2,
        max_grad_norm=MAX_GRAD_NORM,  # Gradient clipping
        optim="paged_adamw_8bit" if USE_8BIT_ADAM else "adamw_torch",  # 8-bit Adam optimizer
    )

    # --- トレーナーのセットアップとトレーニング ---
    print("Setting up trainer...")

    # エポックあたりのステップ数を計算
    steps_per_epoch = len(train_ds) // (TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)  # gradient_accumulation_stepsを考慮
    total_steps = steps_per_epoch * EPOCHS
    print(f"\nDataset statistics:")
    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    print(f"Batch size: {TRAIN_BATCH_SIZE} (with gradient accumulation: {TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total training steps: {total_steps}")
    print(f"Evaluation interval: every {EVAL_STEPS} steps (~{EVAL_STEPS/steps_per_epoch:.2f} epochs)")
    print(f"Early stopping after {EARLY_STOPPING_PATIENCE} evaluations without improvement")

    # カスタムデータコレクレーターを使用
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=MAX_LEN)

    # 早期停止・ベスト保存などのコールバック設定
    callbacks = []

    # SaveBestMap3Callbackを追加
    save_best_callback = SaveBestMap3Callback(save_dir=OUTPUT_DIR, tokenizer=tokenizer)
    callbacks.append(save_best_callback)
    print(f"SaveBestMap3Callback enabled - モデルは {OUTPUT_DIR}/best_map3 に保存されます")

    if USE_EARLY_STOPPING:
        # EARLY_STOPPING_PATIENCEは評価回数として直接使用
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
