import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import Dataset
import joblib
import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModel
import wandb
from transformers import EarlyStoppingCallback
import gc

# Import Custom Modules
from config import *
from models import LLMForSequenceClassification
from utils import (
    prepare_correct_answers,
    format_input,
    tokenize_dataset,
    compute_map3,
    SaveBestMap3Callback,
)

import warnings

warnings.filterwarnings("ignore")


def main():
    # WANDB
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
                "early_stopping_patience": EARLY_STOPPING_PATIENCE
                if USE_EARLY_STOPPING
                else None,
                "lora_rank": LORA_RANK,
                "lora_alpha": LORA_ALPHA,
                "lora_target_modules": LORA_TARGET_MODULES,
                "lora_dropout": LORA_DROPOUT,
                "lora_bias": LORA_BIAS,
            },
        )

    # GPU
    if CUDA_VISIBLE_DEVICES is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
        print(f"Using CUDA device(s): {CUDA_VISIBLE_DEVICES}")

    # Clear Memory Cache
    torch.cuda.empty_cache()
    gc.collect()

    # Create Output Directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Data Loading and Preprocessing ---
    print("Loading and preprocessing training data...")
    le = LabelEncoder()

    # Preprocess
    train = pd.read_parquet(TRAIN_DATA_PATH)

    train.Misconception = train.Misconception.fillna("NA")
    train["target"] = train.Category + ":" + train.Misconception

    train["label"] = le.fit_transform(train["target"])
    n_classes = len(le.classes_)
    print(f"All train shape: {train.shape} with {n_classes} target classes")

    # --- Feature Engineering ---
    print("Performing feature engineering...")
    correct = prepare_correct_answers(train)
    train = train.merge(correct, on=["QuestionId", "MC_Answer"], how="left")
    train.is_correct = train.is_correct.fillna(0)

    # --- Formatting input text ---
    print("Formatting input text...")
    train["text"] = train.apply(format_input, axis=1)
    print("Example prompt for our LLM:")
    print(train.text.values[0])

    # --- Tokenizer ---
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Only for Phi-4
    if "microsoft/phi-4" in MODEL_NAME.lower():
        if tokenizer.pad_token is None:
            tokenizer.pad_token = "<|finetune_right_pad_id|>"
            tokenizer.pad_token_id = 100257

    # --- Dataset Token Length ---
    print("Analyzing token lengths...")
    lengths = [len(tokenizer.encode(t, truncation=False)) for t in train["text"]]
    over_limit = (np.array(lengths) > MAX_LEN).sum()
    print(f"There are {over_limit} train sample(s) with more than {MAX_LEN} tokens")

    # --- Splitting Data ---
    print("Preparing training data...")
    if TRAIN_FULL_DATA:
        print("Training on FULL dataset without validation split")
        train_df = train

        # Will select a meaningless sample
        val_df = train_df.sample(n=4, random_state=42)

        COLS = ["text", "label"]
        train_ds = Dataset.from_pandas(train_df[COLS])
        val_ds = Dataset.from_pandas(val_df[COLS])
    else:
        print(f"Splitting data with validation ratio: {VALIDATION_SPLIT}")
        train_df, val_df = train_test_split(
            train, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED
        )

        COLS = ["text", "label"]
        train_ds = Dataset.from_pandas(train_df[COLS])
        val_ds = Dataset.from_pandas(val_df[COLS])

    # --- Tokenizing Dataset ---
    print("Tokenizing datasets...")
    train_ds = tokenize_dataset(train_ds, tokenizer, MAX_LEN)
    if val_ds:
        val_ds = tokenize_dataset(val_ds, tokenizer, MAX_LEN)

    # --- Label Encoder ---
    print(f"Saving label encoder to: {LABEL_ENCODER_PATH}")
    joblib.dump(le, LABEL_ENCODER_PATH)

    # --- Initializing Model ---
    print("Initializing model...")

    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=n_classes,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
        )
        model.config.pad_token_id = tokenizer.pad_token_id

        # --- LoRA ---
        print("Configuring LoRA adapter...")
        lora_config = LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias=LORA_BIAS,
            task_type=TaskType.SEQ_CLS,
            use_dora=USE_DORA,
        )

    except:
        print("Using custom classification head for Phi-4...")
        base_model = AutoModel.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
        )

        model = LLMForSequenceClassification(base_model, n_classes)

        print("Configuring LoRA adapter...")
        lora_config = LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias=LORA_BIAS,
            task_type=TaskType.FEATURE_EXTRACTION,
            use_dora=USE_DORA,
        )

    # PEFT
    model = get_peft_model(model, lora_config)
    print("Number of trainable parameters:")
    model.print_trainable_parameters()

    # Gradient checkpointing
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    if hasattr(model.base_model, "gradient_checkpointing_enable"):
        model.base_model.gradient_checkpointing_enable()
    elif hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # --- TrainingArguments ---
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        do_train=True,
        do_eval=not TRAIN_FULL_DATA,
        eval_strategy="steps" if not TRAIN_FULL_DATA else "no",
        save_strategy="steps",
        eval_steps=EVAL_STEPS if not TRAIN_FULL_DATA else None,
        save_steps=SAVE_STEPS,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=LOGGING_STEPS,
        metric_for_best_model="map@3" if not TRAIN_FULL_DATA else None,
        greater_is_better=True if not TRAIN_FULL_DATA else None,
        load_best_model_at_end=not TRAIN_FULL_DATA,
        report_to="wandb" if USE_WANDB else "none",
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        remove_unused_columns=False,
        lr_scheduler_type="cosine",
        warmup_ratio=0.0,
        label_smoothing_factor=LABEL_SMOOTHING_FACTOR,
        save_total_limit=2,
        max_grad_norm=MAX_GRAD_NORM,
        optim="adamw_bnb_8bit" if USE_8BIT_ADAM else "adamw_torch",
    )

    print("Setting up trainer...")

    # Training Statistics
    if TRAIN_FULL_DATA:
        steps_per_epoch = len(train_ds) // (
            TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
        )
        total_steps = steps_per_epoch * EPOCHS
        print(f"\nFull dataset training statistics:")
        print(f"Training samples: {len(train_ds)}")
        print(
            f"Batch size: {TRAIN_BATCH_SIZE} (with gradient accumulation: {TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})"
        )
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Total training steps: {total_steps}")
        print(f"Save interval: every {SAVE_STEPS} steps")
    else:
        steps_per_epoch = len(train_ds) // (
            TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
        )  # gradient_accumulation_stepsを考慮
        total_steps = steps_per_epoch * EPOCHS
        print(f"\nDataset statistics:")
        print(f"Training samples: {len(train_ds)}")
        print(f"Validation samples: {len(val_ds)}")
        print(
            f"Batch size: {TRAIN_BATCH_SIZE} (with gradient accumulation: {TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})"
        )
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Total training steps: {total_steps}")
        print(
            f"Evaluation interval: every {EVAL_STEPS} steps (~{EVAL_STEPS / steps_per_epoch:.2f} epochs)"
        )
        print(
            f"Early stopping after {EARLY_STOPPING_PATIENCE} evaluations without improvement"
        )

    # Data Collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Callbacks
    callbacks = []

    # Add a simple model checkpoint saver for full data training
    if TRAIN_FULL_DATA:
        from transformers import TrainerCallback

        class SaveFinalModelCallback(TrainerCallback):
            def __init__(self, output_dir, tokenizer):
                self.output_dir = output_dir
                self.tokenizer = tokenizer

            def on_train_end(self, args, state, control, model=None, **kwargs):
                final_path = os.path.join(self.output_dir, "final_model")
                os.makedirs(final_path, exist_ok=True)
                model.save_pretrained(final_path)
                self.tokenizer.save_pretrained(final_path)
                print(f"\nTraining completed. Final model saved to: {final_path}")

        callbacks.append(SaveFinalModelCallback(OUTPUT_DIR, tokenizer))

        compute_metrics_full = None

    else:
        save_best_callback = SaveBestMap3Callback(
            save_dir=OUTPUT_DIR, tokenizer=tokenizer
        )
        callbacks.append(save_best_callback)
        print(
            f"SaveBestMap3Callback enabled - モデルは {OUTPUT_DIR}/best_map3 に保存されます"
        )

        if USE_EARLY_STOPPING:
            early_stopping_callback = EarlyStoppingCallback(
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
                early_stopping_threshold=EARLY_STOPPING_THRESHOLD,
            )
            callbacks.append(early_stopping_callback)
            print(f"Early stopping enabled:")
            print(
                f"  - Patience (evaluations without improvement): {EARLY_STOPPING_PATIENCE}"
            )
            print(f"  - Threshold: {EARLY_STOPPING_THRESHOLD}")

        compute_metrics_full = compute_map3

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_full,
        callbacks=callbacks,
    )

    print("Starting training...")
    trainer.train()

    if not TRAIN_FULL_DATA:
        print("\nEvaluating on validation set...")
        eval_results = trainer.evaluate()
        print(f"\nValidation MAP@3: {eval_results.get('eval_map@3', 'N/A'):.4f}")

    print("\nSaving model...")
    model.save_pretrained(BEST_MODEL_PATH)
    tokenizer.save_pretrained(BEST_MODEL_PATH)

    print("Training completed successfully!")
    print(f"Model saved to: {BEST_MODEL_PATH}")
    print(f"Label encoder saved to: {LABEL_ENCODER_PATH}")

    if USE_WANDB:
        wandb.finish()


if __name__ == "__main__":

    main()
