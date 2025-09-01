import os
import json
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
import pickle

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


def calculate_aggregate_map3(all_predictions, all_labels, k=3):
    """
    Calculate MAP@3 for all predictions combined
    """
    if len(all_predictions) == 0:
        return 0.0

    # Convert to numpy arrays
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)

    # Get top k predictions for each sample
    top_k_preds = np.argsort(-predictions, axis=1)[:, :k]

    # Calculate MAP@3
    map_scores = []
    for i, (pred_indices, true_label) in enumerate(zip(top_k_preds, labels)):
        # Check if true label is in top k predictions
        if true_label in pred_indices:
            # Find position of true label (1-indexed)
            position = np.where(pred_indices == true_label)[0][0] + 1
            map_scores.append(1.0 / position)
        else:
            map_scores.append(0.0)

    return np.mean(map_scores)


def train_single_fold(fold_id, train_df, val_df, tokenizer, n_classes):
    """
    Train a single fold and return validation predictions and labels
    """
    print(f"\n{'=' * 50}")
    print(f"Training Fold {fold_id}")
    print(f"{'=' * 50}")

    # Create fold-specific output directory
    fold_output_dir = f"{OUTPUT_DIR}/fold_{fold_id}"
    os.makedirs(fold_output_dir, exist_ok=True)

    # Prepare datasets
    COLS = ["text", "label"]
    train_ds = Dataset.from_pandas(train_df[COLS])
    val_ds = Dataset.from_pandas(val_df[COLS])

    # Tokenize datasets
    print("Tokenizing datasets...")
    train_ds = tokenize_dataset(train_ds, tokenizer, MAX_LEN)
    val_ds = tokenize_dataset(val_ds, tokenizer, MAX_LEN)

    # Initialize model
    print("Initializing model...")
    torch.cuda.empty_cache()
    gc.collect()

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

        # Configure LoRA
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

        lora_config = LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias=LORA_BIAS,
            task_type=TaskType.FEATURE_EXTRACTION,
            use_dora=USE_DORA,
        )

    # Apply PEFT
    model = get_peft_model(model, lora_config)
    print("Number of trainable parameters:")
    model.print_trainable_parameters()

    # Enable gradient checkpointing
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    if hasattr(model.base_model, "gradient_checkpointing_enable"):
        model.base_model.gradient_checkpointing_enable()
    elif hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Training arguments
    training_args = TrainingArguments(
        output_dir=fold_output_dir,
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
        logging_dir=f"{fold_output_dir}/logs",
        logging_steps=LOGGING_STEPS,
        metric_for_best_model="map@3",
        greater_is_better=True,
        load_best_model_at_end=True,
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

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Callbacks
    callbacks = []
    save_best_callback = SaveBestMap3Callback(
        save_dir=fold_output_dir, tokenizer=tokenizer
    )
    callbacks.append(save_best_callback)

    if USE_EARLY_STOPPING:
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            early_stopping_threshold=EARLY_STOPPING_THRESHOLD,
        )
        callbacks.append(early_stopping_callback)

    # Initialize trainer
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

    # Training statistics
    steps_per_epoch = len(train_ds) // (TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
    total_steps = steps_per_epoch * EPOCHS
    print(f"\nFold {fold_id} training statistics:")
    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total training steps: {total_steps}")

    # Train the model
    print(f"Starting training for fold {fold_id}...")
    trainer.train()

    # Evaluate and get predictions
    print(f"Evaluating fold {fold_id}...")
    eval_results = trainer.evaluate()
    fold_map3 = eval_results.get("eval_map@3", 0.0)
    print(f"Fold {fold_id} MAP@3: {fold_map3:.4f}")

    # Get predictions for validation set
    print("Getting validation predictions...")
    predictions = trainer.predict(val_ds)
    val_predictions = torch.softmax(
        torch.tensor(predictions.predictions), dim=-1
    ).numpy()
    val_labels = predictions.label_ids

    # Save fold model
    fold_best_path = f"{fold_output_dir}/best_model"
    model.save_pretrained(fold_best_path)
    tokenizer.save_pretrained(fold_best_path)

    # Clean up memory
    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "fold_id": fold_id,
        "map3_score": fold_map3,
        "predictions": val_predictions,
        "labels": val_labels,
        "val_indices": val_df.index.tolist(),
    }


def main():
    # WANDB setup
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
                "folds": FOLDS,
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

    # GPU setup
    if CUDA_VISIBLE_DEVICES is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
        print(f"Using CUDA device(s): {CUDA_VISIBLE_DEVICES}")

    # Clear memory cache
    torch.cuda.empty_cache()
    gc.collect()

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load and preprocess data
    print("Loading and preprocessing training data...")
    train = pd.read_parquet(TRAIN_DATA_PATH)

    # Check if fold column exists
    if "fold" not in train.columns:
        raise ValueError(
            "The training data must contain a 'fold' column for cross-validation"
        )

    train.Misconception = train.Misconception.fillna("NA")
    train["target"] = train.Category + ":" + train.Misconception

    # Label encoding
    le = LabelEncoder()
    train["label"] = le.fit_transform(train["target"])
    n_classes = len(le.classes_)
    print(f"Dataset shape: {train.shape} with {n_classes} target classes")
    print(f"Available folds: {sorted(train['fold'].unique())}")

    # Feature engineering
    print("Performing feature engineering...")
    correct = prepare_correct_answers(train)
    train = train.merge(correct, on=["QuestionId", "MC_Answer"], how="left")
    train.is_correct = train.is_correct.fillna(0)

    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Handle Phi-4 tokenizer
    if "microsoft/phi-4" in MODEL_NAME.lower():
        if tokenizer.pad_token is None:
            tokenizer.pad_token = "<|finetune_right_pad_id|>"
            tokenizer.pad_token_id = 100257

    # Format input text
    print("Formatting input text...")
    texts = []
    for _, row in train.iterrows():
        t = format_input(tokenizer=tokenizer, row=row, think="")
        texts.append(t)
    train["text"] = texts

    print("Example prompt:")
    print(train.text.values[0])

    # Analyze token lengths
    print("Analyzing token lengths...")
    lengths = [len(tokenizer.encode(t, truncation=False)) for t in train["text"]]
    over_limit = (np.array(lengths) > MAX_LEN).sum()
    print(f"There are {over_limit} sample(s) with more than {MAX_LEN} tokens")

    # Save label encoder
    print(f"Saving label encoder to: {LABEL_ENCODER_PATH}")
    joblib.dump(le, LABEL_ENCODER_PATH)

    # Cross-validation training
    all_results = []
    all_predictions = []
    all_labels = []
    fold_scores = []

    print(f"\nStarting {FOLDS}-fold cross-validation training...")

    for fold_id in range(FOLDS):
        # Split data by fold
        val_fold_data = train[train["fold"] == fold_id].copy()
        train_fold_data = train[train["fold"] != fold_id].copy()

        print(
            f"\nFold {fold_id}: Train size = {len(train_fold_data)}, Val size = {len(val_fold_data)}"
        )

        # Train single fold
        fold_results = train_single_fold(
            fold_id=fold_id,
            train_df=train_fold_data,
            val_df=val_fold_data,
            tokenizer=tokenizer,
            n_classes=n_classes,
        )

        # Store results
        all_results.append(fold_results)
        all_predictions.extend(fold_results["predictions"])
        all_labels.extend(fold_results["labels"])
        fold_scores.append(fold_results["map3_score"])

        # Log fold results to wandb
        if USE_WANDB:
            wandb.log(
                {
                    f"fold_{fold_id}/map3": fold_results["map3_score"],
                    f"fold_{fold_id}/val_size": len(val_fold_data),
                    f"fold_{fold_id}/train_size": len(train_fold_data),
                }
            )

    # Calculate aggregate metrics
    print(f"\n{'=' * 50}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'=' * 50}")

    # Individual fold results
    for i, score in enumerate(fold_scores):
        print(f"Fold {i} MAP@3: {score:.4f}")

    # Overall results
    mean_cv_score = np.mean(fold_scores)
    std_cv_score = np.std(fold_scores)
    aggregate_map3 = calculate_aggregate_map3(all_predictions, all_labels)

    print(f"\nMean CV MAP@3: {mean_cv_score:.4f} (+/- {std_cv_score:.4f})")
    print(f"Aggregate MAP@3: {aggregate_map3:.4f}")

    # Log final results to wandb
    if USE_WANDB:
        wandb.log(
            {
                "cv/mean_map3": mean_cv_score,
                "cv/std_map3": std_cv_score,
                "cv/aggregate_map3": aggregate_map3,
            }
        )

    # Save all results
    results_summary = {
        "fold_scores": fold_scores,
        "mean_cv_score": mean_cv_score,
        "std_cv_score": std_cv_score,
        "aggregate_map3": aggregate_map3,
        "all_predictions": all_predictions,
        "all_labels": all_labels,
        "detailed_results": all_results,
    }

    results_path = f"{OUTPUT_DIR}/cv_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=4)
    print(f"\nResults saved to: {results_path}")

    # Save summary to CSV for easy viewing
    summary_df = pd.DataFrame(
        {
            "fold": range(FOLDS),
            "map3_score": fold_scores,
            "val_samples": [len(result["labels"]) for result in all_results],
        }
    )
    summary_df.to_csv(f"{OUTPUT_DIR}/cv_summary.csv", index=False)
    print(f"Summary saved to: {OUTPUT_DIR}/cv_summary.csv")

    print(f"\nCross-validation training completed successfully!")
    print(f"Results saved in: {OUTPUT_DIR}")

    if USE_WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()
