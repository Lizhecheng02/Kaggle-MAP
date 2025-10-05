import os
import gc
import pickle
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import Dataset
import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModel
import wandb
from transformers import EarlyStoppingCallback

# Import Custom Modules
from config import Config
from metrics import compute_map_k, calculate_aggregate_map_k
from models import LLMForSequenceClassification
from utils import (
    prepare_correct_answers,
    tokenize_dataset,
    SaveBestMap3Callback,
    convert_numpy_to_list,
)
from prompts import prompt_registry

import warnings

warnings.filterwarnings("ignore")


def train_single_fold(cfg, fold_id, train_df, val_df, tokenizer, n_classes):
    """
    Train a single fold and return validation predictions and labels
    """
    print(f"\n{'=' * 50}")
    print(f"Training Fold {fold_id}")
    print(f"{'=' * 50}")

    # Create fold-specific output directory
    fold_output_dir = f"{cfg.OUTPUT_DIR}/fold_{fold_id}"
    os.makedirs(fold_output_dir, exist_ok=True)

    # Prepare datasets
    COLS = ["text", "label"]
    train_ds = Dataset.from_pandas(train_df[COLS])
    val_ds = Dataset.from_pandas(val_df[COLS])

    # Tokenize datasets
    print("Tokenizing datasets...")
    train_ds = tokenize_dataset(train_ds, tokenizer, cfg.MAX_LEN)
    val_ds = tokenize_dataset(val_ds, tokenizer, cfg.MAX_LEN)

    # Initialize model
    print("Initializing model...")
    torch.cuda.empty_cache()
    gc.collect()

    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.MODEL_NAME,
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
            r=cfg.LORA_RANK,
            lora_alpha=cfg.LORA_ALPHA,
            target_modules=cfg.LORA_TARGET_MODULES,
            lora_dropout=cfg.LORA_DROPOUT,
            bias=cfg.LORA_BIAS,
            task_type=TaskType.SEQ_CLS,
            use_dora=cfg.USE_DORA,
        )

    except Exception:
        print("Using custom classification head for Phi-4...")
        base_model = AutoModel.from_pretrained(
            cfg.MODEL_NAME,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
        )

        model = LLMForSequenceClassification(base_model, n_classes)

        lora_config = LoraConfig(
            r=cfg.LORA_RANK,
            lora_alpha=cfg.LORA_ALPHA,
            target_modules=cfg.LORA_TARGET_MODULES,
            lora_dropout=cfg.LORA_DROPOUT,
            bias=cfg.LORA_BIAS,
            task_type=TaskType.FEATURE_EXTRACTION,
            use_dora=cfg.USE_DORA,
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
        eval_steps=cfg.EVAL_STEPS,
        save_steps=cfg.SAVE_STEPS,
        num_train_epochs=cfg.EPOCHS,
        per_device_train_batch_size=cfg.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=cfg.EVAL_BATCH_SIZE,
        learning_rate=cfg.LEARNING_RATE,
        logging_dir=f"{fold_output_dir}/logs",
        logging_steps=cfg.LOGGING_STEPS,
        metric_for_best_model="map@3",
        greater_is_better=True,
        load_best_model_at_end=True,
        report_to="wandb" if cfg.USE_WANDB else "none",
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        gradient_accumulation_steps=cfg.GRADIENT_ACCUMULATION_STEPS,
        remove_unused_columns=False,
        lr_scheduler_type="cosine",
        warmup_ratio=0.0,
        label_smoothing_factor=cfg.LABEL_SMOOTHING_FACTOR,
        save_total_limit=2,
        max_grad_norm=cfg.MAX_GRAD_NORM,
        optim="adamw_bnb_8bit" if cfg.USE_8BIT_ADAM else "adamw_torch",
    )

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Callbacks
    callbacks = []
    save_best_callback = SaveBestMap3Callback(
        save_dir=fold_output_dir, tokenizer=tokenizer
    )
    callbacks.append(save_best_callback)

    if cfg.USE_EARLY_STOPPING:
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=cfg.EARLY_STOPPING_PATIENCE,
            early_stopping_threshold=cfg.EARLY_STOPPING_THRESHOLD,
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
        compute_metrics=lambda eval_pred: compute_map_k(eval_pred, k=3),
        callbacks=callbacks,
    )

    # Training statistics
    steps_per_epoch = len(train_ds) // (
        cfg.TRAIN_BATCH_SIZE * cfg.GRADIENT_ACCUMULATION_STEPS
    )
    total_steps = steps_per_epoch * cfg.EPOCHS
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

    # Get both logits and labels
    val_logits = predictions.predictions
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
        "logits": val_logits,
        "labels": val_labels,
        "val_indices": val_df.index.tolist(),
    }


def main():
    cfg = Config()

    # GPU setup
    if cfg.CUDA_VISIBLE_DEVICES is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CUDA_VISIBLE_DEVICES
        print(f"Using CUDA device(s): {cfg.CUDA_VISIBLE_DEVICES}")

    # Clear memory cache
    torch.cuda.empty_cache()
    gc.collect()

    # Create output directory
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Load and preprocess data
    print("Loading and preprocessing training data...")
    train = pd.read_parquet(cfg.TRAIN_DATA_PATH)

    synth = pd.read_parquet(cfg.SYNTH_DATA_PATH)
    synth['fold'] = -1
    del synth['is_correct']

    #synth = []
    #for cluster_id, s in df.groupby("ClusterId"):
    #
    #    #t = g[g['fold']==0].copy()
    #    #s = g[g['fold']==1].copy()
    #
    #    if not s.empty:
    #
    #        #if s.shape[0] > 100:
    #        #    s = s.sample(n=100, random_state=42, replace=True)
    #
    #        s = s.drop_duplicates(subset=['QuestionId', 'QuestionText', 'MC_Answer', 'Category', 'Misconception', 'StudentExplanation']).reset_index(drop=True)
    #        synth += s.to_dict(orient='records')
    #
    #    #synth += t.to_dict(orient='records')
    #
    #synth = pd.DataFrame(synth)
    #synth = synth.drop_duplicates(subset=['QuestionId', 'QuestionText', 'MC_Answer', 'Category', 'Misconception', 'StudentExplanation']).reset_index(drop=True)
    
    train['fold'] = 0
    train = pd.concat([
        train,
        synth
    ])
    train = train.drop_duplicates(subset=['QuestionId', 'QuestionText', 'MC_Answer', 'Category', 'Misconception', 'StudentExplanation']).reset_index(drop=True)

    #train = synth
    #synth['fold'] = -1

    #synth = pd.read_parquet(cfg.SYNTH_DATA_PATH)
    #synth = synth[synth['Category'].str.contains("Neither")]
    #
    #train = pd.concat([
    #    train,
    #    synth
    #])
    #train = train.drop_duplicates(subset=['QuestionId', 'QuestionText', 'MC_Answer', 'Category', 'Misconception', 'StudentExplanation'], keep='first').reset_index(drop=True)

    if cfg.DEBUG:
        train = train.sample(n=10, random_state=cfg.RANDOM_SEED)

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
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME, trust_remote_code=True)

    # Handle Phi-4 tokenizer
    if "microsoft/phi-4" in cfg.MODEL_NAME.lower():
        if tokenizer.pad_token is None:
            tokenizer.pad_token = "<|finetune_right_pad_id|>"
            tokenizer.pad_token_id = 100257

    print("Formatting input text...")
    # Get the prompt creation function
    prompt_function = prompt_registry.get(cfg.PROMPT_VERSION, None)

    # Create the prompt using the function
    texts = []
    for _, row in train.iterrows():
        t = prompt_function(tokenizer=tokenizer, row=row)
        texts.append(t)
    train["text"] = texts

    print("Example prompt:")
    print(train.text.values[0])

    # Analyze token lengths
    print("Analyzing token lengths...")
    lengths = [len(tokenizer.encode(t, truncation=False)) for t in train["text"]]
    over_limit = (np.array(lengths) > cfg.MAX_LEN).sum()
    print(f"There are {over_limit} sample(s) with more than {cfg.MAX_LEN} tokens")

    # Save label encoder
    print(f"Saving label encoder to: {cfg.LABEL_ENCODER_PATH}")
    joblib.dump(le, cfg.LABEL_ENCODER_PATH)

    # Cross-validation training
    all_results = []
    all_predictions = []
    all_labels = []
    fold_scores = []

    # Add new collections for out-of-fold logits (organized by original indices)
    oof_logits = np.zeros((len(train), n_classes))
    oof_labels = np.zeros(len(train), dtype=int)
    oof_predictions_mask = np.zeros(len(train), dtype=bool)

    print(f"\nStarting {cfg.FOLDS}-fold cross-validation training...")

    for fold_id in range(cfg.FOLDS):
        # Initialize fold-specific wandb run
        fold_run = None
        if cfg.USE_WANDB:
            fold_run = wandb.init(
                project=cfg.WANDB_PROJECT,
                name=f"{cfg.WANDB_RUN_NAME}_fold_{fold_id}",
                entity=cfg.WANDB_ENTITY,
                group=f"{cfg.WANDB_RUN_NAME}_cv",
                job_type=f"fold_{fold_id}",
                config={
                    "fold_id": fold_id,
                    "model_name": cfg.MODEL_NAME,
                    "seed": cfg.RANDOM_SEED,
                    "epochs": cfg.EPOCHS,
                    "max_len": cfg.MAX_LEN,
                    "train_batch_size": cfg.TRAIN_BATCH_SIZE,
                    "eval_batch_size": cfg.EVAL_BATCH_SIZE,
                    "learning_rate": cfg.LEARNING_RATE,
                    "lora_rank": cfg.LORA_RANK,
                    "lora_alpha": cfg.LORA_ALPHA,
                },
            )

        # Split data by fold
        val_fold_data = train[train["fold"] == fold_id].copy()
        train_fold_data = train[train["fold"] != fold_id].copy()

        print(
            f"\nFold {fold_id}: Train size = {len(train_fold_data)}, Val size = {len(val_fold_data)}"
        )

        # Train single fold
        fold_results = train_single_fold(
            cfg=cfg,
            fold_id=fold_id,
            train_df=train_fold_data,
            val_df=val_fold_data,
            tokenizer=tokenizer,
            n_classes=n_classes,
        )

        # Close fold-specific wandb run
        if cfg.USE_WANDB and fold_run:
            fold_run.finish()

        # Store results
        all_results.append(fold_results)
        all_predictions.extend(fold_results["predictions"])
        all_labels.extend(fold_results["labels"])
        fold_scores.append(fold_results["map3_score"])

        # Store out-of-fold logits organized by original indices
        val_indices = fold_results["val_indices"]
        fold_logits = fold_results["logits"]
        fold_labels = fold_results["labels"]

        # Map back to original dataframe indices
        for i, orig_idx in enumerate(val_indices):
            train_idx = train.index.get_loc(orig_idx)
            oof_logits[train_idx] = fold_logits[i]
            oof_labels[train_idx] = fold_labels[i]
            oof_predictions_mask[train_idx] = True

    # Verify all samples have predictions
    if not np.all(oof_predictions_mask):
        missing_count = np.sum(~oof_predictions_mask)
        print(f"Warning: {missing_count} samples are missing out-of-fold predictions!")

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
    aggregate_map3 = calculate_aggregate_map_k(all_predictions, all_labels, k=3)

    print(f"\nMean CV MAP@3: {mean_cv_score:.4f} (+/- {std_cv_score:.4f})")
    print(f"Aggregate MAP@3: {aggregate_map3:.4f}")

    # Initialize main wandb run for final summary logging
    if cfg.USE_WANDB:
        main_run = wandb.init(
            project=cfg.WANDB_PROJECT,
            name=f"{cfg.WANDB_RUN_NAME}_cv_summary",
            entity=cfg.WANDB_ENTITY,
            group=f"{cfg.WANDB_RUN_NAME}_cv",
            job_type="cv_summary",
            config={
                "model_name": cfg.MODEL_NAME,
                "seed": cfg.RANDOM_SEED,
                "epochs": cfg.EPOCHS,
                "max_len": cfg.MAX_LEN,
                "train_batch_size": cfg.TRAIN_BATCH_SIZE,
                "eval_batch_size": cfg.EVAL_BATCH_SIZE,
                "learning_rate": cfg.LEARNING_RATE,
                "folds": cfg.FOLDS,
                "early_stopping_patience": cfg.EARLY_STOPPING_PATIENCE
                if cfg.USE_EARLY_STOPPING
                else None,
                "lora_rank": cfg.LORA_RANK,
                "lora_alpha": cfg.LORA_ALPHA,
                "lora_target_modules": cfg.LORA_TARGET_MODULES,
                "lora_dropout": cfg.LORA_DROPOUT,
                "lora_bias": cfg.LORA_BIAS,
            },
        )

        # Log all fold scores at once
        for i, score in enumerate(fold_scores):
            main_run.log(
                {
                    "cv_fold_scores": score,
                    "fold_id": i,
                },
                step=i,
            )

        # Log final aggregate results
        main_run.log(
            {
                "cv_mean_map3": mean_cv_score,
                "cv_std_map3": std_cv_score,
                "cv_aggregate_map3": aggregate_map3,
            }
        )

        main_run.finish()

    # Save out-of-fold logits as numpy files
    print("Saving out-of-fold logits...")

    # Save comprehensive out-of-fold data
    oof_data = {
        "logits": oof_logits,
        "labels": oof_labels,
        "probabilities": torch.softmax(torch.tensor(oof_logits), dim=-1).numpy(),
        "class_names": le.classes_,
        "sample_indices": train.index.values,
        "predictions_mask": oof_predictions_mask,
    }

    np.save(f"{cfg.OUTPUT_DIR}/oof_predictions.npy", oof_data)
    print(f"Out-of-fold predictions saved to: {cfg.OUTPUT_DIR}/oof_predictions.npy")

    # Save all results
    results_summary = {
        "fold_scores": fold_scores,
        "mean_cv_score": mean_cv_score,
        "std_cv_score": std_cv_score,
        "aggregate_map3": aggregate_map3,
        "all_predictions": convert_numpy_to_list(all_predictions),
        "all_labels": convert_numpy_to_list(all_labels),
        "detailed_results": convert_numpy_to_list(all_results),
        "oof_logits_info": {
            "shape": oof_logits.shape,
            "n_classes": n_classes,
            "class_names": le.classes_.tolist(),
            "all_samples_predicted": np.all(oof_predictions_mask).item(),
        },
    }

    # Also save as pickle for easier loading in Python
    pickle_path = f"{cfg.OUTPUT_DIR}/cv_results.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(results_summary, f)
    print(f"Results also saved as pickle to: {pickle_path}")

    # Save summary to CSV for easy viewing
    summary_df = pd.DataFrame(
        {
            "fold": range(cfg.FOLDS),
            "map3_score": fold_scores,
            "val_samples": [len(result["labels"]) for result in all_results],
        }
    )
    summary_df.to_csv(f"{cfg.OUTPUT_DIR}/cv_summary.csv", index=False)
    print(f"Summary saved to: {cfg.OUTPUT_DIR}/cv_summary.csv")

    print("\nCross-validation training completed successfully!")
    print(f"Results saved in: {cfg.OUTPUT_DIR}")
    print(f"Out-of-fold logits shape: {oof_logits.shape}")
    print(f"All samples have predictions: {np.all(oof_predictions_mask)}")

    if cfg.USE_WANDB and main_run:
        main_run.finish()


if __name__ == "__main__":
    main()
