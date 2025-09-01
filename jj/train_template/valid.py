import os
import gc

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from transformers import DataCollatorWithPadding, BitsAndBytesConfig
from datasets import Dataset
from peft import PeftModel
import joblib

from config import *
from utils import format_input, prepare_correct_answers

import warnings

warnings.filterwarnings("ignore")


def tokenize_function(examples, tokenizer, max_len):
    return tokenizer(
        examples["text"],
        padding=False,
        truncation=True,
        max_length=max_len,
        return_tensors=None,
    )


def load_model_and_tokenizer():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Only for Phi-4
    if "microsoft/phi-4" in MODEL_NAME.lower():
        if tokenizer.pad_token is None:
            tokenizer.pad_token = "<|finetune_right_pad_id|>"
            tokenizer.pad_token_id = 100257

    print("Loading label encoder...")
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    n_classes = len(label_encoder.classes_)

    print(f"Loading model with {n_classes} classes...")

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.float16,
        bnb_8bit_use_double_quant=True,
        bnb_8bit_quant_type="nf4",
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=n_classes,
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, BEST_MODEL_PATH)

    if hasattr(model, "base_model"):
        model.base_model.config.pad_token_id = tokenizer.pad_token_id
        if hasattr(model.base_model, "model"):
            model.base_model.model.config.pad_token_id = tokenizer.pad_token_id
    else:
        model.config.pad_token_id = tokenizer.pad_token_id

    model.eval()

    if torch.cuda.is_available():
        print("Using GPU for inference")
    else:
        print("Using CPU for inference")

    return model, tokenizer, label_encoder


def prepare_data():
    print("Loading training data...")
    train_df = pd.read_csv(INFERENCE_DATA_PATH)

    print("Performing feature engineering...")
    correct = prepare_correct_answers(train_df)
    train_df = train_df.merge(correct, on=["QuestionId", "MC_Answer"], how="left")
    train_df.is_correct = train_df.is_correct.fillna(0)

    train_df["text"] = train_df.apply(format_input, axis=1)

    dataset = Dataset.from_pandas(train_df[["text"]])

    return train_df, dataset


def inference_with_batches(model, tokenizer, dataset, batch_size=16):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, MAX_LEN),
        batched=True,
        remove_columns=dataset.column_names,
    )

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        data_collator=data_collator,
        args=TrainingArguments(
            output_dir="./tmp_valid",
            per_device_eval_batch_size=batch_size,
            fp16=True,
            dataloader_num_workers=2,
            remove_unused_columns=True,
        ),
    )

    print(f"Running inference with batch size {batch_size}...")
    predictions = trainer.predict(tokenized_dataset)

    logits = predictions.predictions
    probabilities = torch.nn.functional.softmax(
        torch.from_numpy(logits), dim=-1
    ).numpy()

    return probabilities


def main():
    torch.cuda.empty_cache()
    gc.collect()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model, tokenizer, label_encoder = load_model_and_tokenizer()

    train_df, dataset = prepare_data()

    batch_size = EVAL_BATCH_SIZE
    if torch.cuda.is_available():
        batch_size = 32
    try:
        with torch.no_grad():
            probabilities = inference_with_batches(
                model, tokenizer, dataset, batch_size
            )
    except torch.cuda.OutOfMemoryError:
        print("CUDA out of memory. Reducing batch size...")
        torch.cuda.empty_cache()
        batch_size = max(1, batch_size // 2)
        with torch.no_grad():
            probabilities = inference_with_batches(
                model, tokenizer, dataset, batch_size
            )

    print(f"\nInference completed!")
    print(f"Shape of probabilities: {probabilities.shape}")
    print(f"Expected shape: ({len(train_df)}, {len(label_encoder.classes_)})")

    output_path = os.path.join(OUTPUT_DIR, "train_probabilities.npy")
    np.save(output_path, probabilities)
    print(f"\nProbabilities saved to: {output_path}")

    print("\nSample predictions (top 3 classes for first 5 rows):")
    for i in range(min(5, len(probabilities))):
        top_indices = np.argsort(probabilities[i])[::-1][:3]
        top_probs = probabilities[i][top_indices]
        top_labels = label_encoder.inverse_transform(top_indices)
        print(f"Row {i}: {list(zip(top_labels, top_probs))}")

    if torch.cuda.is_available():
        print(
            f"\nGPU memory used: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB"
        )

    return probabilities


if __name__ == "__main__":
    probabilities = main()
