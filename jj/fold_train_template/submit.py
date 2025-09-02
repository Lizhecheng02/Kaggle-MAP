import pandas as pd
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import Dataset
import joblib
import torch
import gc

try:
    from peft import PeftModel

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not available, will use base model only")

from config import Config
from utils import (
    prepare_correct_answers,
    format_input,
    tokenize_dataset,
    create_submission,
)


def main():
    torch.cuda.empty_cache()
    gc.collect()

    import os

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    if torch.cuda.device_count() > 1:
        print(f"Found {torch.cuda.device_count()} GPUs")

    print("Loading label encoder...")

    cfg = Config()

    le = joblib.load(cfg.LABEL_ENCODER_PATH)
    n_classes = len(le.classes_)

    print("Loading trained model and tokenizer...")

    if PEFT_AVAILABLE:
        print(f"Loading fine-tuned LoRA model from: {cfg.BEST_MODEL_PATH}")
        print(f"Loading base model from: {cfg.MODEL_NAME}")

        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_quant_type="nf4",
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.MODEL_NAME,
            num_labels=n_classes,
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        model = PeftModel.from_pretrained(model, cfg.BEST_MODEL_PATH)

        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.MODEL_NAME, trust_remote_code=True
        )
        print("Successfully loaded LoRA fine-tuned model")
    else:
        raise ImportError(
            "PEFT is required to load the fine-tuned model. Please install peft: pip install peft"
        )

    if "microsoft/phi-4" in cfg.MODEL_NAME.lower():
        if tokenizer.pad_token is None:
            tokenizer.pad_token = "<|finetune_right_pad_id|>"
            tokenizer.pad_token_id = 100257

    if hasattr(model, "base_model"):
        model.base_model.config.pad_token_id = tokenizer.pad_token_id
        if hasattr(model.base_model, "model"):
            model.base_model.model.config.pad_token_id = tokenizer.pad_token_id
    else:
        model.config.pad_token_id = tokenizer.pad_token_id

    print("Loading test data...")
    test = pd.read_csv(cfg.TEST_DATA_PATH)

    print("Loading training data for correct answers...")
    train = pd.read_csv(cfg.TRAIN_DATA_PATH)
    train.Misconception = train.Misconception.fillna("NA")
    correct = prepare_correct_answers(train)

    print("Preprocessing test data...")
    test = test.merge(correct, on=["QuestionId", "MC_Answer"], how="left")
    test.is_correct = test.is_correct.fillna(0)
    test["text"] = test.apply(format_input, axis=1)

    print("Tokenizing test data...")
    ds_test = Dataset.from_pandas(test[["text"]])
    ds_test = tokenize_dataset(ds_test, tokenizer, cfg.MAX_LEN)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print("Running inference...")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        data_collator=data_collator,
        args=TrainingArguments(
            output_dir="./tmp",
            report_to="none",
            per_device_eval_batch_size=cfg.EVAL_BATCH_SIZE,
            fp16=True,
            dataloader_pin_memory=True,
            dataloader_num_workers=2,
        ),
    )

    with torch.no_grad():
        predictions = trainer.predict(ds_test)

    print("Creating submission file...")

    submission = create_submission(predictions, test, le)

    submission.to_csv(cfg.SUBMISSION_OUTPUT_PATH, index=False)
    print(f"Submission file saved to: {cfg.SUBMISSION_OUTPUT_PATH}")
    print("\nSubmission preview:")
    print(submission.head())
    print(f"\nSubmission shape: {submission.shape}")


if __name__ == "__main__":
    main()
