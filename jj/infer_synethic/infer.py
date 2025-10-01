import os, argparse, glob
import pandas as pd
import numpy as np
import torch
import joblib
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import Dataset
from peft import PeftModel

import warnings

warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, required=True, help="Saved Model path or name"
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Model base path or name"
    )
    parser.add_argument("--fold_idx", type=int, required=True, help="Fold Idx")
    parser.add_argument(
        "--test_path", type=str, default="./all_synthetic_data_large_v3.parquet"
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default="../../../input/map-charting-student-math-misunderstandings/train.csv",
    )
    parser.add_argument(
        "--label_encoder_path", type=str, default="./label_encoder.joblib"
    )
    parser.add_argument("--output", type=str, default="submission.csv")
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--topk_pred", type=int, default=3)
    parser.add_argument("--batchsize", type=int, default=128)
    parser.add_argument("--use_bitsandbytes", action="store_true")

    args = parser.parse_args()

    MODEL_PATH = args.model_path
    MODEL_NAME = args.model_name
    FOLD_IDX = args.fold_idx
    TEST_PATH = args.test_path
    TRAIN_PATH = args.train_path
    LABEL_ENCODER_PATH = args.label_encoder_path
    OUTPUT_CSV = f"fold_{FOLD_IDX}_sub.csv"
    MAX_LEN = args.max_len
    BATCHSIZE = args.batchsize
    TOPK_PRED = 37

    # --- Format Input ---
    def format_input(row):
        x = "Yes"
        if not row["is_correct"]:
            x = "No"
        # maybe prompt: Check for any misconception student might have
        return (
            f"Question: {row['QuestionText']}\n"
            f"Student Answer: {row['MC_Answer']}\n"
            f"Correct? {x}\n"
            f"Student Explanation: {row['StudentExplanation']}\n"
        )

    # --- Prepare Dataset ---
    def prepare_dataset(df, tokenizer, cols=["text", "label"]):
        df = df[cols].copy().reset_index(drop=True)
        df["label"] = df["label"].astype(np.int64)
        ds = Dataset.from_pandas(df, preserve_index=False)
        ds = ds.map(
            lambda batch: tokenizer(
                batch["text"], padding="max_length", truncation=True, max_length=MAX_LEN
            ),
            batched=True,
            remove_columns=["text"],
        )
        return ds

    # --- Compute MAP@3 ---
    def compute_map3(eval_pred):
        logits, labels = eval_pred
        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
        top3 = np.argsort(-probs, axis=1)[:, :3]
        match = top3 == labels[:, None]
        map3 = np.mean(
            [1 if m[0] else 0.5 if m[1] else 1 / 3 if m[2] else 0 for m in match]
        )
        return {"map@3": map3}

    le = joblib.load(LABEL_ENCODER_PATH)
    n_classes = len(le.classes_)
    print(f"There are {n_classes} classes.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # --- Inference on Test Data ---
    print("\n--- Starting Inference on Test Data ---")
    test = pd.read_parquet(TEST_PATH)

    def get_correct(csv_path=TRAIN_PATH):
        train = pd.read_csv(csv_path)
        train["QuestionId"] = train["QuestionId"].astype(str)
        idx = train["Category"].str.startswith("True")
        correct = (
            train[idx]
            .groupby(["QuestionId", "MC_Answer"])
            .size()
            .reset_index(name="c")
            .sort_values("c", ascending=False)
            .drop_duplicates(["QuestionId"])
            .assign(is_correct=1)[["QuestionId", "MC_Answer", "is_correct"]]
        )

        return correct

    correct = get_correct()

    test = test.merge(correct, on=["QuestionId", "MC_Answer"], how="left")
    test["is_correct"] = test["is_correct"].fillna(0)
    test["text"] = test.apply(format_input, axis=1)

    test_clean = test[["text"]].copy().reset_index(drop=True)
    ds_test = Dataset.from_pandas(test_clean, preserve_index=False)
    ds_test = ds_test.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=MAX_LEN),
        batched=True,
        remove_columns=["text"],
    )

    print(f"Loading model from {MODEL_PATH}...")

    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=n_classes,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=config,
        low_cpu_mem_usage=True,
    )

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    model = PeftModel.from_pretrained(model, MODEL_PATH)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=f"./tmp/tmp_infer_fold_{FOLD_IDX}",
            per_device_train_batch_size=1,
            per_device_eval_batch_size=BATCHSIZE,
            report_to="none",
            bf16=True,
            fp16=False,
        ),
        data_collator=data_collator,
    )

    preds = trainer.predict(ds_test)

    question_label_choices = {
        "31772": [
            "Correct:NA",
            "Neither:NA",
            "Misconception:Incomplete",
            "Misconception:WNB",
        ],
        "31774": [
            "Correct:NA",
            "Neither:NA",
            "Misconception:SwapDividend",
            "Misconception:Mult",
            "Misconception:FlipChange",
        ],
        "31777": [
            "Correct:NA",
            "Neither:NA",
            "Misconception:Incomplete",
            "Misconception:Irrelevant",
            "Misconception:Wrong_Fraction",
        ],
        "31778": [
            "Correct:NA",
            "Neither:NA",
            "Misconception:Additive",
            "Misconception:Irrelevant",
            "Misconception:WNB",
        ],
        "32829": [
            "Correct:NA",
            "Neither:NA",
            "Misconception:Not_variable",
            "Misconception:Adding_terms",
            "Misconception:Inverse_operation",
        ],
        "32833": [
            "Correct:NA",
            "Neither:NA",
            "Misconception:Inversion",
            "Misconception:Duplication",
            "Misconception:Wrong_Operation",
        ],
        "32835": [
            "Correct:NA",
            "Neither:NA",
            "Misconception:Whole_numbers_larger",
            "Misconception:Longer_is_bigger",
            "Misconception:Ignores_zeroes",
            "Misconception:Shorter_is_bigger",
        ],
        "33471": [
            "Correct:NA",
            "Neither:NA",
            "Misconception:Wrong_fraction",
            "Misconception:Incomplete",
        ],
        "33472": [
            "Correct:NA",
            "Neither:NA",
            "Misconception:Adding_across",
            "Misconception:Denominator-only_change",
            "Misconception:Incorrect_equivalent_fraction_addition",
        ],
        "33474": [
            "Correct:NA",
            "Neither:NA",
            "Misconception:Division",
            "Misconception:Subtraction",
        ],
        "76870": [
            "Correct:NA",
            "Neither:NA",
            "Misconception:Unknowable",
            "Misconception:Definition",
            "Misconception:Interior",
        ],
        "89443": [
            "Correct:NA",
            "Neither:NA",
            "Misconception:Positive",
            "Misconception:Tacking",
        ],
        "91695": [
            "Correct:NA",
            "Neither:NA",
            "Misconception:Wrong_term",
            "Misconception:Firstterm",
        ],
        "104665": [
            "Correct:NA",
            "Neither:NA",
            "Misconception:Base_rate",
            "Misconception:Multiplying_by_4",
        ],
        "109465": [
            "Correct:NA",
            "Neither:NA",
            "Misconception:Certainty",
            "Misconception:Scale",
        ],
    }

    # Normalize for Label Encoder
    question_label_choice_ids = {}
    for qid, choices in question_label_choices.items():
        _label_ids = np.where(np.isin(le.classes_, question_label_choices[qid]))[0]

        question_label_choice_ids[qid] = [int(x) for x in _label_ids]

    test_probabilities = []
    test_predictions = []
    test_top3_predictions = []

    for qid, correct, row in zip(
        test.QuestionId.tolist(), test.is_correct.tolist(), preds.predictions
    ):
        candidate_idx = question_label_choice_ids[qid]

        candidate_logits = row[candidate_idx]

        candidate_probs = torch.nn.functional.softmax(
            torch.tensor(candidate_logits), dim=-1
        ).numpy()

        top_k = np.argsort(-candidate_probs)

        # Have to convert back to the original label encoder space
        topk_idx = np.array(candidate_idx)[top_k]

        # Keep the probabilities
        topk_probs = candidate_probs[top_k].tolist()

        # Get the predicted labels
        topk_preds = le.inverse_transform(topk_idx).tolist()

        correct_values = "True_" if correct else "False_"

        topk_preds = [correct_values + x for x in topk_preds]

        test_probabilities.append(topk_probs)
        test_predictions.append(topk_preds)
        test_top3_predictions.append(" ".join(topk_preds[:3]))

    test_submission_data = pd.DataFrame(
        {
            "row_id": test.row_id.tolist(),
            "ClusterId": test.ClusterId.tolist(),
            "QuestionId": test.QuestionId.tolist(),
            "MC_Answer": test.MC_Answer.tolist(),
            "StudentExplanation": test.StudentExplanation.tolist(),
            "Category": test.Category.tolist(),
            "Misconception": test.Misconception.tolist(),
            "is_correct": test.is_correct.tolist(),
            "probs": test_probabilities,
            "preds": test_predictions,
            "Category:Misconception": test_top3_predictions,
        }
    )

    test_submission_data.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSubmission saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
