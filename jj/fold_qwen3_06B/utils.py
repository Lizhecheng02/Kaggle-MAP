import os
import pandas as pd
import numpy as np
import torch
from transformers import TrainerCallback


def prepare_correct_answers(train_data):
    idx = train_data.apply(lambda row: row.Category.split("_")[0] == "True", axis=1)
    correct = train_data.loc[idx].copy()
    correct["c"] = correct.groupby(["QuestionId", "MC_Answer"]).MC_Answer.transform(
        "count"
    )
    correct = correct.sort_values("c", ascending=False)
    correct = correct.drop_duplicates(["QuestionId"])[["QuestionId", "MC_Answer"]]
    correct["is_correct"] = 1
    return correct


def tokenize_dataset(dataset, tokenizer, max_len):
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding=False,
            truncation=True,
            max_length=max_len,
            return_tensors=None,
        )

    dataset = dataset.map(tokenize, batched=True, batch_size=100)
    columns = (
        ["input_ids", "attention_mask", "label"]
        if "label" in dataset.column_names
        else ["input_ids", "attention_mask"]
    )
    dataset.set_format(type="torch", columns=columns)
    return dataset


class SaveBestMap3Callback(TrainerCallback):
    def __init__(self, save_dir, tokenizer):
        self.save_dir = save_dir
        self.tokenizer = tokenizer
        self.best_map3 = 0.0

    def on_evaluate(self, args, state, control, metrics, model=None, **kwargs):
        current_map3 = metrics.get("eval_map@3", 0.0)

        if current_map3 > self.best_map3:
            self.best_map3 = current_map3

            best_map3_path = os.path.join(self.save_dir, "best_map3")
            os.makedirs(best_map3_path, exist_ok=True)

            model.save_pretrained(best_map3_path)
            self.tokenizer.save_pretrained(best_map3_path)

            print(
                f"\nNew Best MAP@3 Score: {current_map3:.4f} - Saved to: {best_map3_path}"
            )

        return control


def compute_map3(eval_pred):
    logits, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    top3 = np.argsort(-probs, axis=1)[:, :3]
    score = 0.0
    for i, label in enumerate(labels):
        ranks = top3[i]
        if ranks[0] == label:
            score += 1.0
        elif ranks[1] == label:
            score += 1.0 / 2
        elif ranks[2] == label:
            score += 1.0 / 3
    return {"map@3": score / len(labels)}


# Convert numpy arrays to lists for JSON serialization
def convert_numpy_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, list):
        return [convert_numpy_to_list(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_list(value) for key, value in obj.items()}
    else:
        return obj


def create_submission(predictions, test_data, label_encoder):
    probs = torch.nn.functional.softmax(
        torch.tensor(predictions.predictions), dim=1
    ).numpy()
    top3 = np.argsort(-probs, axis=1)[:, :3]
    flat = top3.flatten()
    decoded = label_encoder.inverse_transform(flat)
    top3_labels = decoded.reshape(top3.shape)
    pred_strings = [" ".join(r) for r in top3_labels]

    submission = pd.DataFrame(
        {"row_id": test_data.row_id.values, "Category:Misconception": pred_strings}
    )
    return submission
