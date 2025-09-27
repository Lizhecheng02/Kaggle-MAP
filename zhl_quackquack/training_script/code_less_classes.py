import os, shutil, warnings
from pathlib import Path
from collections import Counter

import numpy as np
import argparse
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    LogitsProcessor,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
)
from transformers.trainer_callback import TrainerCallback

from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training, TaskType
import joblib
import gc

TRAIN_CSV = "./raw_data/train.csv"
TEST_CSV  = "./raw_data/test.csv"
CLEAN_MISLABEL = "ignore"   # ignore | fix | remove

def clean_mislabel_entries(train: pd.DataFrame) -> pd.DataFrame:
    print(f"Using {CLEAN_MISLABEL} for data cleaning Strat")
    qid = 31778
    correct_answer = r"\( 6 \)"
    rows_to_fix = []
    for idx, row in train[train['QuestionId'] == qid].iterrows():
        is_correct_answer = row['MC_Answer'] == correct_answer
        is_true = str(row['Category']).startswith("True")
        if is_correct_answer and not is_true:
            rows_to_fix.append(idx)
        elif not is_correct_answer and is_true:
            rows_to_fix.append(idx)
    assert len(rows_to_fix) == 18, "Expected 18 mislabeled entries to fix, found a different number."

    if CLEAN_MISLABEL == "ignore":
        return train
    elif CLEAN_MISLABEL == "remove":
        return train.drop(index=rows_to_fix).reset_index(drop=True)
    elif CLEAN_MISLABEL == "fix":
        for idx in rows_to_fix:
            row = train.loc[idx]
            cat = str(row['Category']).split("_", 1)[-1]
            prefix = "True" if row['MC_Answer'] == correct_answer else "False"
            train.at[idx, 'Category'] = f"{prefix}_{cat}"
        return train
    else:
        raise ValueError("CLEAN_MISLABEL must be 'ignore', 'remove', or 'fix'")


def load_and_preprocess_data():
    train = pd.read_csv(TRAIN_CSV)
    train = clean_mislabel_entries(train)
    train['Misconception'] = train['Misconception'].fillna('NA')
    # Store full label for evaluation
    train['full_target'] = train['Category'] + ":" + train['Misconception']
    
    # Train only on suffix (remove True_/False_)
    train['misconception_target'] = (
        train['Category'].str.split("_", n=1).str[-1] + ":" + train['Misconception']
    )

    le = LabelEncoder()
    train['label'] = le.fit_transform(train['misconception_target'])

    if "is_correct" in train.columns:
        train = train.drop(columns = "is_correct")
    
    idx = train['Category'].str.startswith("True")
    correct = (
        train[idx].groupby(['QuestionId','MC_Answer']).size()
        .reset_index(name='c').sort_values('c', ascending=False)
        .drop_duplicates(['QuestionId']).assign(is_correct=1)[['QuestionId','MC_Answer','is_correct']]
    )

    train = train.merge(correct, on=['QuestionId','MC_Answer'], how='left')
    train['is_correct'] = train['is_correct'].fillna(0)

    # suppose you also have a QuestionId -> CorrectAnswerText mapping
    answers = train.loc[train["is_correct"] == 1, ["QuestionId", "MC_Answer"]].rename(
        columns={"MC_Answer": "TrueAnswer"}
    ).drop_duplicates(['QuestionId'], keep="first")
    
    train = train.merge(answers, on="QuestionId", how="left")

    train["split_key"] = (train['QuestionId'].astype(str) + "_" + train['label'].astype(str)).astype('category').cat.codes
    return train, le


def format_input(row):
    x = "Yes" if row['is_correct'] else "No"
    return (
        f"Question: {row['QuestionText']}\n"
        f"Student Answer: {row['MC_Answer']}\n"
        f"Correct? {x}\n"
        f"Student Explanation: {row['StudentExplanation']}\n"
    )

def format_input_v2(row):
    x = "Yes" if row['is_correct'] else "No"
    return (
        f"Question: {row['QuestionText']}\n"
        f"True Answer: {row['TrueAnswer']}\n"
        f"Student Answer: {row['MC_Answer']}\n"
        f"Correct? {x}\n"
        f"Student Explanation: {row['StudentExplanation']}\n"
    )

def prepare_dataset(df, tokenizer, cols=['text', 'label'], MAX_LEN=300):
    df = df[cols].copy().reset_index(drop=True)
    ds = Dataset.from_pandas(df, preserve_index=False)
    ds = ds.map(
        lambda batch: tokenizer(batch['text'], truncation=True, max_length=MAX_LEN),
        batched=True,
        remove_columns=['text']
    )
    
    # Only "input_ids", "attention_mask", "labels" should be tensors
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return ds


def load_base_model_bf16(MODEL_NAME, N_CLASSES):
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=N_CLASSES,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    # model.config.use_cache = False  # better for training/checkpointing
    return model

def load_base_model_nf4(MODEL_NAME, N_CLASSES):
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=N_CLASSES,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=config,
        low_cpu_mem_usage=True,
    )
    
    model = prepare_model_for_kbit_training(model)
    return model

def make_compute_map3(eval_df, le):
    gold_full = eval_df["full_target"].to_numpy()
    is_correct = eval_df["is_correct"].to_numpy()
    row_ids   = eval_df["row_id"].to_numpy()

    def compute_map3(eval_pred):
        logits, labels = eval_pred
        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
        top3 = np.argsort(-probs, axis=1)[:, :3]

        # Decode predictions
        pred_misconceptions = le.inverse_transform(top3.ravel())
        pred_misconceptions = pred_misconceptions.reshape(top3.shape)

        # Add True_/False_ prefix
        prefix = np.where(is_correct, "True_", "False_")
        # pred_full = np.char.add(prefix[:, None], pred_misconceptions) # it gives bug when environment numpy version is low
        pred_full = np.array([[ (p + m) for m in row ] for p, row in zip(prefix, pred_misconceptions)])


        # Compare
        match = (pred_full == gold_full[:, None])
        map3 = np.mean([
            1 if m[0] else 0.5 if m[1] else 1/3 if m[2] else 0
            for m in match
        ])

        return {"map@3": map3}

    return compute_map3
    

# --- Beta Schedules ---
def linear_beta(step, total_steps, start=1.0, end=0.6):
    frac = step / max(1, total_steps)
    return start + (end - start) * frac

def cosine_beta(step, total_steps, start=1.0, end=0.6):
    cos = (1 + math.cos(math.pi * step / total_steps)) / 2
    return end + (start - end) * cos

def step_beta(step, total_steps, warmup_frac=0.1, start=1.0, mid=0.7, end=0.3):
    warmup_steps = int(total_steps * warmup_frac)
    if step < warmup_steps:
        return start  # CE warmup
    else:
        frac = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
        return mid + (end - mid) * frac


# --- Bootstrap Trainer ---
class BootstrapTrainer(Trainer):
    def __init__(self, *args, beta_schedule="linear", start=1.0, mid=0.7, end=0.3, warmup_frac=0.1, top_3_focused=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta_schedule = beta_schedule
        self.start = start
        self.mid = mid
        self.end = end
        self.warmup_frac = warmup_frac
        self.top_3_focused = top_3_focused

    def compute_loss(self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        labels = inputs.pop("labels")
        # inputs.pop("is_correct", None)
        # inputs.pop("full_target", None)
        
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # --- Step + Schedule ---
        step = self.state.global_step
        total_steps = max(1, self.state.max_steps)

        if self.beta_schedule == "linear":
            beta = linear_beta(step, total_steps, start=self.start, end=self.end)
        elif self.beta_schedule == "cosine":
            beta = cosine_beta(step, total_steps, start=self.start, end=self.end)
        else:  # step decay
            beta = step_beta(step, total_steps, warmup_frac=self.warmup_frac,
                             start=self.start, mid=self.mid, end=self.end)

        # --- Bootstrapped Loss ---
        one_hot = torch.zeros_like(probs).scatter_(1, labels.unsqueeze(1), 1)

        if self.top_3_focused:
            # Get top 3 indices and probs
            topk_probs, topk_idx = torch.topk(probs, k=3, dim=-1)
            # Normalize so top 3 sum to 1
            topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
            # Build a zero tensor and scatter normalized top3
            focused_probs = torch.zeros_like(probs)
            focused_probs.scatter_(1, topk_idx, topk_probs)
            boot_targets = beta * one_hot + (1 - beta) * focused_probs.detach()
        else:
            boot_targets = beta * one_hot + (1 - beta) * probs.detach()
        
        # loss = -(boot_targets * torch.log(probs)).sum(dim=1).mean()
        loss = -(boot_targets * log_probs).sum(dim=1).mean()

        return (loss, outputs) if return_outputs else loss

class TxtLoggerCallback(TrainerCallback):
    def __init__(self, save_path):
        super().__init__()
        self.save_path = save_path
        with open(self.save_path, "w") as f:
            f.write("Step\ttrain/loss\teval/loss\teval/map@3\n")
        self.last_train_loss = ""     # cache latest train loss
        self.last_written_step = None # avoid dup writes per eval

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        step = state.global_step

        # Update cache on train logs (no write)
        if "loss" in logs:
            self.last_train_loss = logs["loss"]
            return

        # Write ONLY on evaluation logs
        is_eval_log = any(k.startswith("eval_") for k in logs)
        if not is_eval_log:
            return

        # Avoid duplicate writes at same step
        if step == self.last_written_step:
            return

        eval_loss = logs.get("eval_loss", "")
        # metric key may be literally "eval_map@3"
        eval_map3 = logs.get("eval_map@3", logs.get("eval_map3", ""))

        with open(self.save_path, "a") as f:
            f.write(f"{step}\t{self.last_train_loss}\t{eval_loss}\t{eval_map3}\n")
            f.flush()
        self.last_written_step = step

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ver", type=int, required=True, help="Version number (e.g. 54)")
    parser.add_argument("--model_name", type=str, required=True, help="Model name (e.g. Qwen/Qwen3-8B)")
    parser.add_argument("--max_len", type=int, default=300)
    parser.add_argument("--use_single_fold", action="store_true")
    parser.add_argument("--cv_fold", type=int, default=5)
    parser.add_argument("--cv_seed", type=int, default=42)

    # LoRA + training hyperparams
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr_scheduler", type=str, default="linear")
    parser.add_argument("--beta_scheduler", type=str, default="step")
    parser.add_argument("--start", type=float, default=1.0)
    parser.add_argument("--mid", type=float, default=0.7)
    parser.add_argument("--end", type=float, default=0.3)
    parser.add_argument("--warmup_frac", type=float, default=0.1)
    args = parser.parse_args()

    # Setup
    DIR = f"ver_{args.ver}"
    os.makedirs(DIR, exist_ok=True)
    MODEL_NAME = args.model_name
    MAX_LEN = args.max_len

    print(args)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        print(f"Missing tokenizer.pad_token")
        tokenizer.pad_token = tokenizer.eos_token

    train_df, le = load_and_preprocess_data()
    n_classes = train_df['label'].nunique()
    print(f"Total of {n_classes} classes.")
    train_df['text'] = train_df.apply(format_input, axis=1)

    skf = StratifiedKFold(n_splits=args.cv_fold, shuffle=True, random_state=args.cv_seed)
    fold_indices = list(skf.split(train_df, train_df['split_key']))
    if args.use_single_fold:
        fold_indices = [fold_indices[0]]

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    for fold, (tr_idx, va_idx) in enumerate(fold_indices):
        try:
            del model
            del trainer
        except NameError:
            pass

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        
        tr, va = train_df.iloc[tr_idx].copy(), train_df.iloc[va_idx].copy()

        n_classes_tr_fold = tr['label'].nunique()
        n_classes_va_fold = tr['label'].nunique()
        print(f"Fold {fold}: train classes = {n_classes_tr_fold}, val classes = {n_classes_va_fold}, total = {n_classes}")

        # Ensure train fold covers *all* classes
        if n_classes_tr_fold != n_classes:
            raise AssertionError(
                f"Train fold missing classes! Found {n_classes_tr_fold}, expected {n_classes}"
            )

        
        ds_tr = prepare_dataset(tr, tokenizer, MAX_LEN=MAX_LEN)
        ds_va = prepare_dataset(va, tokenizer, MAX_LEN=MAX_LEN)

        lora_config = LoraConfig(
            r=args.r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
            lora_dropout=args.dropout,
            bias="none",
            task_type=TaskType.SEQ_CLS,
            modules_to_save=["classifier", "score"],
            inference_mode=False,
        )

        model = load_base_model_bf16(MODEL_NAME, n_classes)
        model = get_peft_model(model, lora_config)

        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        output_fold_dir = f"{DIR}/fold_{fold}"
        os.makedirs(output_fold_dir, exist_ok=True)

        compute_metrics = make_compute_map3(va, le)

        training_args = TrainingArguments(
            output_dir=output_fold_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            eval_strategy="steps",
            save_strategy="steps",
            eval_steps=1/(10*args.epochs),
            save_steps=1/(10*args.epochs),
            save_total_limit=5,
            learning_rate=args.lr,
            metric_for_best_model="map@3",
            greater_is_better=True,
            # load_best_model_at_end=True,
            save_only_model=True,
            logging_dir=f"{DIR}/logs_fold_{fold}",
            logging_steps=1/(10*args.epochs),
            report_to=[],
            bf16=True,
            fp16=False,
            eval_accumulation_steps=1,
            gradient_accumulation_steps=16//8,
            lr_scheduler_type=args.lr_scheduler,
            remove_unused_columns=False,
        )

        results_path = f"{DIR}/fold_{fold}/results.txt"
        trainer = BootstrapTrainer(
            model=model,
            args=training_args,
            train_dataset=ds_tr,
            eval_dataset=ds_va,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            beta_schedule=args.beta_scheduler,
            start=args.start,
            mid=args.mid,
            end=args.end,
            warmup_frac=args.warmup_frac,
            callbacks=[TxtLoggerCallback(results_path)],
        )

        trainer.train()
        # final_map = trainer.evaluate()["eval_map@3"]
        # print(f"Fold {fold+1} eval/map@3 = {final_map:.6f}")

        # save_dir = f"{DIR}/fold_{fold}/best"
        # os.makedirs(save_dir, exist_ok=True)
        # trainer.save_model(save_dir)
        joblib.dump(le, f"{DIR}/fold_{fold}/label_encoder.joblib")

        # for ckpt in sorted(Path(f"{DIR}/fold_{fold}").glob("checkpoint-*")):
        #     shutil.rmtree(ckpt, ignore_errors=True)


if __name__ == "__main__":
    main()






