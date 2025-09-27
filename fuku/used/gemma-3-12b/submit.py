"""
LLM モデル（Gemma / Phi 等）推論スクリプト - 提出用予測ファイルの生成
学習時と同一の設定（config.py）と LoRA アダプターを用いて推論します。
"""

import os
import gc
import joblib
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

# プロジェクト設定・ユーティリティ
from config import *
from utils import (
    prepare_correct_answers,
    format_input,
    tokenize_dataset,
    create_submission,
)
from data_collator import DataCollatorWithPadding as CustomDataCollator

# PEFT（LoRA）
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False
    print("Warning: peft not available; inference will use base model only.")


def load_model_and_tokenizer(n_classes: int):
    """学習時と同じベースモデルをロードし、LoRA アダプターを適用して返す"""
    # トークナイザー
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # attention 実装 / dtype
    attn_impl = globals().get("ATTENTION_IMPLEMENTATION", None)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base_model_kwargs = {
        "trust_remote_code": True,
        "device_map": None,
        "torch_dtype": dtype,
    }
    if attn_impl:
        base_model_kwargs["attn_implementation"] = attn_impl

    # 分類モデル: Gemma3ForCausalLM + 線形分類ヘッド（学習時と同様）
    from train import LLMForSequenceClassification  # 再利用
    model = LLMForSequenceClassification(
        MODEL_NAME,
        n_classes,
        attn_implementation=attn_impl,
        torch_dtype=dtype,
    )
    try:
        model.config.pad_token_id = tokenizer.pad_token_id
    except Exception:
        pass

    # LoRA アダプター
    if PEFT_AVAILABLE and os.path.isdir(BEST_MODEL_PATH):
        print(f"Loading LoRA adapter from: {BEST_MODEL_PATH}")
        model = PeftModel.from_pretrained(model, BEST_MODEL_PATH, is_trainable=False)
    else:
        print("PEFT not available or adapter path missing; using base model only.")

    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model, tokenizer


def main():
    # クリーンアップ
    torch.cuda.empty_cache()
    gc.collect()

    print("Loading label encoder...")
    le = joblib.load(LABEL_ENCODER_PATH)
    n_classes = len(le.classes_)

    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(n_classes)

    print("Loading test and preparing features...")
    test = pd.read_csv(TEST_DATA_PATH)
    train = pd.read_csv(TRAIN_DATA_PATH)
    train.Misconception = train.Misconception.fillna('NA')
    correct = prepare_correct_answers(train)
    test = test.merge(correct, on=['QuestionId', 'MC_Answer'], how='left')
    test.is_correct = test.is_correct.fillna(0)
    test['text'] = test.apply(format_input, axis=1)

    print("Tokenizing test data...")
    ds_test = Dataset.from_pandas(test[['text']])
    ds_test = tokenize_dataset(ds_test, tokenizer, MAX_LEN)

    collator = CustomDataCollator(tokenizer=tokenizer, max_length=MAX_LEN)
    dataloader = DataLoader(ds_test, batch_size=EVAL_BATCH_SIZE, collate_fn=collator, shuffle=False)

    print("Running inference...")
    preds = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="inference"):
            device = next(model.parameters()).device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs.logits
            preds.append(logits.float().cpu().numpy())

    preds = np.vstack(preds)

    print("Creating submission...")
    from transformers.trainer_utils import EvalPrediction
    prediction_obj = EvalPrediction(predictions=preds, label_ids=None)
    submission = create_submission(prediction_obj, test, le)
    submission.to_csv(SUBMISSION_OUTPUT_PATH, index=False)

    print(f"Saved submission to: {SUBMISSION_OUTPUT_PATH}")
    print(submission.head())


if __name__ == "__main__":
    main()
