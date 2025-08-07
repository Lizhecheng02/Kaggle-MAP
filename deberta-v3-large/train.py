#!/usr/bin/env python3
"""
train.py: モデルの学習を行い、学習済みモデルとラベルエンコーダを保存するスクリプト
"""
import os
import joblib
import torch
import numpy as np
from transformers import AutoTokenizer, Trainer, TrainingArguments, DebertaV2ForSequenceClassification
from sklearn.model_selection import train_test_split
from datasets import Dataset

import config
from utils import (
    load_data,
    feature_engineer,
    prepare_correct_df,
    encode_labels,
    format_input,
    tokenize,
    compute_map3,
)


def main():
    # 環境変数の設定
    os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES
    # 出力ディレクトリ作成
    os.makedirs(config.DIR, exist_ok=True)
    # モデル名と学習エポック数
    model_name = config.MODEL_NAME
    EPOCHS = config.EPOCHS

    # データ読み込みと前処理
    train = load_data(config.TRAIN_CSV_PATH)
    train = feature_engineer(train)

    # 正解ラベルの準備
    correct = prepare_correct_df(train)
    train = train.merge(correct, on=['QuestionId', 'MC_Answer'], how='left')
    train['is_correct'] = train['is_correct'].fillna(0)

    # ターゲットエンコード
    train, le = encode_labels(train)

    # テキスト整形
    train['text'] = train.apply(format_input, axis=1)

    # 訓練/評価データ分割
    train_df, val_df = train_test_split(train, test_size=0.05, random_state=42)
    COLS = ['text', 'label']
    train_ds = Dataset.from_pandas(train_df[COLS])
    val_ds = Dataset.from_pandas(val_df[COLS])

    # トークナイザー
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds = train_ds.map(lambda batch: tokenize(batch, tokenizer), batched=True)
    val_ds = val_ds.map(lambda batch: tokenize(batch, tokenizer), batched=True)

    # PyTorchフォーマット設定
    columns = ['input_ids', 'attention_mask', 'label']
    train_ds.set_format(type='torch', columns=columns)
    val_ds.set_format(type='torch', columns=columns)

    # モデル準備
    num_labels = len(le.classes_)
    model = DebertaV2ForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )

    # 学習設定
    training_args = TrainingArguments(
        output_dir=config.DIR,
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        save_strategy="steps",
        num_train_epochs=config.EPOCHS,
        per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.LEARNING_RATE,
        logging_dir=config.LOGGING_DIR,
        logging_steps=config.LOGGING_STEPS,
        save_steps=config.SAVE_STEPS,
        eval_steps=config.EVAL_STEPS,
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        metric_for_best_model=config.METRIC_FOR_BEST_MODEL,
        greater_is_better=True,
        load_best_model_at_end=True,
        report_to="none",
    )

    # Trainer設定
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_map3,
    )

    # 学習実行
    trainer.train()

    # モデルとラベルエンコーダ保存
    # モデルとラベルエンコーダ保存
    trainer.save_model(f"{config.DIR}/best")
    joblib.dump(le, f"{config.DIR}/label_encoder.joblib")


if __name__ == '__main__':
    main()
