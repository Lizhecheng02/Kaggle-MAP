"""
階層的分類アプローチのトレーニングスクリプト
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import (
    DebertaV2ForSequenceClassification,
    DebertaV2Tokenizer,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
import joblib
import torch

# カスタムモジュールのインポート
from config import *
from utils import *


def train_category_model(train_data, tokenizer):
    """Categoryを予測するモデルの学習"""
    print("\n=== Training Category Model ===")

    # Categoryのエンコード
    category_encoder = LabelEncoder()
    train_data['category_label'] = category_encoder.fit_transform(train_data['Category'])
    n_categories = len(category_encoder.classes_)
    print(f"Number of categories: {n_categories}")

    # 入力テキストのフォーマット
    train_data['text'] = train_data.apply(lambda x: format_input_hierarchical(x, task='category'), axis=1)

    # トレーニング/検証データの分割
    train_df, val_df = train_test_split(
        train_data[['text', 'category_label']],
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_SEED,
        stratify=train_data['category_label']
    )

    # データセットの作成
    train_dataset = Dataset.from_pandas(train_df.rename(columns={'category_label': 'label'}))
    val_dataset = Dataset.from_pandas(val_df.rename(columns={'category_label': 'label'}))

    # トークナイズ
    train_dataset = tokenize_dataset(train_dataset, tokenizer, MAX_LEN)
    val_dataset = tokenize_dataset(val_dataset, tokenizer, MAX_LEN)

    # モデルの初期化
    model = DebertaV2ForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=n_categories
    )

    # トレーニング引数
    training_args = TrainingArguments(
        output_dir=CATEGORY_MODEL_PATH,
        overwrite_output_dir=True,
        eval_strategy="steps",
        save_strategy="steps",
        learning_rate=LEARNING_RATE_CATEGORY,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=EPOCHS_CATEGORY,
        weight_decay=0.01,
        logging_steps=LOGGING_STEPS,
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        fp16=True,
        report_to="none"
    )

    # トレーナーの初期化
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_map3_category
    )

    # トレーニング
    trainer.train()

    # 最良モデルの保存
    trainer.save_model(CATEGORY_MODEL_PATH)
    joblib.dump(category_encoder, CATEGORY_ENCODER_PATH)

    return category_encoder


def train_misconception_model(train_data, category, tokenizer):
    """特定のCategoryのMisconceptionを予測するモデルの学習"""
    print(f"\n=== Training Misconception Model for {category} ===")

    # データの準備
    subset_data, misconception_encoder = prepare_misconception_data(train_data, category)
    n_misconceptions = len(misconception_encoder.classes_)
    print(f"Number of misconceptions for {category}: {n_misconceptions}")

    # 入力テキストのフォーマット
    subset_data['text'] = subset_data.apply(
        lambda x: format_input_hierarchical(x, task='misconception'),
        axis=1
    )

    # トレーニング/検証データの分割
    # 各クラスのサンプル数をチェック
    class_counts = subset_data['misconception_label'].value_counts()
    min_samples_per_class = class_counts.min()
    
    # 層化抽出が可能かチェック（各クラスに最低2サンプル必要）
    if min_samples_per_class >= 2:
        train_df, val_df = train_test_split(
            subset_data[['text', 'misconception_label']],
            test_size=VALIDATION_SPLIT,
            random_state=RANDOM_SEED,
            stratify=subset_data['misconception_label']
        )
    else:
        # 層化抽出できない場合は通常の分割
        train_df, val_df = train_test_split(
            subset_data[['text', 'misconception_label']],
            test_size=VALIDATION_SPLIT,
            random_state=RANDOM_SEED
        )

    # データセットの作成
    train_dataset = Dataset.from_pandas(train_df.rename(columns={'misconception_label': 'label'}))
    val_dataset = Dataset.from_pandas(val_df.rename(columns={'misconception_label': 'label'}))

    # トークナイズ
    train_dataset = tokenize_dataset(train_dataset, tokenizer, MAX_LEN)
    val_dataset = tokenize_dataset(val_dataset, tokenizer, MAX_LEN)

    # モデルの初期化
    model = DebertaV2ForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=n_misconceptions
    )

    # 出力ディレクトリ
    output_dir = TRUE_MISCONCEPTION_MODEL_PATH if category == 'True_Misconception' else FALSE_MISCONCEPTION_MODEL_PATH
    encoder_path = TRUE_MISCONCEPTION_ENCODER_PATH if category == 'True_Misconception' else FALSE_MISCONCEPTION_ENCODER_PATH

    # トレーニング引数
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        eval_strategy="steps",
        save_strategy="steps",
        learning_rate=LEARNING_RATE_MISCONCEPTION,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=EPOCHS_MISCONCEPTION,
        weight_decay=0.01,
        logging_steps=LOGGING_STEPS,
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="map@3",
        greater_is_better=True,
        save_total_limit=2,
        fp16=True,
        report_to="none"
    )

    # トレーナーの初期化
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_map3_misconception
    )

    # トレーニング
    trainer.train()

    # 最良モデルの保存
    trainer.save_model(output_dir)
    joblib.dump(misconception_encoder, encoder_path)

    return misconception_encoder


def main():
    """メイン関数"""

    # 出力ディレクトリの作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # データの読み込み
    print("Loading training data...")
    train = pd.read_csv(TRAIN_DATA_PATH)
    train['Misconception'] = train['Misconception'].fillna('NA')

    # 特徴量エンジニアリング
    print("Performing feature engineering...")
    correct = prepare_correct_answers(train)
    train = train.merge(correct, on=['QuestionId','MC_Answer'], how='left')
    train['is_correct'] = train['is_correct'].fillna(0)

    # トークナイザーの初期化
    print("Initializing tokenizer...")
    tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME)

    # 1. Categoryモデルの学習
    category_encoder = train_category_model(train, tokenizer)

    # 2. 各CategoryのMisconceptionモデルの学習
    misconception_encoders = {}
    for category in MISCONCEPTION_CATEGORIES:
        if len(train[train['Category'] == category]) > 0:
            encoder = train_misconception_model(train, category, tokenizer)
            misconception_encoders[category] = encoder

    print("\n=== Training completed successfully! ===")
    print(f"Models saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
