"""
Deberta モデルトレーニングスクリプト
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    DebertaV2ForSequenceClassification, 
    DebertaV2Tokenizer, 
    TrainingArguments, 
    Trainer,
    get_cosine_schedule_with_warmup
)
from datasets import Dataset
import joblib
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

# カスタムモジュールのインポート
from config import *
from utils import prepare_correct_answers, format_input, tokenize_dataset, compute_map3
from data_augmentation import DataAugmenter
from multitask_model import MultiTaskDebertaModel, FocalLoss


class CustomTrainer(Trainer):
    """カスタムTrainerクラス - Focal Lossとクラス重みをサポート"""
    
    def __init__(self, *args, class_weights=None, use_focal_loss=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.use_focal_loss = use_focal_loss
        
        if self.use_focal_loss:
            self.loss_fn = FocalLoss(alpha=self.class_weights, gamma=FOCAL_GAMMA)
        elif self.class_weights is not None:
            self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.loss_fn = nn.CrossEntropyLoss()
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        if labels is not None and self.loss_fn is not None:
            loss = self.loss_fn(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        
        return (loss, outputs) if return_outputs else loss


def prepare_auxiliary_labels(train_data):
    """マルチタスク学習用の補助ラベルを準備"""
    # 正解/不正解ラベル
    train_data['correctness_label'] = train_data['Category'].apply(
        lambda x: 0 if x.startswith('False') else 1
    )
    
    # カテゴリラベル (Correct/Misconception/Neither)
    category_map = {
        'Correct': 0,
        'Misconception': 1,
        'Neither': 2
    }
    train_data['category_label'] = train_data['Category'].apply(
        lambda x: category_map[x.split('_')[1]]
    )
    
    # Misconceptionラベル
    misconception_encoder = LabelEncoder()
    misconceptions = train_data[train_data['Misconception'] != 'NA']['Misconception']
    misconception_encoder.fit(misconceptions)
    
    train_data['misconception_label'] = train_data['Misconception'].apply(
        lambda x: misconception_encoder.transform([x])[0] if x != 'NA' else -1
    )
    
    return train_data, misconception_encoder


def tokenize_dataset_multitask(dataset, tokenizer, max_len):
    """マルチタスク用のデータセットトークナイズ"""
    def tokenize(batch):
        tokens = tokenizer(batch['text'], padding='max_length', truncation=True, max_length=max_len)
        # 補助ラベルも含める
        if 'correctness_label' in batch:
            tokens['correctness_labels'] = batch['correctness_label']
        if 'category_label' in batch:
            tokens['category_labels'] = batch['category_label']
        if 'misconception_label' in batch:
            tokens['misconception_labels'] = batch['misconception_label']
        return tokens
    
    dataset = dataset.map(tokenize, batched=True)
    columns = ['input_ids', 'attention_mask', 'label']
    
    if USE_MULTITASK:
        columns.extend(['correctness_labels', 'category_labels', 'misconception_labels'])
    
    dataset.set_format(type='torch', columns=columns)
    return dataset


def main():
    """メイントレーニング関数"""
    
    # 出力ディレクトリの作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- データの読み込みと前処理 ---
    print("Loading and preprocessing training data...")
    le = LabelEncoder()
    train = pd.read_csv(TRAIN_DATA_PATH)
    train.Misconception = train.Misconception.fillna('NA')
    train['target'] = train.Category + ":" + train.Misconception
    train['label'] = le.fit_transform(train['target'])
    n_classes = len(le.classes_)
    print(f"Train shape: {train.shape} with {n_classes} target classes")
    
    # --- データ拡張 ---
    if USE_DATA_AUGMENTATION:
        print("Applying data augmentation...")
        augmenter = DataAugmenter(train)
        train = augmenter.apply_all_augmentations()
        print(f"Train shape after augmentation: {train.shape}")
    
    # --- 補助ラベルの準備 ---
    misconception_encoder = None
    if USE_MULTITASK:
        print("Preparing auxiliary labels for multi-task learning...")
        train, misconception_encoder = prepare_auxiliary_labels(train)
        joblib.dump(
            {'misconception_encoder': misconception_encoder},
            AUXILIARY_ENCODERS_PATH
        )
    
    # --- 特徴量エンジニアリング ---
    print("Performing feature engineering...")
    correct = prepare_correct_answers(train)
    train = train.merge(correct, on=['QuestionId','MC_Answer'], how='left')
    train.is_correct = train.is_correct.fillna(0)
    
    # --- 入力テキストのフォーマット ---
    print("Formatting input text...")
    train['text'] = train.apply(format_input, axis=1)
    print("Example prompt for our LLM:")
    print(train.text.values[0])
    
    # --- クラス重みの計算 ---
    class_weights = None
    if USE_CLASS_WEIGHTS:
        print("Computing class weights...")
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(train['label']),
            y=train['label']
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    # --- トークナイザーの初期化 ---
    print("Initializing tokenizer...")
    tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME)
    
    # --- トークン長の分析 ---
    print("Analyzing token lengths...")
    lengths = [len(tokenizer.encode(t, truncation=False)) for t in train['text']]
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50)
    plt.title("Token Length Distribution")
    plt.xlabel("Number of tokens")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(f'{OUTPUT_DIR}/token_length_distribution.png')
    plt.close()
    
    over_limit = (np.array(lengths) > MAX_LEN).sum()
    print(f"There are {over_limit} train sample(s) with more than {MAX_LEN} tokens")
    
    # --- データの分割 ---
    print("Splitting data into train and validation sets...")
    
    # 各クラスのサンプル数を確認
    label_counts = train['label'].value_counts()
    min_samples_in_class = label_counts.min()
    
    if min_samples_in_class < 2:
        print(f"Warning: Some classes have only {min_samples_in_class} sample(s).")
        print("Using simple random split instead of stratified split.")
        train_df, val_df = train_test_split(
            train, test_size=VALIDATION_SPLIT, 
            random_state=RANDOM_SEED, stratify=None
        )
    else:
        train_df, val_df = train_test_split(
            train, test_size=VALIDATION_SPLIT, 
            random_state=RANDOM_SEED, stratify=train['label']
        )
    
    # データセットの準備
    COLS = ['text', 'label']
    if USE_MULTITASK:
        COLS.extend(['correctness_label', 'category_label', 'misconception_label'])
    
    train_ds = Dataset.from_pandas(train_df[COLS])
    val_ds = Dataset.from_pandas(val_df[COLS])
    
    # --- データセットのトークナイズ ---
    print("Tokenizing datasets...")
    if USE_MULTITASK:
        train_ds = tokenize_dataset_multitask(train_ds, tokenizer, MAX_LEN)
        val_ds = tokenize_dataset_multitask(val_ds, tokenizer, MAX_LEN)
    else:
        train_ds = tokenize_dataset(train_ds, tokenizer, MAX_LEN)
        val_ds = tokenize_dataset(val_ds, tokenizer, MAX_LEN)
    
    # --- モデルの初期化 ---
    print("Initializing model...")
    if USE_MULTITASK:
        num_misconceptions = len(misconception_encoder.classes_) if misconception_encoder else 36
        model = MultiTaskDebertaModel.from_pretrained(
            MODEL_NAME, 
            num_labels=n_classes,
            num_misconceptions=num_misconceptions
        )
    else:
        model = DebertaV2ForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=n_classes
        )
    
    # --- トレーニング引数の設定 ---
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        save_strategy="steps",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        save_total_limit=1,
        metric_for_best_model="map@3",
        greater_is_better=True,
        load_best_model_at_end=True,
        report_to="none",
        label_smoothing_factor=LABEL_SMOOTHING if LABEL_SMOOTHING > 0 else 0,
        lr_scheduler_type=SCHEDULER_TYPE,
        dataloader_num_workers=4,
        fp16=True,  # 混合精度トレーニング
    )
    
    # --- トレーナーのセットアップとトレーニング ---
    print("Setting up trainer...")
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_map3,
        class_weights=class_weights,
        use_focal_loss=USE_FOCAL_LOSS,
    )
    
    print("Starting training...")
    trainer.train()
    
    # --- 最終的なMAP@3スコアを表示 ---
    print("\nEvaluating on validation set...")
    eval_results = trainer.evaluate()
    print(f"\nValidation MAP@3: {eval_results.get('eval_map@3', 'N/A'):.4f}")
    
    # --- モデルとエンコーダーの保存 ---
    print("\nSaving model and label encoder...")
    trainer.save_model(BEST_MODEL_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)
    
    print("Training completed successfully!")
    print(f"Model saved to: {BEST_MODEL_PATH}")
    print(f"Label encoder saved to: {LABEL_ENCODER_PATH}")


if __name__ == "__main__":
    main()