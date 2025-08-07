"""
QuestionID 31772のデータに対してモデルの推論を実行し、MAP@3を計測するスクリプト
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from datasets import Dataset
from peft import PeftModel
import joblib
from tqdm import tqdm

# 設定ファイルをインポート
from config import *
from utils import format_input, prepare_correct_answers, compute_map3

def tokenize_function(examples, tokenizer, max_len):
    """トークナイズ関数"""
    return tokenizer(
        examples['text'],
        padding=False,
        truncation=True,
        max_length=max_len,
        return_tensors=None
    )

def load_model_and_tokenizer():
    """モデルとトークナイザーをロード"""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # パディングトークンの設定
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print("Loading label encoder...")
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    n_classes = len(label_encoder.classes_)
    
    print(f"Loading model with {n_classes} classes...")
    # ベースモデルのロード
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=n_classes,
        trust_remote_code=True
    )
    
    # LoRAアダプターを適用
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, BEST_MODEL_PATH)
    
    # モデルの設定を更新してパディングトークンを認識させる
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # モデルを評価モードに設定
    model.eval()
    
    # GPUが利用可能な場合は使用
    if torch.cuda.is_available():
        model = model.cuda()
        print("Using GPU for inference")
    else:
        print("Using CPU for inference")
    
    return model, tokenizer, label_encoder

def prepare_question_31772_data():
    """QuestionID 31772のデータを準備"""
    print("Loading training data...")
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    
    # QuestionID 31772のデータのみを抽出
    question_df = train_df[train_df['QuestionId'] == 31772].copy()
    print(f"Found {len(question_df)} samples for QuestionID 31772")
    
    # train.pyと同じ前処理を適用
    print("Performing feature engineering...")
    correct = prepare_correct_answers(train_df)
    question_df = question_df.merge(correct, on=['QuestionId','MC_Answer'], how='left')
    question_df['is_correct'] = question_df['is_correct'].fillna(0)
    
    # プロンプトを作成
    question_df['text'] = question_df.apply(format_input, axis=1)
    
    # ラベルエンコーディング
    # train.pyと同じ方法でtargetカラムを作成
    question_df['Misconception'] = question_df['Misconception'].fillna('NA')
    question_df['target'] = question_df['Category'] + ":" + question_df['Misconception']
    
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    question_df['label'] = label_encoder.transform(question_df['target'])
    
    # データセットに変換
    dataset = Dataset.from_pandas(question_df[['text', 'label']])
    
    return question_df, dataset

def inference_and_calculate_map3(model, tokenizer, dataset, label_encoder):
    """推論を実行してMAP@3を計算"""
    # データコレーターの設定
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='pt')
    
    # トークナイズ
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, MAX_LEN),
        batched=True,
        remove_columns=['text']  # labelは保持
    )
    
    # Trainerを使用した推論
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        data_collator=data_collator,
        args=TrainingArguments(
            output_dir="./tmp_valid_31772",
            per_device_eval_batch_size=16,
            fp16=True,
            dataloader_num_workers=2,
            remove_unused_columns=True,
        )
    )
    
    print("Running inference...")
    predictions = trainer.predict(tokenized_dataset)
    
    # MAP@3を計算
    map3_result = compute_map3((predictions.predictions, predictions.label_ids))
    
    # ロジットから確率に変換
    logits = predictions.predictions
    probabilities = torch.nn.functional.softmax(torch.from_numpy(logits), dim=-1).numpy()
    
    # 上位3クラスを取得
    top3_indices = np.argsort(-probabilities, axis=1)[:, :3]
    
    return map3_result['map@3'], probabilities, top3_indices, predictions.label_ids

def main():
    """メイン処理"""
    # モデルとトークナイザーをロード
    model, tokenizer, label_encoder = load_model_and_tokenizer()
    
    # QuestionID 31772のデータを準備
    question_df, dataset = prepare_question_31772_data()
    
    # 推論実行とMAP@3計算
    try:
        map3_score, probabilities, top3_indices, true_labels = inference_and_calculate_map3(
            model, tokenizer, dataset, label_encoder
        )
    except torch.cuda.OutOfMemoryError:
        print("CUDA out of memory. Reducing batch size...")
        torch.cuda.empty_cache()
        # バッチサイズを小さくして再実行
        trainer = Trainer(
            model=model,
            processing_class=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='pt'),
            args=TrainingArguments(
                output_dir="./tmp_valid_31772",
                per_device_eval_batch_size=8,
                fp16=True,
                dataloader_num_workers=2,
                remove_unused_columns=True,
            )
        )
        tokenized_dataset = dataset.map(
            lambda x: tokenize_function(x, tokenizer, MAX_LEN),
            batched=True,
            remove_columns=['text']
        )
        predictions = trainer.predict(tokenized_dataset)
        map3_result = compute_map3((predictions.predictions, predictions.label_ids))
        map3_score = map3_result['map@3']
        probabilities = torch.nn.functional.softmax(torch.from_numpy(predictions.predictions), dim=-1).numpy()
        top3_indices = np.argsort(-probabilities, axis=1)[:, :3]
        true_labels = predictions.label_ids
    
    # 結果の表示
    print(f"\n=== QuestionID 31772 MAP@3 Score ===")
    print(f"MAP@3: {map3_score:.4f}")
    print(f"Total samples: {len(question_df)}")
    
    # 各サンプルの詳細を表示
    print("\n=== Detailed Predictions ===")
    for i in range(len(probabilities)):
        true_label_idx = true_labels[i]
        true_label = label_encoder.inverse_transform([true_label_idx])[0]
        
        top3_labels = label_encoder.inverse_transform(top3_indices[i])
        top3_probs = probabilities[i][top3_indices[i]]
        
        # MAP@3への寄与を計算
        contribution = 0.0
        position = None
        for j, idx in enumerate(top3_indices[i]):
            if idx == true_label_idx:
                contribution = 1.0 / (j + 1)
                position = j + 1
                break
        
        print(f"\nSample {i+1}:")
        print(f"  True label: {true_label}")
        print(f"  Predictions:")
        for j, (label, prob) in enumerate(zip(top3_labels, top3_probs)):
            marker = " <-- TRUE" if label == true_label else ""
            print(f"    {j+1}. {label}: {prob:.4f}{marker}")
        print(f"  MAP@3 contribution: {contribution:.4f}")
        if position:
            print(f"  (True label found at position {position})")
    
    # サマリー統計
    print("\n=== Summary Statistics ===")
    correct_at_1 = sum(1 for i in range(len(true_labels)) if top3_indices[i][0] == true_labels[i])
    correct_at_2 = sum(1 for i in range(len(true_labels)) if top3_indices[i][1] == true_labels[i])
    correct_at_3 = sum(1 for i in range(len(true_labels)) if top3_indices[i][2] == true_labels[i])
    
    print(f"Correct at position 1: {correct_at_1}/{len(true_labels)} ({correct_at_1/len(true_labels)*100:.1f}%)")
    print(f"Correct at position 2: {correct_at_2}/{len(true_labels)} ({correct_at_2/len(true_labels)*100:.1f}%)")
    print(f"Correct at position 3: {correct_at_3}/{len(true_labels)} ({correct_at_3/len(true_labels)*100:.1f}%)")
    
    # カテゴリごとの分布
    print("\n=== Category Distribution ===")
    category_counts = question_df['Category'].value_counts()
    for category, count in category_counts.items():
        print(f"{category}: {count} samples")
    
    # メモリ使用量の情報
    if torch.cuda.is_available():
        print(f"\nGPU memory used: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    
    return map3_score

if __name__ == "__main__":
    map3_score = main()