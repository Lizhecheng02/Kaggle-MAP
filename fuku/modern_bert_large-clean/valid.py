"""
全訓練データに対する推論スクリプト
20%検証用モデルを使用して訓練データ全体の確率を取得
"""

import os
import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
import joblib
from tqdm import tqdm

# カスタムモジュールのインポート
from config import *
from utils import prepare_correct_answers, format_input, tokenize_dataset, preprocess_data


def get_predictions_with_probs(model, dataloader, device):
    """モデルから全クラスの確率を取得"""
    model.eval()
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inferencing"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'}
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy())

    return np.vstack(all_probs)


def main():
    """メイン推論関数"""

    # デバイスの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # モデルパスの設定（20%検証用モデルを使用）
    model_path = BEST_MODEL_PATH
    label_encoder_path = LABEL_ENCODER_PATH

    # モデルとラベルエンコーダーが存在するか確認
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using train.py")
        return

    if not os.path.exists(label_encoder_path):
        print(f"Error: Label encoder not found at {label_encoder_path}")
        print("Please train the model first using train.py")
        return

    # --- データの読み込みと前処理 ---
    print("Loading and preprocessing training data...")
    train = pd.read_csv(TRAIN_DATA_PATH)
    train = preprocess_data(train)  # preprocess_data関数を使用
    train['target'] = train.Category + ":" + train.Misconception

    # ラベルエンコーダーの読み込み
    le = joblib.load(label_encoder_path)
    train['label'] = le.transform(train['target'])
    n_classes = len(le.classes_)
    print(f"Train shape: {train.shape} with {n_classes} target classes")

    # --- 特徴量エンジニアリング ---
    print("Performing feature engineering...")
    correct = prepare_correct_answers(train)
    train = train.merge(correct, on=['QuestionId','MC_Answer'], how='left')
    train.is_correct = train.is_correct.fillna(0)

    # --- 入力テキストのフォーマット ---
    print("Formatting input text...")
    train['text'] = train.apply(format_input, axis=1)

    # --- データセットの準備 ---
    print("Preparing dataset...")
    COLS = ['text', 'label']
    train_ds = Dataset.from_pandas(train[COLS])

    # --- トークナイザーとモデルの読み込み ---
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=n_classes
    ).to(device)

    # --- データセットのトークナイズ ---
    print("Tokenizing dataset...")
    train_ds = tokenize_dataset(train_ds, tokenizer, MAX_LEN)

    # --- データローダーの作成 ---
    train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    dataloader = DataLoader(
        train_ds,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False
    )

    # --- 推論の実行 ---
    print(f"\nRunning inference on {len(train)} samples...")
    probabilities = get_predictions_with_probs(model, dataloader, device)

    # --- 結果の保存 ---
    output_path = os.path.join(OUTPUT_DIR, 'train_probabilities.npy')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(output_path, probabilities)

    print(f"\nInference completed!")
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Expected shape: ({len(train)}, {n_classes})")
    print(f"Saved to: {output_path}")

    # 確認のため最初の数行の確率を表示
    print("\nSample probabilities (first 3 rows, first 5 classes):")
    print(probabilities[:3, :5])
    print(f"\nSum of probabilities for first row: {probabilities[0].sum():.6f}")


if __name__ == "__main__":
    main()
