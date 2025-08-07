"""
フルトレーニングデータに対する推論スクリプト - 全確率値を(n, 65)のnumpy配列として出力
"""

import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import joblib
import torch
from tqdm import tqdm

# カスタムモジュールのインポート
from config import *
from utils import prepare_correct_answers, format_input, tokenize_dataset, preprocess_data


def main():
    """メイン推論関数 - フルトレーニングデータに対して推論を実行"""
    
    print("Loading trained model and tokenizer...")
    # モデルとトークナイザーの読み込み
    model = AutoModelForSequenceClassification.from_pretrained(BEST_MODEL_PATH, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(BEST_MODEL_PATH)
    
    print("Loading label encoder...")
    # ラベルエンコーダーの読み込み
    le = joblib.load(LABEL_ENCODER_PATH)
    num_classes = len(le.classes_)
    print(f"Number of classes: {num_classes}")
    
    print("Loading full training data...")
    # フルトレーニングデータの読み込み
    train = pd.read_csv(TRAIN_DATA_PATH)
    train.Misconception = train.Misconception.fillna('NA')
    
    # preprocess_data関数を適用
    print("Applying data preprocessing...")
    train = preprocess_data(train)
    
    print("Preparing correct answers...")
    # 正解答案データの準備
    correct = prepare_correct_answers(train)
    
    print("Preprocessing training data...")
    # トレーニングデータの前処理
    train_processed = train.merge(correct, on=['QuestionId','MC_Answer'], how='left')
    train_processed.is_correct = train_processed.is_correct.fillna(0)
    
    # FixedStudentExplanationを使用するようにformat_inputを適用
    train_for_format = train_processed.copy()
    train_for_format['StudentExplanation'] = train_processed['FixedStudentExplanation']
    train_processed['text'] = train_for_format.apply(format_input, axis=1)
    
    print(f"Total samples in training data: {len(train_processed)}")
    
    print("Tokenizing training data...")
    # トレーニングデータのトークナイズ
    ds_train = Dataset.from_pandas(train_processed[['text']])
    ds_train = tokenize_dataset(ds_train, tokenizer, MAX_LEN)
    
    print("Running inference on full training data...")
    # 推論の実行
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=TrainingArguments(
            output_dir="./tmp_valid",  # 一時ディレクトリ
            report_to="none",         # wandbを無効化
            per_device_eval_batch_size=64,  # バッチサイズの設定
            dataloader_num_workers=2,  # データローダーのワーカー数
            remove_unused_columns=False,
        )
    )
    
    # 推論を実行してロジットを取得
    predictions = trainer.predict(ds_train)
    logits = predictions.predictions
    
    print("Converting logits to probabilities...")
    # ロジットを確率に変換
    probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()
    
    # 結果の確認
    print(f"\nOutput shape: {probabilities.shape}")
    print(f"Expected shape: ({len(train_processed)}, {num_classes})")
    
    # numpy配列を保存
    output_path = f"{OUTPUT_DIR}/train_probabilities.npy"
    np.save(output_path, probabilities)
    print(f"\nProbabilities saved to: {output_path}")
    
    # 検証のための統計情報
    print("\nValidation statistics:")
    print(f"Min probability: {probabilities.min():.6f}")
    print(f"Max probability: {probabilities.max():.6f}")
    print(f"Mean probability: {probabilities.mean():.6f}")
    print(f"Sum of probabilities for first sample: {probabilities[0].sum():.6f}")
    
    # Top-3予測の例を表示
    top3_indices = np.argsort(-probabilities[0])[:3]
    top3_probs = probabilities[0][top3_indices]
    top3_labels = le.inverse_transform(top3_indices)
    
    print("\nExample predictions for first sample:")
    for i, (label, prob) in enumerate(zip(top3_labels, top3_probs)):
        print(f"  Top-{i+1}: {label} (probability: {prob:.4f})")
    
    return probabilities


if __name__ == "__main__":
    probs = main()
    print(f"\nFinal output shape: {probs.shape}")